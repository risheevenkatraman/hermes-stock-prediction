"""Verify prospective snapshots and assemble version-aware observation histories."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import exchange_calendars as xcals

from backend.news import utc_timestamp
from benchmarks.news_collect import prepare
from benchmarks.news_data import SYMBOLS


def verified_run(path: Path):
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported schema: {path.name}")
    files = manifest["files"]
    if "report.json" not in files:
        raise ValueError("Report absent from manifest")
    for name, expected in files.items():
        target = (path / name).resolve()
        if target.parent != path.resolve() or not target.is_file():
            raise ValueError("Invalid manifest file path")
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Snapshot hash mismatch: {path.name}/{name}")
    report = json.loads((path / "report.json").read_text())
    if report.get("mode") != "prospective_observation_only":
        raise ValueError("Only prospective observations may be assembled")
    if not report.get("collection_complete"):
        return [], "partial_collection"
    requested = report["requested_tickers"]
    requests = report["requests"]
    if len(set(requested)) != len(requested) or sorted(
        r["ticker"] for r in requests
    ) != sorted(requested):
        raise ValueError("Request list mismatch")
    result = []
    for request in requests:
        ticker = request["ticker"]
        if ticker not in SYMBOLS or request["status"] != "received":
            raise ValueError("Invalid completed request")
        response_name, records_name = (
            ticker + "-response.json",
            ticker + "-records.json",
        )
        if response_name not in files or records_name not in files:
            raise ValueError("Response or records absent from manifest")
        response = json.loads((path / response_name).read_text())
        if response["observed_at"] != request["observed_at"]:
            raise ValueError("Receipt timestamp mismatch")
        if utc_timestamp(response["observed_at"]) < utc_timestamp(
            report["publication_window_end_exclusive"]
        ):
            raise ValueError("Receipt precedes requested window end")
        records, stats = prepare(
            response["payload"],
            ticker,
            response["observed_at"],
            report["publication_window_start"],
            report["publication_window_end_exclusive"],
        )
        if stats["response_at_cap"] or any(
            request.get(k) != v for k, v in stats.items()
        ):
            raise ValueError("Provider counts mismatch or capped response")
        if records != json.loads((path / records_name).read_text()):
            raise ValueError("Saved records disagree with response reconstruction")
        result.extend({**r, "snapshot": path.name} for r in records)
    return result, None


def assemble_events(records):
    events = {}
    versions = {}
    for record in sorted(
        records,
        key=lambda r: (
            utc_timestamp(r["article"]["available_at"]),
            r["snapshot"],
            r["raw_feed_index"],
        ),
    ):
        # Repeated rows within one response must not replace its first version.
        if "duplicate_url_in_response" in record["quality_flags"]:
            continue
        identity, version = record["identity"], record["version_sha256"]
        observed = record["article"]["available_at"]
        key = (identity, observed)
        if key in events:
            if events[key]["version_sha256"] != version:
                raise ValueError("Conflicting versions at identical receipt time")
            continue
        events[key] = record
        version_key = (identity, version)
        if version_key not in versions:
            versions[version_key] = {
                "identity": identity,
                "version_sha256": version,
                "first_observed_at": observed,
                "last_observed_at": observed,
                "observations": 0,
            }
        versions[version_key]["last_observed_at"] = observed
        versions[version_key]["observations"] += 1
    return list(events.values()), list(versions.values())


def at_cutoff(events, cutoff):
    """Latest observed version per URL; filter quality only AFTER version selection."""
    cutoff = utc_timestamp(cutoff)
    latest = {}
    for record in sorted(
        events, key=lambda r: utc_timestamp(r["article"]["available_at"])
    ):
        if utc_timestamp(record["article"]["available_at"]) <= cutoff:
            latest[record["identity"]] = record
    selected = []
    headlines = set()
    for record in latest.values():
        article = record["article"]
        published = utc_timestamp(article["published_at"])
        if (
            record["disposition"] != "candidate"
            or not cutoff.to_pydatetime() - timedelta(days=3)
            < published.to_pydatetime()
            <= cutoff.to_pydatetime()
        ):
            continue
        headline = (article["ticker"], " ".join(article["title"].lower().split()))
        if headline not in headlines:
            headlines.add(headline)
            selected.append(record)
    return selected


def build(root: Path, output: Path):
    records, included, excluded = [], [], []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if not (path / "manifest.json").exists():
            excluded.append({"snapshot": path.name, "reason": "missing_manifest"})
            continue
        rows, reason = verified_run(path)
        if reason:
            excluded.append({"snapshot": path.name, "reason": reason})
            continue
        included.append(
            {
                "snapshot": path.name,
                "manifest_sha256": hashlib.sha256(
                    (path / "manifest.json").read_bytes()
                ).hexdigest(),
            }
        )
        records.extend(rows)
    events, versions = assemble_events(records)
    daily = Counter(
        (
            utc_timestamp(r["article"]["available_at"]).date().isoformat(),
            r["article"]["ticker"],
            r["disposition"],
        )
        for r in events
    )
    coverage = []
    if events:
        first = min(utc_timestamp(r["article"]["available_at"]) for r in events)
        last = max(utc_timestamp(r["article"]["available_at"]) for r in events)
        cal = xcals.get_calendar(
            "XNYS",
            start=first.date() - timedelta(days=7),
            end=last.date() + timedelta(days=7),
        )
        for date, session in cal.schedule.iterrows():
            if first.date() <= date.date() <= last.date() and session["close"] <= last:
                eligible = Counter(
                    r["article"]["ticker"]
                    for r in at_cutoff(events, session["close"].isoformat())
                )
                for ticker in SYMBOLS:
                    coverage.append(
                        {
                            "date": date.date().isoformat(),
                            "ticker": ticker,
                            "close_utc": session["close"].isoformat(),
                            "eligible_articles": eligible[ticker],
                            "collection_started_by_close": first <= session["close"],
                        }
                    )
    report = {
        "input_snapshots": included,
        "excluded_snapshots": excluded,
        "input_records": len(records),
        "observation_events": len(events),
        "unique_versions": len(versions),
        "unique_ticker_urls": len({r["identity"] for r in events}),
        "training_ready": False,
        "policy": "No outcome data. Missing observations are not proof of no news. Quality policy v1; replay requires matching policy. Partial runs excluded in full.",
    }
    run = output / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8]
    )
    run.mkdir(parents=True, exist_ok=False)
    artifacts = {
        "events.json": events,
        "versions.json": versions,
        "daily-observations.json": [
            {"utc_date": d, "ticker": t, "disposition": q, "observations": n}
            for (d, t, q), n in sorted(daily.items())
        ],
        "session-coverage.json": coverage,
        "report.json": report,
    }
    for name, value in artifacts.items():
        (run / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
    for source in (
        Path(__file__),
        Path("benchmarks/news_collect.py"),
        Path("backend/news.py"),
    ):
        (run / source.name).write_bytes(source.read_bytes())
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now(UTC).isoformat(),
                "files": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(run.iterdir())
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root", type=Path, default=Path("data/news-observations")
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/news-assembled"))
    args = parser.parse_args()
    print(build(args.input_root, args.output_root))


if __name__ == "__main__":
    main()


"""Run a collection/assembly pass and assess observation coverage without outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import exchange_calendars as xcals

from backend.news import NewsProviderError, utc_timestamp
from benchmarks import news_assemble, news_collect
from benchmarks.news_data import safe_provider_reason

DEFAULT_PROTOCOL = Path("benchmarks/protocols/news_prospective_v1.json")


def verify_files(root: Path):
    manifest = json.loads((root / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        target = (root / name).resolve()
        if target.parent != root.resolve() or not target.is_file():
            raise ValueError("Invalid artifact path")
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError("Artifact hash mismatch")
    return manifest


def assess(events, receipts, protocol, as_of):
    as_of = utc_timestamp(as_of)
    start = utc_timestamp(protocol["eligible_origin_start"] + "T00:00:00Z")
    phase_sizes = {
        "fit": protocol["fit_sessions"],
        "validation": protocol["validation_sessions"],
        "assessment": protocol["assessment_sessions"],
    }
    if start.date().isoformat() <= protocol["protected_outcome_window"]["end"]:
        raise ValueError("News experiment overlaps the protected outcome window")
    if any(not isinstance(n, int) or n < 1 for n in phase_sizes.values()):
        raise ValueError("Phase lengths must be positive integers")
    phases = {name: [] for name in phase_sizes}
    session_rows = []
    if as_of >= start:
        cal = xcals.get_calendar(
            protocol["calendar"],
            start=start.date(),
            end=as_of.date() + timedelta(days=15),
        )
        schedule = cal.schedule
        phase_index = 0
        next_allowed = 0
        names = list(phases)
        for index, (date, row) in enumerate(schedule.iterrows()):
            close = row["close"]
            if close > as_of:
                break
            successful = set()
            for receipt in receipts:
                observed = utc_timestamp(receipt["observed_at"])
                if (
                    close.to_pydatetime()
                    - timedelta(hours=protocol["collection_freshness_hours"])
                    <= observed.to_pydatetime()
                    <= close.to_pydatetime()
                ):
                    successful.add(receipt["ticker"])
            candidates = Counter(
                r["article"]["ticker"]
                for r in news_assemble.at_cutoff(events, close.isoformat())
            )
            qualifies = set(protocol["symbols"]) <= successful
            entry = {
                "date": date.date().isoformat(),
                "close_utc": close.isoformat(),
                "qualifies": qualifies,
                "missing_poll_tickers": sorted(set(protocol["symbols"]) - successful),
                "candidate_counts": {s: candidates[s] for s in protocol["symbols"]},
            }
            session_rows.append(entry)
            if not qualifies or phase_index >= len(names) or index < next_allowed:
                continue
            name = names[phase_index]
            phases[name].append(entry)
            if len(phases[name]) == phase_sizes[name]:
                phase_index += 1
                next_allowed = index + protocol["boundary_embargo_sessions"] + 1
    phase_reports = {}
    for name, entries in phases.items():
        fractions = {
            ticker: sum(r["candidate_counts"][ticker] > 0 for r in entries)
            / len(entries)
            if entries
            else 0.0
            for ticker in protocol["symbols"]
        }
        phase_reports[name] = {
            "sessions": len(entries),
            "required_sessions": phase_sizes[name],
            "first_date": entries[0]["date"] if entries else None,
            "last_date": entries[-1]["date"] if entries else None,
            "news_fraction_by_ticker": fractions,
            "coverage_gate_passed": len(entries) == phase_sizes[name]
            and all(
                v >= protocol["minimum_news_fraction_per_ticker_per_phase"]
                for v in fractions.values()
            ),
        }
    ready = all(p["coverage_gate_passed"] for p in phase_reports.values())
    return {
        "as_of": as_of.isoformat(),
        "eligible_origin_start": protocol["eligible_origin_start"],
        "observation_coverage_ready": ready,
        "training_ready": False,
        "phases": phase_reports,
        "session_monitoring": session_rows,
        "note": "Coverage only. No price outcomes or label maturity verified. Readiness never authorizes training or promotion. Phase dates and coverage require a separate frozen dataset and evaluation runner.",
    }


def progress(assembly: Path, input_root: Path, output_root: Path, protocol_path: Path):
    verify_files(assembly)
    protocol = json.loads(protocol_path.read_text())
    report = json.loads((assembly / "report.json").read_text())
    events = json.loads((assembly / "events.json").read_text())
    receipts = []
    for source in report["input_snapshots"]:
        run = input_root / source["snapshot"]
        if run.resolve().parent != input_root.resolve():
            raise ValueError("Invalid snapshot path")
        if (
            hashlib.sha256((run / "manifest.json").read_bytes()).hexdigest()
            != source["manifest_sha256"]
        ):
            raise ValueError("Input snapshot manifest changed")
        _, reason = news_assemble.verified_run(run)
        if reason:
            raise ValueError("Assembly references incomplete input")
        raw_report = json.loads((run / "report.json").read_text())
        receipts.extend(
            {"ticker": r["ticker"], "observed_at": r["observed_at"]}
            for r in raw_report["requests"]
        )
    now = datetime.now(UTC)
    result = assess(events, receipts, protocol, now.isoformat())
    result["complete_collections"] = len(report["input_snapshots"])
    result["successful_ticker_requests"] = len(receipts)
    result["unique_article_versions"] = report["unique_versions"]
    result["assembly"] = str(assembly)
    result["assembly_manifest_sha256"] = hashlib.sha256(
        (assembly / "manifest.json").read_bytes()
    ).hexdigest()
    run = output_root / (now.strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    (run / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (run / "protocol.json").write_bytes(protocol_path.read_bytes())
    (run / "news_progress.py").write_bytes(Path(__file__).read_bytes())
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "files": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(run.iterdir())
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Make five provider requests before assembly; requires local API key",
    )
    parser.add_argument(
        "--input-root", type=Path, default=Path("data/news-observations")
    )
    parser.add_argument(
        "--assembly-root", type=Path, default=Path("data/news-assembled")
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/news-progress"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    failed = False
    if args.collect:
        print("Collecting five companies, 60 seconds between requests...", flush=True)
        try:
            collected = news_collect.collect(
                args.input_root, list(news_collect.SYMBOLS), 24, 60
            )
        except NewsProviderError as error:
            parser.exit(1, safe_provider_reason(error) + "\n")
        failed = not json.loads((collected / "report.json").read_text())[
            "collection_complete"
        ]
        print("Collection saved: " + str(collected), flush=True)
    if not args.input_root.is_dir():
        parser.error(
            "No observations directory; run with --collect in your configured terminal."
        )
    assembled = news_assemble.build(args.input_root, args.assembly_root)
    result = progress(assembled, args.input_root, args.output_root, args.protocol)
    print("Assembly: " + str(assembled))
    print("Coverage report: " + str(result))
    print("No training started; see protocol and coverage gates in report.json.")
    if failed:
        parser.exit(
            1, "Partial collection excluded from assembly. No automatic retries.\n"
        )


if __name__ == "__main__":
    main()

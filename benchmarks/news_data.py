"""Snapshot and audit news coverage without backdating observed availability."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import exchange_calendars as xcals
import pandas as pd

from backend.news import NewsArticle, NewsProviderError, fetch_news, utc_timestamp
from backend.news_features import build_news_features

SYMBOLS = ("AAPL", "MSFT", "NVDA", "AMZN", "TSLA")
HOLDOUT_START = pd.Timestamp("2026-09-28", tz="UTC")


def safe_provider_reason(error: NewsProviderError) -> str:
    allowed = {
        "News provider returned an error or rate-limit response.",
        "News provider request failed.",
        "News response exceeds the 10 MB limit.",
        "News response reached 1000 articles; request a narrower time range.",
        "Set ALPHAVANTAGE_API_KEY to refresh news.",
    }
    return str(error) if str(error) in allowed else "News provider request failed."


def read_archive(path: Path) -> list[NewsArticle]:
    """Read a consistent SQLite snapshot without creating or modifying the archive."""
    if not path.exists():
        return []
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        rows = connection.execute("SELECT payload FROM news ORDER BY id").fetchall()
    finally:
        connection.close()
    return [NewsArticle(**json.loads(row[0])) for row in rows]


def audit(articles: list[NewsArticle], start: str, end: str) -> dict:
    first, last = utc_timestamp(start), utc_timestamp(end)
    if first.normalize() != first or last.normalize() != last:
        raise ValueError("Coverage boundaries must be UTC midnight.")
    if first > last or last >= HOLDOUT_START:
        raise ValueError(
            "Use an ordered development range before the reserved holdout."
        )
    calendar = xcals.get_calendar("XNYS", start=first.date(), end=last.date())
    dates = calendar.sessions_in_range(first.tz_localize(None), last.tz_localize(None))
    if len(dates) == 0:
        raise ValueError("Coverage range contains no trading sessions.")
    frame = pd.DataFrame({"date": dates})
    end_exclusive = pd.Timestamp(last.to_pydatetime() + timedelta(days=1))
    unique = {}
    for article in sorted(articles, key=lambda a: utc_timestamp(a.available_at)):
        unique.setdefault(article.identity, article)
    rows = []
    for symbol in SYMBOLS:
        selected = [a for a in unique.values() if a.ticker == symbol]
        features = build_news_features(frame, selected, symbol)
        published = [
            a
            for a in selected
            if first <= utc_timestamp(a.published_at) < end_exclusive
        ]
        covered = int((features.news_count > 0).sum())
        rows.append(
            {
                "ticker": symbol,
                "unique_articles": len(selected),
                "published_in_development": len(published),
                "published_in_development_observed_after_window": sum(
                    utc_timestamp(a.available_at) >= end_exclusive for a in published
                ),
                "first_available_at": min(
                    (a.available_at for a in selected), default=None
                ),
                "last_available_at": max(
                    (a.available_at for a in selected), default=None
                ),
                "news_covered_sessions": covered,
                "total_sessions": len(dates),
                "coverage_fraction": covered / len(dates),
            }
        )
    return {
        "start": first.isoformat(),
        "end": last.isoformat(),
        "input_articles": len(articles),
        "unique_articles": len(unique),
        "duplicate_urls_removed": len(articles) - len(unique),
        "coverage_rule": "Existing three-calendar-day feature window, relevance >=0.2, first-observed availability and headline deduplication",
        "timestamp_evidence": "Caller-supplied archive timestamps; audit does not independently verify provenance",
        "training_readiness": "not_established",
        "tickers": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/news.sqlite3"))
    parser.add_argument("--output-root", type=Path, default=Path("data/news-datasets"))
    parser.add_argument("--start", default="2021-04-12T00:00:00Z")
    parser.add_argument("--end", default="2026-09-11T00:00:00Z")
    parser.add_argument("--tickers", nargs="+", choices=SYMBOLS, default=list(SYMBOLS))
    parser.add_argument(
        "--collect-from",
        help="Optional UTC historical request start; makes five provider calls",
    )
    parser.add_argument(
        "--collect-to",
        help="UTC historical request end strictly before holdout start",
    )
    args = parser.parse_args()
    if bool(args.collect_from) != bool(args.collect_to):
        parser.error("Supply both collection boundaries.")
    # Validate the audit range before any requests or filesystem writes.
    audit([], args.start, args.end)
    if args.collect_from:
        first, last = utc_timestamp(args.collect_from), utc_timestamp(args.collect_to)
        if first >= last or last >= HOLDOUT_START:
            parser.error("Collection must end strictly before the reserved holdout.")
        if not os.getenv("ALPHAVANTAGE_API_KEY", "").strip():
            parser.error(
                "Set ALPHAVANTAGE_API_KEY locally; never put the key in CLI arguments."
            )
    run = args.output_root / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8]
    )
    run.mkdir(parents=True, exist_ok=False)
    articles = read_archive(args.db)
    requests = []
    if args.collect_from:
        for symbol in dict.fromkeys(args.tickers):
            try:
                fetched = fetch_news(
                    symbol, time_from=args.collect_from, time_to=args.collect_to
                )
            except NewsProviderError as error:
                # Do not print chained provider errors: request URLs can contain keys.
                requests.append(
                    {
                        "ticker": symbol,
                        "status": "failed",
                        "reason": safe_provider_reason(error),
                        "action": "Check credentials, quota and window size; collection stopped",
                    }
                )
                break
            articles.extend(fetched)
            requests.append(
                {
                    "ticker": symbol,
                    "status": "received",
                    "normalized_articles": len(fetched),
                }
            )
    report = audit(articles, args.start, args.end)
    report["requests"] = requests
    report["collection_complete"] = (
        len(requests) == len(set(args.tickers))
        and all(r["status"] == "received" for r in requests)
        if args.collect_from
        else None
    )
    report["historical_completeness_verified"] = False
    report["requested_tickers"] = list(dict.fromkeys(args.tickers))
    report["provider_configured"] = bool(os.getenv("ALPHAVANTAGE_API_KEY", "").strip())
    (run / "articles.json").write_text(
        json.dumps([asdict(a) for a in articles], indent=2), encoding="utf-8"
    )
    (run / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for source in (
        Path(__file__),
        Path("backend/news.py"),
        Path("backend/news_features.py"),
    ):
        (run / source.name).write_bytes(source.read_bytes())
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "collection_from": args.collect_from,
        "collection_to": args.collect_to,
        "content_scope": "Normalized provider headlines, summaries and scores; no full article scraping or raw HTTP archive",
        "files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(run.iterdir())
        },
    }
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(run)
    if report["collection_complete"] is False:
        print("Collection incomplete: " + requests[-1]["reason"])
        raise SystemExit(1)


if __name__ == "__main__":
    main()

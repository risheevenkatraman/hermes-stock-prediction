"""Collect versioned prospective news snapshots without reading price outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from backend.news import (
    NewsArticle,
    NewsProviderError,
    fetch_news_payload,
    normalize_alpha_vantage,
    utc_timestamp,
)
from benchmarks.news_data import SYMBOLS, safe_provider_reason

ALIASES = {
    "AAPL": r"\b(apple|aapl|iphone|ipad|macbook)\b",
    "MSFT": r"\b(microsoft|msft|azure|xbox)\b",
    "NVDA": r"\b(nvidia|nvda|geforce)\b",
    "AMZN": r"\b(amazon|amzn|aws)\b",
    "TSLA": r"\b(tesla|tsla)\b",
}


def quality_flags(article: NewsArticle, start: str, end: str) -> list[str]:
    published = utc_timestamp(article.published_at)
    flags = []
    if not utc_timestamp(start) <= published < utc_timestamp(end):
        flags.append("outside_requested_window")
    if published.hour == published.minute == published.second == 0:
        flags.append("midnight_timestamp_precision_unknown")
    if article.relevance < 0.5:
        flags.append("low_company_relevance")
    text = article.title + " " + article.summary
    if not re.search(ALIASES[article.ticker], text, flags=re.IGNORECASE):
        flags.append("no_company_alias_in_text")
    if any(
        int(year) > published.year for year in re.findall(r"\b20\d{2}\b", article.title)
    ):
        flags.append("future_year_in_title_review_revision")
    return flags


def prepare(
    payload: dict, ticker: str, observed: str, start: str, end: str
) -> tuple[list, dict]:
    feed = payload.get("feed") if isinstance(payload, dict) else None
    if not isinstance(feed, list):
        raise NewsProviderError(
            "News provider returned an error or rate-limit response."
        )
    records = []
    rejected = 0
    seen = set()
    for index, row in enumerate(feed):
        normalized = normalize_alpha_vantage({"feed": [row]}, ticker, observed)
        if not normalized:
            rejected += 1
        for article in normalized:
            flags = quality_flags(article, start, end)
            if article.identity in seen:
                flags.append("duplicate_url_in_response")
            seen.add(article.identity)
            content = asdict(article)
            # Receipt time changes each poll; hash the actual normalized content.
            version = {k: v for k, v in content.items() if k != "available_at"}
            records.append(
                {
                    "article": content,
                    "identity": article.identity,
                    "version_sha256": hashlib.sha256(
                        json.dumps(version, sort_keys=True).encode()
                    ).hexdigest(),
                    "raw_feed_index": index,
                    "quality_flags": flags,
                    "disposition": "review" if flags else "candidate",
                }
            )
    return records, {
        "raw_feed_count": len(feed),
        "normalized_records": len(records),
        "rows_not_normalized": rejected,
        "response_at_cap": len(feed) >= 1000,
        "candidate_records": sum(r["disposition"] == "candidate" for r in records),
        "review_records": sum(r["disposition"] == "review" for r in records),
    }


def collect(
    output: Path, tickers: list[str], lookback_hours: int, interval: float
) -> Path:
    # This is a new observation stream, never a backdated historical import.
    end = datetime.now(UTC).replace(second=0, microsecond=0)
    start = end - timedelta(hours=lookback_hours)
    key = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()
    if not key:
        raise NewsProviderError("Set ALPHAVANTAGE_API_KEY to refresh news.")
    run = output / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8]
    )
    run.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        # Provider errors may echo credentials. Archive parsed JSON with key redaction.
        encoded = json.dumps(value, indent=2).replace(key, "[REDACTED]")
        (run / name).write_text(encoded, encoding="utf-8")

    requests = []
    for index, ticker in enumerate(tickers):
        if index:
            time.sleep(interval)
        try:
            payload, observed = fetch_news_payload(
                ticker, time_from=start.isoformat(), time_to=end.isoformat()
            )
            save(
                ticker + "-response.json", {"observed_at": observed, "payload": payload}
            )
            records, stats = prepare(
                payload, ticker, observed, start.isoformat(), end.isoformat()
            )
            save(ticker + "-records.json", records)
            requests.append(
                {
                    "ticker": ticker,
                    "observed_at": observed,
                    **stats,
                    "status": "incomplete_at_cap"
                    if stats["response_at_cap"]
                    else "received",
                }
            )
            if stats["response_at_cap"]:
                break
        except NewsProviderError as error:
            requests.append(
                {
                    "ticker": ticker,
                    "status": "failed",
                    "reason": safe_provider_reason(error),
                }
            )
            break
    complete = len(requests) == len(tickers) and all(
        r["status"] == "received" for r in requests
    )
    save(
        "report.json",
        {
            "mode": "prospective_observation_only",
            "requested_tickers": tickers,
            "publication_window_start": start.isoformat(),
            "publication_window_end_exclusive": end.isoformat(),
            "collection_complete": complete,
            "historical_completeness_verified": False,
            "training_ready": False,
            "requests": requests,
            "policy": "Conservative review flags are heuristics, not verified errors or event labels. No price outcomes read.",
        },
    )
    for source in (
        Path(__file__),
        Path("backend/news.py"),
        Path("benchmarks/news_data.py"),
    ):
        (run / source.name).write_bytes(source.read_bytes())
    save(
        "manifest.json",
        {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "provider": "alpha_vantage",
            "function": "NEWS_SENTIMENT",
            "sort": "LATEST",
            "limit": 1000,
            "response_scope": "Parsed JSON with credential redaction; no HTTP headers or full article scraping",
            "files": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(run.iterdir())
            },
        },
    )
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", choices=SYMBOLS, default=list(SYMBOLS))
    parser.add_argument("--lookback-hours", type=int, choices=range(1, 73), default=24)
    parser.add_argument("--interval-seconds", type=float, default=60)
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/news-observations")
    )
    args = parser.parse_args()
    if not 60 <= args.interval_seconds <= 3600:
        parser.error("Use a finite request interval from 60 to 3600 seconds.")
    try:
        run = collect(
            args.output_root,
            list(dict.fromkeys(args.tickers)),
            args.lookback_hours,
            args.interval_seconds,
        )
    except NewsProviderError as error:
        parser.exit(1, safe_provider_reason(error) + "\n")
    print(run)
    if not json.loads((run / "report.json").read_text())["collection_complete"]:
        parser.exit(
            1, "Collection incomplete; see the saved report. No automatic retry.\n"
        )


if __name__ == "__main__":
    main()

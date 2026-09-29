"""Timestamped, ticker-specific news ingestion and an immutable SQLite archive."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

import pandas as pd


class NewsProviderError(RuntimeError):
    pass


def utc_timestamp(value: object) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if pd.isna(parsed) or parsed.tzinfo is None:
        raise ValueError("News timestamps must include an explicit timezone.")
    return parsed.tz_convert("UTC")


def ticker_symbol(value: str) -> str:
    symbol = value.strip().upper()
    if not re.fullmatch(r"[A-Z0-9.-]{1,10}", symbol):
        raise ValueError("Invalid ticker symbol.")
    return symbol


@dataclass(frozen=True)
class NewsArticle:
    ticker: str
    title: str
    summary: str
    url: str
    source: str
    published_at: str
    available_at: str
    sentiment: float
    relevance: float
    sentiment_model: str = "alpha_vantage"

    def __post_init__(self):
        object.__setattr__(self, "ticker", ticker_symbol(self.ticker))
        if (
            not self.title.strip()
            or len(self.title) > 2000
            or len(self.summary) > 20000
        ):
            raise ValueError(
                "News needs a title <= 2000 characters and summary <= 20000."
            )
        published = utc_timestamp(self.published_at)
        available = utc_timestamp(self.available_at)
        if available < published:
            raise ValueError("News cannot be available before publication.")
        if not math.isfinite(self.sentiment) or not -1 <= self.sentiment <= 1:
            raise ValueError("Sentiment must be finite and between -1 and 1.")
        if not math.isfinite(self.relevance) or not 0 <= self.relevance <= 1:
            raise ValueError("Relevance must be finite and between 0 and 1.")
        parts = urlsplit(self.url)
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            raise ValueError("Article URL must use HTTP or HTTPS.")
        object.__setattr__(self, "published_at", published.isoformat())
        object.__setattr__(self, "available_at", available.isoformat())

    @property
    def identity(self) -> str:
        # Strip tracking parameters, preserving content IDs in query strings.
        parts = urlsplit(self.url)
        canonical = urlunsplit(
            (
                parts.scheme.lower(),
                parts.netloc.lower(),
                parts.path,
                urlencode(
                    sorted(
                        (key, value)
                        for key, value in parse_qsl(parts.query)
                        if not key.lower().startswith("utm_")
                        and key.lower() not in {"fbclid", "gclid"}
                    )
                ),
                "",
            )
        )
        return hashlib.sha256(f"{self.ticker}|{canonical}".encode()).hexdigest()


def normalize_alpha_vantage(
    payload: dict, ticker: str, observed_at: str
) -> list[NewsArticle]:
    """Use company-specific tone/relevance, never an article's overall sentiment."""
    symbol = ticker_symbol(ticker)
    observed = utc_timestamp(observed_at)
    if not isinstance(payload, dict) or not isinstance(payload.get("feed"), list):
        raise NewsProviderError(
            "News provider returned an error or rate-limit response."
        )
    articles = []
    for row in payload["feed"]:
        if not isinstance(row, dict):
            continue
        sentiment_items = row.get("ticker_sentiment", [])
        if not isinstance(sentiment_items, list):
            continue
        for item in sentiment_items:
            if not isinstance(item, dict) or item.get("ticker") != symbol:
                continue
            try:
                published = pd.to_datetime(
                    row["time_published"], format="%Y%m%dT%H%M%S", utc=True
                )
                if published > observed:
                    continue
                article = NewsArticle(
                    ticker=symbol,
                    title=row["title"],
                    summary=row.get("summary", ""),
                    url=row["url"],
                    source=row.get("source", "unknown"),
                    published_at=published.isoformat(),
                    available_at=observed.isoformat(),
                    sentiment=float(item["ticker_sentiment_score"]),
                    relevance=float(item["relevance_score"]),
                )
            except (KeyError, TypeError, ValueError, AttributeError):
                continue
            if article.relevance >= 0.2:
                articles.append(article)
    return articles


def fetch_news_payload(
    ticker: str, *, time_from: str | None = None, time_to: str | None = None
) -> tuple[dict, str]:
    """Return parsed provider response and receipt time; never expose request URLs."""
    api_key = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()
    if not api_key:
        raise NewsProviderError("Set ALPHAVANTAGE_API_KEY to refresh news.")
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker_symbol(ticker),
        "sort": "LATEST",
        "limit": 1000,
        "apikey": api_key,
    }
    for key, value in (("time_from", time_from), ("time_to", time_to)):
        if value:
            params[key] = utc_timestamp(value).strftime("%Y%m%dT%H%M")
    if time_from and time_to and utc_timestamp(time_from) >= utc_timestamp(time_to):
        raise ValueError("time_from must be earlier than time_to.")
    request = Request(
        "https://www.alphavantage.co/query?" + urlencode(params),
        headers={"User-Agent": "Hermes/0.3", "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            raw = response.read(10_000_001)
        if len(raw) > 10_000_000:
            raise NewsProviderError("News response exceeds the 10 MB limit.")
        payload = json.loads(raw)
    except (URLError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError) as error:
        # Provider URLs contain credentials: never return exception text to clients.
        raise NewsProviderError("News provider request failed.") from error
    return payload, datetime.now(UTC).isoformat()


def fetch_news(
    ticker: str, *, time_from: str | None = None, time_to: str | None = None
) -> list[NewsArticle]:
    payload, observed_at = fetch_news_payload(
        ticker, time_from=time_from, time_to=time_to
    )
    if (
        isinstance(payload, dict)
        and isinstance(payload.get("feed"), list)
        and len(payload["feed"]) >= 1000
    ):
        raise NewsProviderError(
            "News response reached 1000 articles; request a narrower time range."
        )
    return normalize_alpha_vantage(payload, ticker, observed_at)


class NewsStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.getenv("NEWS_DB_PATH", "data/news.sqlite3"))

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=20)
        connection.execute(
            "CREATE TABLE IF NOT EXISTS news (id TEXT PRIMARY KEY, ticker TEXT NOT NULL, available_at TEXT NOT NULL, payload TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS news_ticker_time ON news(ticker, available_at)"
        )
        return connection

    def add(self, articles: list[NewsArticle]) -> int:
        with closing(self._connect()) as connection, connection:
            before = connection.total_changes
            connection.executemany(
                "INSERT OR IGNORE INTO news VALUES (?, ?, ?, ?)",
                [
                    (a.identity, a.ticker, a.available_at, json.dumps(asdict(a)))
                    for a in articles
                ],
            )
            inserted = connection.total_changes - before
        return inserted

    def articles(self, ticker: str) -> list[NewsArticle]:
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                "SELECT payload FROM news WHERE ticker=? ORDER BY available_at, id",
                (ticker_symbol(ticker),),
            ).fetchall()
        return [NewsArticle(**json.loads(row[0])) for row in rows]

    def status(self, ticker: str) -> dict:
        with closing(self._connect()) as connection, connection:
            count, first, last = connection.execute(
                "SELECT COUNT(*), MIN(available_at), MAX(available_at) FROM news WHERE ticker=?",
                (ticker_symbol(ticker),),
            ).fetchone()
        return {
            "ticker": ticker_symbol(ticker),
            "articles": count,
            "provider_configured": bool(os.getenv("ALPHAVANTAGE_API_KEY", "").strip()),
            "first_available_at": first,
            "last_available_at": last,
        }

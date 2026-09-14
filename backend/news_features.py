"""Point-in-time daily news features aligned to actual US exchange closes."""

from __future__ import annotations

from datetime import timedelta

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from .news import NewsArticle, utc_timestamp

NEWS_COLUMNS = [
    "news_count",
    "news_sentiment",
    "news_positive",
    "news_negative",
    "news_disagreement",
]


def session_closes(frame: pd.DataFrame) -> pd.DatetimeIndex:
    if "date" not in frame or frame["date"].isna().any():
        raise ValueError("Dated prices are required for news alignment.")
    dates = (
        pd.DatetimeIndex(pd.to_datetime(frame["date"], utc=True))
        .tz_localize(None)
        .normalize()
    )
    calendar = xcals.get_calendar(
        "XNYS",
        start=dates.min() - timedelta(days=10),
        end=dates.max() + timedelta(days=10),
    )
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Price session dates must be unique and chronological.")
    closes = calendar.schedule["close"].reindex(dates)
    if closes.isna().any():
        raise ValueError("News direction analysis requires US exchange session dates.")
    return pd.DatetimeIndex(closes)


def build_news_features(
    frame: pd.DataFrame, articles: list[NewsArticle], ticker: str
) -> pd.DataFrame:
    cutoffs = session_closes(frame)
    records = []
    # Syndicated copies with the same headline must not multiply the signal.
    seen = set()
    for article in sorted(articles, key=lambda a: a.available_at):
        key = " ".join(article.title.lower().split())
        if article.ticker != ticker.upper() or article.relevance < 0.2 or key in seen:
            continue
        seen.add(key)
        records.append(
            (
                article,
                utc_timestamp(article.published_at),
                utc_timestamp(article.available_at),
            )
        )
    rows = []
    active = []
    position = 0
    for cutoff in cutoffs:
        while position < len(records) and records[position][2] <= cutoff:
            article, published, _ = records[position]
            active.append((article, published))
            position += 1
        oldest = cutoff - timedelta(days=3)
        active = [
            (article, published) for article, published in active if published > oldest
        ]
        recent = active
        if not recent:
            rows.append([0.0] * len(NEWS_COLUMNS) + ["no_news"])
            continue
        weights = np.array(
            [
                a.relevance * np.exp(-(cutoff - published).total_seconds() / 86400)
                for a, published in recent
            ]
        )
        scores = np.array([a.sentiment for a, _ in recent])
        sentiment = float(np.average(scores, weights=weights))
        rows.append(
            [
                float(len(recent)),
                sentiment,
                float(np.average(scores > 0.15, weights=weights)),
                float(np.average(scores < -0.15, weights=weights)),
                float(np.sqrt(np.average((scores - sentiment) ** 2, weights=weights))),
                " ".join(a.title + " " + a.summary for a, _ in recent)[:50000],
            ]
        )
    return pd.DataFrame(
        rows, index=frame.index, columns=NEWS_COLUMNS + ["article_text"]
    )

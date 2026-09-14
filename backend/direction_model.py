"""Experimental classification head: price direction plus learned news reactions."""

from __future__ import annotations

from datetime import timedelta

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .model import FEATURE_COLUMNS, _build_feature_frame, _validate_prices
from .news import NewsArticle
from .news_features import NEWS_COLUMNS, build_news_features, session_closes


MIN_DIRECTION_ROWS = 120
MIN_NEWS_DAYS = 30


def direction_dataset(prices: pd.DataFrame, articles: list[NewsArticle], ticker: str):
    frame = _validate_prices(prices)
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
    if not dates.equals(calendar.sessions_in_range(dates.min(), dates.max())):
        raise ValueError("Direction targets require consecutive US trading sessions.")
    news = build_news_features(frame, articles, ticker)
    features = _build_feature_frame(frame).join(news)
    target = frame["close"].shift(-1) / frame["close"] - 1
    valid = features[FEATURE_COLUMNS].notna().all(axis=1) & target.notna()
    return frame, features, target, valid


def _predict_probability(
    train: pd.DataFrame,
    labels: pd.Series,
    inference: pd.DataFrame,
    *,
    with_news: bool = False,
) -> np.ndarray:
    if labels.nunique() == 1:
        return np.full(len(inference), float(labels.iloc[0]))
    columns = FEATURE_COLUMNS + (NEWS_COLUMNS if with_news else [])
    transformations = [("numeric", StandardScaler(), columns)]
    if with_news:
        # Vocabulary and IDF are fit only on this historical training window.
        transformations.append(
            (
                "text",
                TfidfVectorizer(
                    max_features=500, ngram_range=(1, 2), min_df=1, sublinear_tf=True
                ),
                "article_text",
            )
        )
    model = Pipeline(
        [
            ("features", ColumnTransformer(transformations)),
            ("classifier", LogisticRegression(C=0.1, max_iter=500, random_state=42)),
        ]
    )
    model.fit(train, labels)
    return model.predict_proba(inference)[:, 1]


def _select_policy(
    labels: np.ndarray, price: np.ndarray, news: np.ndarray, news_allowed: bool
) -> tuple[float, float]:
    # Explicit always-up candidate; ties preserve this simpler baseline.
    best_accuracy = float(np.mean(labels == 1))
    best = (0.0, 0.0)
    for weight in ((0.0, 0.25, 0.5) if news_allowed else (0.0,)):
        probability = (1 - weight) * price + weight * news
        for threshold in (0.45, 0.5, 0.55):
            accuracy = float(np.mean((probability >= threshold) == labels))
            if accuracy > best_accuracy:
                best_accuracy, best = accuracy, (weight, threshold)
    return best


def _scores(labels: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    predicted = probability >= threshold
    return {
        "observations": len(labels),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "brier_score": float(brier_score_loss(labels, probability)),
        "always_up_accuracy": float(np.mean(labels == 1)),
        "predicted_down_fraction": float(np.mean(~predicted)),
    }


def predict_direction(
    prices: pd.DataFrame, articles: list[NewsArticle], ticker: str
) -> dict:
    frame, features, target, valid = direction_dataset(prices, articles, ticker)
    train = features.loc[valid]
    labels = (target.loc[valid] > 0).astype(int)
    if len(train) < MIN_DIRECTION_ROWS:
        raise ValueError(
            f"Direction analysis requires at least {MIN_DIRECTION_ROWS} usable rows."
        )
    if features[FEATURE_COLUMNS].iloc[-1].isna().any():
        raise ValueError("Latest price features are incomplete.")
    fit_end, selection_end = int(len(train) * 0.6), int(len(train) * 0.8)
    fit = train.iloc[:fit_end]
    heldout = train.iloc[fit_end:]
    fit_labels = labels.iloc[:fit_end]
    news_days = int((fit["news_count"] > 0).sum())
    news_allowed = news_days >= MIN_NEWS_DAYS
    price_probability = _predict_probability(fit, fit_labels, heldout)
    news_probability = (
        _predict_probability(fit, fit_labels, heldout, with_news=True)
        if news_allowed
        else price_probability
    )
    news_probability = np.where(
        heldout["news_count"].to_numpy() > 0, news_probability, price_probability
    )
    selection_rows = selection_end - fit_end
    weight, threshold = _select_policy(
        labels.iloc[fit_end:selection_end].to_numpy(),
        price_probability[:selection_rows],
        news_probability[:selection_rows],
        news_allowed,
    )
    evaluation_probability = (1 - weight) * price_probability[
        selection_rows:
    ] + weight * news_probability[selection_rows:]
    evaluation_labels = labels.iloc[selection_end:].to_numpy()
    metrics = _scores(evaluation_labels, evaluation_probability, threshold)
    metrics["excess_accuracy"] = metrics["accuracy"] - metrics["always_up_accuracy"]
    metrics["price_only_accuracy_at_0_5"] = float(
        accuracy_score(evaluation_labels, price_probability[selection_rows:] >= 0.5)
    )
    metrics["training_prior_brier"] = float(
        brier_score_loss(
            evaluation_labels, np.full(len(evaluation_labels), float(fit_labels.mean()))
        )
    )
    latest = features.iloc[[-1]]
    price_latest = float(_predict_probability(train, labels, latest)[0])
    news_latest = (
        float(_predict_probability(train, labels, latest, with_news=True)[0])
        if news_allowed
        else None
    )
    selected_weight = weight
    if float(latest.iloc[0]["news_count"]) == 0:
        weight = 0.0
    probability = (1 - weight) * price_latest + weight * (
        news_latest if news_latest is not None else price_latest
    )
    cutoff = session_closes(frame)[-1]
    snapshot = {name: float(latest.iloc[0][name]) for name in NEWS_COLUMNS}
    return {
        "ticker": ticker.upper(),
        "status": "experimental",
        "horizon": "next trading-day close versus latest close",
        "as_of": cutoff.isoformat(),
        "direction": "up" if probability >= threshold else "down_or_flat",
        "up_probability": probability,
        "decision_threshold": threshold,
        "policy": "always_up" if threshold == 0 else "learned_direction",
        "news_weight": weight,
        "selected_news_weight": selected_weight,
        "price_up_probability": price_latest,
        "news_up_probability": news_latest,
        "news_status": (
            "trained" if news_allowed else "insufficient_point_in_time_history"
        ),
        "news_training_days": news_days,
        "minimum_news_training_days": MIN_NEWS_DAYS,
        "news_features": snapshot,
        "news_tone": (
            "positive"
            if snapshot["news_sentiment"] > 0.15
            else "negative" if snapshot["news_sentiment"] < -0.15 else "neutral"
        ),
        "evaluation": metrics,
        "evaluation_start": str(frame.loc[train.index[selection_end], "date"]),
        "evaluation_end": str(frame.loc[train.index[-1], "date"]),
        "training_rows": len(train),
        "articles_after_cutoff": sum(
            pd.Timestamp(a.available_at) > cutoff for a in articles
        ),
    }

"""Fixed research heads; training-only transforms and no automatic promotion."""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .compact_model import compact_groups

NEWS_NUMERIC = [
    "news_count",
    "news_sentiment",
    "news_positive",
    "news_negative",
    "news_disagreement",
]
LONG_FEATURES = [
    "stock_return_21",
    "stock_return_63",
    "stock_return_126",
    "stock_return_252",
    "stock_volatility_21",
    "market_return_63",
    "market_return_126",
    "market_return_252",
    "relative_sector_return_63",
    "relative_sector_return_126",
    "relative_sector_return_252",
]
LONG_HORIZONS = (126, 252)


def columns_for(variant):
    if variant == "long_term":
        return LONG_FEATURES
    if variant not in {"price_sector_only", "news_only", "price_sector_plus_news"}:
        raise ValueError("Unknown research model variant")
    return ([] if variant == "news_only" else compact_groups()["sector"]) + (
        [] if variant == "price_sector_only" else NEWS_NUMERIC
    )


def clean_features(frame, columns, text):
    result = frame.copy()
    if not np.isfinite(result[columns].to_numpy(dtype=float)).all():
        raise ValueError("Research features must be finite and complete")
    if text:
        if result.article_text.isna().any():
            raise ValueError(
                "Missing article text; use empty text for monitored quiet days"
            )
        # An explicit token allows a completely quiet training window without an
        # empty vocabulary; no future text is used to construct the vocabulary.
        result["article_text"] = result.article_text.map(
            lambda x: x if str(x).strip() else "no_news"
        )
    return result


class ResearchHead:
    def __init__(self, variant, classification=True):
        self.variant = variant
        self.classification = classification
        self.columns = columns_for(variant)
        self.with_text = variant in {"news_only", "price_sector_plus_news"}

    def fit(self, features, returns):
        if not features.index.equals(returns.index) or len(features) < 40:
            raise ValueError("At least 40 aligned training rows are required")
        y = returns.to_numpy(dtype=float)
        if not np.isfinite(y).all():
            raise ValueError("Training labels must be finite and matured")
        x = clean_features(features, self.columns, self.with_text)
        transforms = [("numeric", StandardScaler(), self.columns)]
        if self.with_text:
            transforms.append(
                (
                    "text",
                    TfidfVectorizer(max_features=500, ngram_range=(1, 2)),
                    "article_text",
                )
            )
        labels = (y > 0).astype(int) if self.classification else y
        estimator = (
            (
                DummyClassifier(strategy="prior")
                if self.classification and len(np.unique(labels)) == 1
                else LogisticRegression(C=0.01, max_iter=1000, random_state=42)
            )
            if self.classification
            else Ridge(alpha=100.0)
        )
        self.model = Pipeline(
            [
                ("features", ColumnTransformer(transforms, sparse_threshold=0)),
                ("head", estimator),
            ]
        )
        self.model.fit(x, labels)
        self.training_prior = float(np.mean(y > 0))
        self.training_mean = float(y.mean())
        return self

    def predict(self, features):
        x = clean_features(features, self.columns, self.with_text)
        if not self.classification:
            return self.model.predict(x)
        estimator = self.model.named_steps["head"]
        if len(estimator.classes_) == 1:
            return np.full(len(x), float(estimator.classes_[0]))
        return self.model.predict_proba(x)[:, list(estimator.classes_).index(1)]


def long_term_signal(probability, horizon, as_of):
    if (
        horizon not in LONG_HORIZONS
        or not np.isfinite(probability)
        or not 0 <= probability <= 1
    ):
        raise ValueError("Invalid long-term horizon or probability")
    direction = (
        "positive"
        if probability >= 0.6
        else "negative_or_flat"
        if probability <= 0.4
        else "uncertain"
    )
    return {
        "horizon_sessions": horizon,
        "months": 6 if horizon == 126 else 12,
        "as_of": as_of,
        "signal": direction,
        "positive_probability": float(probability),
        "status": "experimental",
        "calibrated": False,
        "recommendation_eligible": False,
        "explanation": "Price and sector research model estimates the chance of a positive adjusted close return over this horizon. Probabilities are uncalibrated, and the uncertain band is a fixed research rule. No validated recommendation or precise future price is implied.",
    }

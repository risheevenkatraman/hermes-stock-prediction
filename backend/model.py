"""Leakage-aware price trend forecasting with scikit-learn."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .deep_model import predict_return

FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "sma_ratio_5",
    "sma_ratio_20",
    "volatility_20d",
    "range_pct",
    "volume_change",
    "rsi_14",
]


class InsufficientHistoryError(ValueError):
    """Raised when there are not enough usable observations to train."""


@dataclass(frozen=True)
class Forecast:
    predicted_return: float
    expected_price: float
    direction: str
    confidence: float
    training_rows: int
    metrics: dict[str, float]
    feature_snapshot: dict[str, float]
    five_day_return: float
    five_day_direction: str
    direction_probability: float
    deep_predicted_return: float
    hybrid_predicted_return: float


def _regression_model() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    max_iter=150,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    random_state=42,
                ),
            ),
        ]
    )


def _validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    required = {"close", "high", "low", "volume"}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(
            f"Missing required price columns: {', '.join(sorted(missing))}"
        )

    frame = prices.copy()
    frame.columns = [str(column).lower() for column in frame.columns]
    if "date" in frame:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.sort_values("date")
    for column in ("close", "high", "low", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["close", "high", "low", "volume"])
    frame = frame[(frame["close"] > 0) & (frame["high"] > 0) & (frame["low"] > 0)]
    if len(frame) < 60:
        raise InsufficientHistoryError("At least 60 valid price rows are required.")
    return frame.reset_index(drop=True)


def _build_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    close = frame["close"]
    returns = close.pct_change()
    features = pd.DataFrame(index=frame.index)
    features["return_1d"] = returns
    features["return_5d"] = close.pct_change(5)
    features["return_20d"] = close.pct_change(20)
    features["sma_ratio_5"] = close / close.rolling(5).mean() - 1
    features["sma_ratio_20"] = close / close.rolling(20).mean() - 1
    features["volatility_20d"] = returns.rolling(20).std()
    features["range_pct"] = (frame["high"] - frame["low"]) / close
    features["volume_change"] = frame["volume"].pct_change().clip(-5, 5)
    delta = close.diff()
    gains = delta.clip(lower=0).rolling(14).mean()
    losses = -delta.clip(upper=0).rolling(14).mean()
    relative_strength = gains / losses.replace(0, np.nan)
    features["rsi_14"] = 100 - (100 / (1 + relative_strength))
    features["rsi_14"] = features["rsi_14"].fillna(50)
    return features.replace([np.inf, -np.inf], np.nan)


def build_features(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build training features using only information before the next close."""
    frame = _validate_prices(prices)
    features = _build_feature_frame(frame)
    close = frame["close"]
    target = close.shift(-1) / close - 1
    usable = features.join(target.rename("target")).dropna()
    return usable[FEATURE_COLUMNS], usable["target"]


def _training_data(
    frame: pd.DataFrame,
    horizon: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create feature rows whose forward return is known."""
    features = _build_feature_frame(frame)
    close = frame["close"]
    target = close.shift(-horizon) / close - 1
    usable = features.join(target.rename("target")).dropna()
    return usable[FEATURE_COLUMNS], usable["target"]


def _confidence(predicted_return: float, validation_error: float) -> float:
    signal_strength = min(abs(predicted_return) / 0.02, 1.0)
    error_penalty = min(validation_error / 0.03, 1.0)
    return round(
        float(np.clip(0.5 + signal_strength * 0.35 - error_penalty * 0.2, 0.5, 0.95)), 3
    )


def forecast(prices: pd.DataFrame) -> Forecast:
    """Train on historical rows and forecast the next trading-day return."""
    frame = _validate_prices(prices)
    all_features = _build_feature_frame(frame)
    features, target = _training_data(frame, horizon=1)
    five_day_features, five_day_target = _training_data(frame, horizon=5)
    if len(features) < 40:
        raise InsufficientHistoryError("At least 40 usable feature rows are required.")
    inference_features = all_features.iloc[[-1]][FEATURE_COLUMNS]
    if inference_features.isna().any().any():
        raise InsufficientHistoryError(
            "The latest price row does not have enough history for features."
        )

    split = max(30, int(len(features) * 0.8))
    if split >= len(features):
        split = len(features) - 1
    model = _regression_model()
    model.fit(features.iloc[:split], target.iloc[:split])
    validation_prediction = model.predict(features.iloc[split:])
    validation_error = float(
        mean_absolute_error(target.iloc[split:], validation_prediction)
    )
    directional_accuracy = float(
        (
            np.sign(target.iloc[split:].to_numpy()) == np.sign(validation_prediction)
        ).mean()
    )
    r2 = (
        float(r2_score(target.iloc[split:], validation_prediction))
        if len(validation_prediction) > 1
        else 0.0
    )

    model.fit(features, target)
    statistical_return = float(model.predict(inference_features)[0])
    deep_result = predict_return(features, target, inference_features)
    deep_weight = 0.3 if deep_result.validation_mae < validation_error else 0.0
    predicted_return = (
        1 - deep_weight
    ) * statistical_return + deep_weight * deep_result.predicted_return
    last_close = float(frame["close"].iloc[-1])
    five_day_model = _regression_model()
    five_day_model.fit(five_day_features, five_day_target)
    five_day_return = float(five_day_model.predict(inference_features)[0])

    classifier = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.05,
        max_leaf_nodes=15,
        random_state=42,
    )
    classifier.fit(features, (target > 0).astype(int))
    five_day_probability = float(classifier.predict_proba(inference_features)[0, 1])
    expected_price = last_close * (1 + predicted_return)
    direction = (
        "up"
        if predicted_return > 0.002
        else "down" if predicted_return < -0.002 else "flat"
    )
    return Forecast(
        predicted_return=round(predicted_return, 6),
        expected_price=round(expected_price, 2),
        direction=direction,
        confidence=_confidence(predicted_return, validation_error),
        training_rows=len(features),
        metrics={
            "mae": round(validation_error, 6),
            "statistical_mae": round(validation_error, 6),
            "deep_mae": round(deep_result.validation_mae, 6),
            "deep_directional_accuracy": round(
                deep_result.validation_directional_accuracy, 4
            ),
            "deep_weight": deep_weight,
            "r2": round(r2, 4),
            "directional_accuracy": round(directional_accuracy, 4),
        },
        feature_snapshot={
            key: round(float(inference_features.iloc[0][key]), 6)
            for key in FEATURE_COLUMNS
        },
        five_day_return=round(five_day_return, 6),
        five_day_direction=(
            "up"
            if five_day_return > 0.005
            else "down" if five_day_return < -0.005 else "flat"
        ),
        direction_probability=round(five_day_probability, 3),
        deep_predicted_return=round(deep_result.predicted_return, 6),
        hybrid_predicted_return=round(predicted_return, 6),
    )

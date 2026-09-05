"""Leakage-aware price trend forecasting with scikit-learn."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


def build_features(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build features using only information available before the next close."""
    frame = _validate_prices(prices)
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
    target = close.shift(-1) / close - 1
    usable = (
        features.join(target.rename("target"))
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    return usable[FEATURE_COLUMNS], usable["target"]


def _confidence(predicted_return: float, validation_error: float) -> float:
    signal_strength = min(abs(predicted_return) / 0.02, 1.0)
    error_penalty = min(validation_error / 0.03, 1.0)
    return round(
        float(np.clip(0.5 + signal_strength * 0.35 - error_penalty * 0.2, 0.5, 0.95)), 3
    )


def forecast(prices: pd.DataFrame) -> Forecast:
    """Train on historical rows and forecast the next trading-day return."""
    features, target = build_features(prices)
    if len(features) < 40:
        raise InsufficientHistoryError("At least 40 usable feature rows are required.")

    split = max(30, int(len(features) * 0.8))
    if split >= len(features):
        split = len(features) - 1
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    max_iter=150, learning_rate=0.05, max_leaf_nodes=15, random_state=42
                ),
            ),
        ]
    )
    model.fit(features.iloc[:split], target.iloc[:split])
    validation_prediction = model.predict(features.iloc[split:])
    validation_error = float(
        mean_absolute_error(target.iloc[split:], validation_prediction)
    )
    r2 = (
        float(r2_score(target.iloc[split:], validation_prediction))
        if len(validation_prediction) > 1
        else 0.0
    )

    model.fit(features, target)
    predicted_return = float(model.predict(features.iloc[[-1]])[0])
    last_close = float(_validate_prices(prices)["close"].iloc[-1])
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
        metrics={"mae": round(validation_error, 6), "r2": round(r2, 4)},
        feature_snapshot={
            key: round(float(features.iloc[-1][key]), 6) for key in FEATURE_COLUMNS
        },
    )

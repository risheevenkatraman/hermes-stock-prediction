"""Walk-forward evaluation and simple benchmark strategies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from .model import _validate_prices, build_features


@dataclass(frozen=True)
class StrategyMetrics:
    name: str
    observations: int
    mae: float
    directional_accuracy: float
    cumulative_return: float


def _model():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

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


def _directional_accuracy(actual: pd.Series, predicted: np.ndarray) -> float:
    return float((np.sign(actual.to_numpy()) == np.sign(predicted)).mean())


def _metrics(name: str, actual: pd.Series, predicted: np.ndarray) -> StrategyMetrics:
    actual_values = actual.to_numpy()
    returns = np.where(predicted > 0, actual_values, 0.0)
    return StrategyMetrics(
        name=name,
        observations=len(actual),
        mae=round(float(mean_absolute_error(actual_values, predicted)), 6),
        directional_accuracy=round(_directional_accuracy(actual, predicted), 4),
        cumulative_return=round(float(np.prod(1 + returns) - 1), 4),
    )


def walk_forward_backtest(
    prices: pd.DataFrame,
    *,
    min_train_rows: int = 120,
    step: int = 1,
) -> dict[str, StrategyMetrics]:
    """Evaluate one-step forecasts without training on future observations."""
    if min_train_rows < 40:
        raise ValueError("min_train_rows must be at least 40.")
    if step < 1:
        raise ValueError("step must be at least 1.")

    frame = _validate_prices(prices)
    features, target = build_features(frame)
    if len(features) <= min_train_rows:
        raise ValueError("More history is required for the requested backtest window.")

    model_predictions: list[float] = []
    actual_values: list[float] = []
    baseline_predictions: list[float] = []
    for index in range(min_train_rows, len(features), step):
        model = _model()
        model.fit(features.iloc[:index], target.iloc[:index])
        model_predictions.append(float(model.predict(features.iloc[[index]])[0]))
        actual_values.append(float(target.iloc[index]))
        baseline_predictions.append(float(features.iloc[index]["return_1d"]))

    actual = pd.Series(actual_values)
    model_metrics = _metrics("Hermes model", actual, np.array(model_predictions))
    previous_day_metrics = _metrics(
        "Previous-day return", actual, np.array(baseline_predictions)
    )
    buy_hold_actual = actual.copy()
    buy_hold = StrategyMetrics(
        name="Buy and hold",
        observations=len(actual),
        mae=round(
            float(mean_absolute_error(buy_hold_actual, np.zeros(len(actual)))), 6
        ),
        directional_accuracy=round(float((buy_hold_actual > 0).mean()), 4),
        cumulative_return=round(float(np.prod(1 + buy_hold_actual) - 1), 4),
    )
    return {
        "model": model_metrics,
        "previous_day": previous_day_metrics,
        "buy_and_hold": buy_hold,
    }


def serialize_metrics(
    metrics: dict[str, StrategyMetrics],
) -> dict[str, dict[str, float | int | str]]:
    return {
        key: {
            "name": value.name,
            "observations": value.observations,
            "mae": value.mae,
            "directional_accuracy": value.directional_accuracy,
            "cumulative_return": value.cumulative_return,
        }
        for key, value in metrics.items()
    }

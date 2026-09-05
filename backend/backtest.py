"""Walk-forward evaluation and simple benchmark strategies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_absolute_error

from .model import _regression_model, _training_data, _validate_prices


@dataclass(frozen=True)
class StrategyMetrics:
    name: str
    observations: int
    mae: float
    directional_accuracy: float
    cumulative_return: float
    max_drawdown: float
    annualized_volatility: float
    brier_score: float | None = None


def _risk_metrics(strategy_returns: np.ndarray) -> tuple[float, float]:
    equity = np.cumprod(1 + strategy_returns)
    drawdown = equity / np.maximum.accumulate(equity) - 1
    volatility = (
        float(np.std(strategy_returns, ddof=1) * np.sqrt(252))
        if len(strategy_returns) > 1
        else 0.0
    )
    return float(drawdown.min()), volatility


def _classifier_model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                HistGradientBoostingClassifier(
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


def _metrics(
    name: str,
    actual: pd.Series,
    predicted: np.ndarray,
    *,
    probabilities: np.ndarray | None = None,
) -> StrategyMetrics:
    actual_values = actual.to_numpy()
    returns = np.where(predicted > 0, actual_values, 0.0)
    max_drawdown, annualized_volatility = _risk_metrics(returns)
    return StrategyMetrics(
        name=name,
        observations=len(actual),
        mae=round(float(mean_absolute_error(actual_values, predicted)), 6),
        directional_accuracy=round(_directional_accuracy(actual, predicted), 4),
        cumulative_return=round(float(np.prod(1 + returns) - 1), 4),
        max_drawdown=round(max_drawdown, 4),
        annualized_volatility=round(annualized_volatility, 4),
        brier_score=(
            round(
                float(
                    brier_score_loss(
                        (actual_values > 0).astype(int),
                        probabilities,
                    )
                ),
                6,
            )
            if probabilities is not None
            else None
        ),
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
    one_day_features, one_day_target = _training_data(frame, horizon=1)
    five_day_features, five_day_target = _training_data(frame, horizon=5)
    if (
        len(one_day_features) <= min_train_rows
        or len(five_day_features) <= min_train_rows
    ):
        raise ValueError("More history is required for the requested backtest window.")

    model_predictions: list[float] = []
    one_day_actual: list[float] = []
    baseline_predictions: list[float] = []
    five_day_predictions: list[float] = []
    five_day_actual: list[float] = []
    classifier_predictions: list[float] = []
    classifier_probabilities: list[float] = []

    for index in range(min_train_rows, len(five_day_features), step):
        one_day_model = _regression_model()
        one_day_model.fit(one_day_features.iloc[:index], one_day_target.iloc[:index])
        model_predictions.append(
            float(one_day_model.predict(one_day_features.iloc[[index]])[0])
        )
        one_day_actual.append(float(one_day_target.iloc[index]))
        baseline_predictions.append(float(one_day_features.iloc[index]["return_1d"]))

        five_day_model = _regression_model()
        five_day_model.fit(five_day_features.iloc[:index], five_day_target.iloc[:index])
        five_day_predictions.append(
            float(five_day_model.predict(five_day_features.iloc[[index]])[0])
        )
        five_day_actual.append(float(five_day_target.iloc[index]))

        classifier = _classifier_model()
        classifier.fit(
            one_day_features.iloc[:index],
            (one_day_target.iloc[:index] > 0).astype(int),
        )
        classifier_predictions.append(
            1.0 if classifier.predict(one_day_features.iloc[[index]])[0] else -1.0
        )
        classifier_probabilities.append(
            float(classifier.predict_proba(one_day_features.iloc[[index]])[0, 1])
        )

    actual = pd.Series(one_day_actual)
    model_metrics = _metrics("Hermes model", actual, np.array(model_predictions))
    previous_day_metrics = _metrics(
        "Previous-day return", actual, np.array(baseline_predictions)
    )
    buy_hold_actual = actual.copy()
    buy_hold_returns = buy_hold_actual.to_numpy()
    max_drawdown, annualized_volatility = _risk_metrics(buy_hold_returns)
    buy_hold = StrategyMetrics(
        name="Buy and hold",
        observations=len(actual),
        mae=round(
            float(mean_absolute_error(buy_hold_actual, np.zeros(len(actual)))), 6
        ),
        directional_accuracy=round(float((buy_hold_actual > 0).mean()), 4),
        cumulative_return=round(float(np.prod(1 + buy_hold_actual) - 1), 4),
        max_drawdown=round(max_drawdown, 4),
        annualized_volatility=round(annualized_volatility, 4),
    )
    five_day_actual_series = pd.Series(five_day_actual)
    five_day_metrics = _metrics(
        "Hermes five-day model",
        five_day_actual_series,
        np.array(five_day_predictions),
    )
    classifier_metrics = _metrics(
        "Hermes direction classifier",
        actual,
        np.array(classifier_predictions),
        probabilities=np.array(classifier_probabilities),
    )
    return {
        "model": model_metrics,
        "previous_day": previous_day_metrics,
        "buy_and_hold": buy_hold,
        "five_day_model": five_day_metrics,
        "direction_classifier": classifier_metrics,
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
            "max_drawdown": value.max_drawdown,
            "annualized_volatility": value.annualized_volatility,
            "brier_score": value.brier_score,
        }
        for key, value in metrics.items()
    }

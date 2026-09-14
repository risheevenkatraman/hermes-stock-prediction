import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.backtest import walk_forward_backtest
from backend.main import app
from backend.model import build_features, forecast


def synthetic_prices(rows: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0.15, 1.0, rows))
    close = np.maximum(close, 10)
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=rows, freq="B"),
            "close": close,
            "high": close + rng.uniform(0.2, 1.2, rows),
            "low": close - rng.uniform(0.2, 1.2, rows),
            "volume": rng.integers(100_000, 500_000, rows),
        }
    )


def test_feature_pipeline_has_expected_shape():
    features, target = build_features(synthetic_prices())
    assert list(features.columns) == [
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
    assert len(features) == len(target)
    assert len(features) > 40


def test_forecast_returns_bounded_prediction_and_validation_metrics():
    result = forecast(synthetic_prices())
    assert result.direction in {"up", "down", "flat"}
    assert 0.5 <= result.confidence <= 0.95
    assert result.expected_price > 0
    assert result.metrics["mae"] >= 0
    assert 0 <= result.metrics["directional_accuracy"] <= 1
    assert result.training_rows > 40
    assert result.five_day_direction in {"up", "down", "flat"}
    assert -1 < result.five_day_return < 1
    assert 0 <= result.direction_probability <= 1
    assert -1 < result.deep_predicted_return < 1
    assert result.hybrid_predicted_return == result.predicted_return
    assert result.metrics["deep_mae"] >= 0
    assert 0 <= result.metrics["deep_directional_accuracy"] <= 1


def test_forecast_uses_latest_price_row_for_inference_features():
    prices = synthetic_prices(100)
    result = forecast(prices)
    expected_return = prices["close"].iloc[-1] / prices["close"].iloc[-2] - 1

    assert result.feature_snapshot["return_1d"] == round(float(expected_return), 6)


def test_live_prediction_endpoint_uses_ingested_history(monkeypatch):
    from backend import main

    monkeypatch.setattr(main, "fetch_daily_prices", lambda _: synthetic_prices())
    response = TestClient(app).get("/predict/live/MSFT")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ticker"] == "MSFT"
    assert payload["market_data"]["latest_price"] > 0
    assert payload["market_data"]["as_of"] == "2025-05-20"
    assert "deep_learning" in payload
    assert "hybrid_predicted_return" in payload


def test_live_prices_endpoint_returns_selected_chart_window(monkeypatch):
    from backend import main

    def fake_fetch(*args, **kwargs):
        del args, kwargs
        return synthetic_prices(100)

    monkeypatch.setattr(
        main,
        "fetch_daily_prices",
        fake_fetch,
    )
    response = TestClient(app).get("/prices/live/MSFT?range=1M")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ticker"] == "MSFT"
    assert payload["range"] == "1M"
    assert len(payload["prices"]) == 22
    assert payload["prices"][0]["close"] > 0


def test_walk_forward_backtest_returns_model_and_baselines():
    results = walk_forward_backtest(synthetic_prices(220), min_train_rows=80)

    assert set(results) == {
        "model",
        "deep_model",
        "hybrid_model",
        "previous_day",
        "buy_and_hold",
        "five_day_model",
        "direction_classifier",
    }
    assert all(result.observations > 0 for result in results.values())
    assert all(0 <= result.directional_accuracy <= 1 for result in results.values())
    assert all(result.cumulative_return > -1 for result in results.values())
    assert all(result.annualized_volatility >= 0 for result in results.values())
    assert results["direction_classifier"].brier_score is not None
    assert 0 <= results["direction_classifier"].brier_score <= 1
    assert all(result.max_drawdown <= 0 for result in results.values())
    assert results["hybrid_model"].mae >= 0


def test_multi_backtest_endpoint_aggregates_tickers(monkeypatch):
    from backend import main

    def fake_fetch(*args, **kwargs):
        del args, kwargs
        return synthetic_prices(220)

    monkeypatch.setattr(
        main,
        "fetch_daily_prices",
        fake_fetch,
    )
    response = TestClient(app).get("/backtest/live?tickers=SPY,QQQ&period=1y")

    assert response.status_code == 200
    payload = response.json()
    assert payload["period"] == "1y"
    assert set(payload["results"]) == {"SPY", "QQQ"}
    assert payload["summary"]["model"]["tickers_evaluated"] == 2
    assert payload["summary"]["model"]["average_annualized_volatility"] >= 0


@pytest.mark.parametrize("trend, expected", [(1, 100), (-1, 0), (0, 50)])
def test_rsi_handles_one_sided_and_flat_prices(trend, expected):
    prices = synthetic_prices()
    prices["close"] = 200 + np.arange(len(prices)) * trend
    prices["high"] = prices["close"] + 1
    prices["low"] = prices["close"] - 1
    prices["volume"] = 0
    features, _ = build_features(prices)
    assert (features["rsi_14"] == expected).all()
    assert (features["volume_change"] == 0).all()


def test_price_validation_normalizes_columns_and_sorts_without_mutation():
    prices = synthetic_prices()
    expected = build_features(prices)
    shuffled = prices.iloc[::-1].rename(columns=str.upper)
    original = shuffled.copy(deep=True)
    actual = build_features(shuffled)
    pd.testing.assert_frame_equal(actual[0], expected[0])
    pd.testing.assert_series_equal(actual[1], expected[1])
    pd.testing.assert_frame_equal(shuffled, original)


@pytest.mark.parametrize(
    "column, value",
    [
        ("close", np.inf),
        ("volume", -1),
        ("high", 1),
        ("low", 10000),
        ("date", "invalid"),
    ],
)
def test_invalid_price_rows_are_rejected(column, value):
    prices = synthetic_prices()
    if column == "date":
        prices[column] = prices[column].astype(object)
    prices.loc[99, column] = value
    with pytest.raises(ValueError):
        build_features(prices)


def test_duplicate_dates_are_rejected():
    prices = synthetic_prices()
    prices.loc[99, "date"] = prices.loc[98, "date"]
    with pytest.raises(ValueError, match="unique"):
        build_features(prices)


@pytest.mark.parametrize("value, expected", [(-0.01, 0), (0.01, 1), (0, 0)])
def test_direction_probability_handles_single_class(value, expected):
    from backend.model import _up_probability

    features, target = build_features(synthetic_prices())
    target[:] = value
    probability = _up_probability(features, target, features.iloc[[-1]])
    assert probability.tolist() == [expected]


def test_drawdown_includes_initial_capital():
    from backend.backtest import _risk_metrics

    drawdown, _ = _risk_metrics(np.array([-0.1, 0.05]))
    assert drawdown == pytest.approx(-0.1)


def test_five_day_signal_does_not_compound_overlapping_returns():
    from backend.backtest import _metrics

    result = _metrics(
        "five-day",
        pd.Series([0.5, 0.5]),
        np.array([0.1, 0.1]),
        realized_returns=np.array([0.01, -0.02]),
    )
    assert result.mae == pytest.approx(0.4)
    assert result.cumulative_return == pytest.approx(-0.0102)


def test_features_do_not_depend_on_future_prices():
    prices = synthetic_prices()
    before, _ = build_features(prices)
    prices.loc[80:, ["close", "high", "low"]] *= 2
    after, _ = build_features(prices)
    pd.testing.assert_frame_equal(before.loc[:79], after.loc[:79])


def test_backtest_only_trains_on_observable_targets(monkeypatch):
    from types import SimpleNamespace

    from backend import backtest

    class RecordingRegressor:
        def fit(self, features, target):
            self.last_training_row = features.index[-1]
            self.horizon = 5 if target.name == "five_day" else 1
            return self

        def predict(self, features):
            assert self.last_training_row + self.horizon <= features.index[0]
            return np.zeros(len(features))

    original_training_data = backtest._training_data

    def training_data(frame, horizon, features=None):
        x, y = original_training_data(frame, horizon, features)
        return x, y.rename("five_day" if horizon == 5 else "one_day")

    monkeypatch.setattr(backtest, "_training_data", training_data)
    monkeypatch.setattr(backtest, "_regression_model", RecordingRegressor)
    monkeypatch.setattr(
        backtest,
        "predict_return",
        lambda *args: SimpleNamespace(
            validation_mae=1,
            predicted_return=0,
            validation_predictions=np.zeros(
                len(args[0]) - max(30, int(len(args[0]) * 0.8))
            ),
        ),
    )
    monkeypatch.setattr(backtest, "_up_probability", lambda *args: np.array([0.5]))
    backtest.walk_forward_backtest(synthetic_prices(100), min_train_rows=40)

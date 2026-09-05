import numpy as np
import pandas as pd
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
    assert result.training_rows > 40


def test_live_prediction_endpoint_uses_ingested_history(monkeypatch):
    from backend import main

    monkeypatch.setattr(main, "fetch_daily_prices", lambda _: synthetic_prices())
    response = TestClient(app).get("/predict/live/MSFT")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ticker"] == "MSFT"
    assert payload["market_data"]["latest_price"] > 0
    assert payload["market_data"]["as_of"] == "2025-05-20"


def test_walk_forward_backtest_returns_model_and_baselines():
    results = walk_forward_backtest(synthetic_prices(220), min_train_rows=80)

    assert set(results) == {"model", "previous_day", "buy_and_hold"}
    assert all(result.observations > 0 for result in results.values())
    assert all(0 <= result.directional_accuracy <= 1 for result in results.values())
    assert all(result.cumulative_return > -1 for result in results.values())

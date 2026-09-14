import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.data import completed_daily_prices
from backend.main import app
from backend.trader_pipeline import normalize_records


def test_daily_bars_exclude_an_unfinished_early_close():
    prices = pd.DataFrame(
        {"date": ["2025-11-26", "2025-11-28"], "close": [100.0, 101.0]}
    )
    before = completed_daily_prices(prices, pd.Timestamp("2025-11-28T17:59:59Z"))
    after = completed_daily_prices(prices, pd.Timestamp("2025-11-28T18:00:00Z"))
    assert before.close.tolist() == [100.0]
    assert after.close.tolist() == [100.0, 101.0]
    assert len(prices) == 2


def test_frontend_is_served_without_exposing_the_repository():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "Understand the signal." in response.text
    assert client.get("/app.js").status_code == 200
    assert client.get("/styles.css").status_code == 200
    for path in ["/backend/main.py", "/.git/config", "/data/news.sqlite3"]:
        assert client.get(path).status_code == 404


def test_invalid_disclosures_do_not_receive_an_invented_date():
    records = normalize_records(
        "test",
        [
            {"ticker": "SPY", "action": "buy"},
            {"ticker": "SPY", "date": "invalid", "action": "buy"},
            {"ticker": "SPY", "date": "2099-01-01", "action": "buy"},
            {
                "ticker": "SPY",
                "date": "2025-01-01",
                "action": "buy",
                "reported_return": "nan",
            },
        ],
    )
    assert len(records) == 1
    assert records[0].trade_date == "2025-01-01"
    assert records[0].reported_return is None


def test_optional_news_failure_preserves_price_forecast(monkeypatch):
    from backend import main
    from backend.model import Forecast

    prices = pd.DataFrame({"date": ["2025-01-06"], "close": [100]})
    result = Forecast(
        predicted_return=0.01,
        expected_price=101,
        direction="up",
        confidence=0.5,
        training_rows=100,
        metrics={"deep_mae": 0.02, "deep_directional_accuracy": 0.5},
        feature_snapshot={},
        five_day_return=0.02,
        five_day_direction="up",
        direction_probability=0.5,
        deep_predicted_return=0.01,
        hybrid_predicted_return=0.01,
    )
    monkeypatch.setattr(main, "fetch_daily_prices", lambda _: prices)
    monkeypatch.setattr(main, "forecast", lambda _: result)
    monkeypatch.setattr(main.NewsStore, "articles", lambda *_: [])

    def fail(*_):
        raise ValueError("Not enough history for direction")

    monkeypatch.setattr(main, "predict_direction", fail)
    response = TestClient(app).get("/predict/live/SPY?include_news=true")
    assert response.status_code == 200
    assert response.json()["expected_price"] == 101
    assert response.json()["direction_analysis"] is None
    assert "Not enough" in response.json()["direction_error"]

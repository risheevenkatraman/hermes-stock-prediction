from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from backend.main import app
from backend.trader_pipeline import (
    TraderTrade,
    build_recommendations,
    normalize_records,
    pipeline,
)


def test_normalize_records_accepts_common_vendor_fields():
    records = normalize_records(
        "stockcircle",
        {
            "data": [
                {
                    "symbol": " nvda ",
                    "investor": "Taylor",
                    "type": "BUY",
                    "transaction_date": "2026-09-01",
                    "performance": "0.24",
                }
            ]
        },
    )

    assert records == [
        TraderTrade(
            source="stockcircle",
            trader="Taylor",
            ticker="NVDA",
            action="buy",
            trade_date="2026-09-01",
            reported_return=0.24,
            portfolio_weight=None,
            source_url=None,
        )
    ]


def test_recommendations_prioritize_recent_consensus_and_keep_model_optional():
    recent = datetime.now(tz=datetime.timezone.utc).date() - timedelta(days=1)
    records = [
        TraderTrade("quiver_quantitative", "A", "NVDA", "buy", recent.isoformat(), 0.2),
        TraderTrade("stockcircle", "B", "NVDA", "buy", recent.isoformat(), 0.3),
        TraderTrade("tradingview", "C", "NVDA", "sell", "2020-01-01", -0.1),
    ]

    result = build_recommendations(records, include_price_model=False)

    assert result[0].ticker == "NVDA"
    assert result[0].action == "buy"
    assert result[0].trader_count == 3
    assert result[0].sources == (
        "quiver_quantitative",
        "stockcircle",
        "tradingview",
    )
    assert result[0].price_forecast is None


def test_recommendations_endpoint_serializes_cached_records(monkeypatch):
    records = [
        TraderTrade(
            "quiver_quantitative",
            "A",
            "MSFT",
            "buy",
            datetime.now(tz=datetime.timezone.utc).date().isoformat(),
            0.1,
        )
    ]
    monkeypatch.setattr(pipeline, "records", records)
    monkeypatch.setattr(pipeline, "last_refresh", "2026-09-07T00:00:00+00:00")

    response = TestClient(app).get(
        "/recommendations/live?refresh=false&include_price_model=false"
    )

    assert response.status_code == 200
    assert response.json()["recommendations"][0]["ticker"] == "MSFT"

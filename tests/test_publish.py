from types import SimpleNamespace

import pandas as pd
import pytest

from backend import publish, storage


def test_publisher_saves_only_available_horizons_and_preserves_previous_on_failure(
    monkeypatch, tmp_path
):
    monkeypatch.setenv(
        "DATABASE_URL", f"sqlite:///{(tmp_path / 'research.sqlite3').as_posix()}"
    )
    monkeypatch.setenv("NEWS_DB_PATH", str(tmp_path / "news.sqlite3"))
    monkeypatch.delenv("RESEARCH_BUCKET", raising=False)
    monkeypatch.setenv("FORECAST_ARCHIVE_DIR", str(tmp_path / "forecasts"))
    storage.initialize()
    prices = pd.DataFrame(
        {"date": ["2026-01-02", "2026-01-05"], "close": [100.0, 101.0]}
    )
    monkeypatch.setattr(publish, "fetch_daily_prices", lambda _: prices)
    monkeypatch.setattr(
        publish,
        "forecast",
        lambda _: SimpleNamespace(
            expected_price=102, predicted_return=0.01, five_day_return=0.02
        ),
    )
    publish.publish("aapl")
    snapshot = storage.read_record(storage.research, "AAPL")["payload"]
    assert [row["horizon"] for row in snapshot["forecasts"]] == [1, 5]
    assert snapshot["forecasts"][1]["expected_price"] == 103.02
    assert snapshot["market_data"]["latest_price"] == 101
    assert len(snapshot["data_hash"]) == 64
    assert "confidence" not in snapshot
    import json

    record = json.loads(
        next((tmp_path / "forecasts").glob("AAPL/*/record.json")).read_text()
    )
    assert record["eligible_for_prospective_scoring"] is False
    assert record["model_name"] == "legacy_price_hybrid"

    def unavailable(_):
        raise RuntimeError("Provider unavailable")

    monkeypatch.setattr(publish, "fetch_daily_prices", unavailable)
    with pytest.raises(RuntimeError):
        publish.publish("AAPL")
    assert storage.read_record(storage.research, "AAPL")["payload"] == snapshot

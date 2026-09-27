from types import SimpleNamespace

import pandas as pd
import pytest

from backend import publish, storage


def test_publisher_saves_only_available_horizons_and_preserves_previous_on_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'research.sqlite3').as_posix()}")
    monkeypatch.setenv("NEWS_DB_PATH", str(tmp_path / "news.sqlite3"))
    monkeypatch.delenv("RESEARCH_BUCKET", raising=False)
    storage.initialize()
    prices = pd.DataFrame({"date": ["2026-01-02", "2026-01-05"], "close": [100., 101.]})
    monkeypatch.setattr(publish, "fetch_daily_prices", lambda _: prices)
    monkeypatch.setattr(publish, "forecast", lambda _: SimpleNamespace(expected_price=102, predicted_return=.01, five_day_return=.02))
    publish.publish("aapl")
    snapshot = storage.read_record(storage.research, "AAPL")["payload"]
    assert [row["horizon"] for row in snapshot["forecasts"]] == [1, 5]
    assert snapshot["forecasts"][1]["expected_price"] == 103.02
    assert snapshot["market_data"]["latest_price"] == 101
    assert len(snapshot["data_hash"]) == 64
    assert "confidence" not in snapshot

    def unavailable(_):
        raise RuntimeError("Provider unavailable")
    monkeypatch.setattr(publish, "fetch_daily_prices", unavailable)
    with pytest.raises(RuntimeError):
        publish.publish("AAPL")
    assert storage.read_record(storage.research, "AAPL")["payload"] == snapshot

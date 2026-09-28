import hashlib
import json
from datetime import UTC, datetime

import pandas as pd

from backend import forecast_archive


def test_archive_preserves_each_run_and_marks_late_forecasts(monkeypatch, tmp_path):
    monkeypatch.setenv("FORECAST_ARCHIVE_DIR", str(tmp_path))

    class FixedClock:
        @staticmethod
        def now(tz):
            return datetime(2026, 9, 28, 12, 0, tzinfo=UTC)

    monkeypatch.setattr(forecast_archive, "datetime", FixedClock)
    prices = pd.DataFrame({"date": ["2026-09-25"], "close": [100.0]})
    payload = {
        "ticker": "AAPL",
        "published_at": "2026-09-28T12:00:00+00:00",
        "model_version": "example",
        "market_data": {"as_of": "2026-09-25"},
        "data_hash": hashlib.sha256(prices.to_csv(index=False).encode()).hexdigest(),
        "forecasts": [
            {"horizon": 1, "expected_price": 101},
            {"horizon": 5, "expected_price": 103},
        ],
    }
    first = forecast_archive.archive_forecast(payload, prices)
    second = forecast_archive.archive_forecast(payload, prices)
    assert first != second
    record = json.loads((first / "record.json").read_text())
    assert record["eligible_for_prospective_scoring"] is True
    assert [f["target_date"] for f in record["forecasts"]] == [
        "2026-09-28",
        "2026-10-02",
    ]
    for name, digest in record["files"].items():
        assert hashlib.sha256((first / name).read_bytes()).hexdigest() == digest

    class LateClock:
        @staticmethod
        def now(tz):
            return datetime(2026, 9, 28, 15, 0, tzinfo=UTC)

    monkeypatch.setattr(forecast_archive, "datetime", LateClock)
    late = forecast_archive.archive_forecast(payload, prices)
    assert (
        json.loads((late / "record.json").read_text())[
            "eligible_for_prospective_scoring"
        ]
        is False
    )

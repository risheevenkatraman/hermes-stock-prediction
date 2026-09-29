import hashlib
import json
from datetime import date

import exchange_calendars as xcals
import pandas as pd
import pytest

from benchmarks import sector_data


@pytest.fixture
def history():
    dates = xcals.get_calendar("XNYS").sessions_in_range("2024-01-02", "2024-05-01")
    return pd.DataFrame(
        {"High": 110.0, "Low": 90.0, "Close": 100.0, "Adj Close": 50.0, "Volume": 1000},
        index=dates.tz_localize("America/New_York"),
    )


def test_adjusted_ohlc_and_provider_volume(history):
    frame = sector_data.normalize(history, date(2024, 1, 2), date(2024, 5, 1))
    assert (frame.close == 50).all()
    assert (frame.high == 55).all()
    assert (frame.low == 45).all()
    assert (frame.volume == 1000).all()


@pytest.mark.parametrize("index", [0, 30, -1])
def test_missing_sessions_including_boundaries_fail(history, index):
    with pytest.raises(ValueError):
        sector_data.normalize(
            history.drop(history.index[index]), date(2024, 1, 2), date(2024, 5, 1)
        )


def test_download_freezes_bounded_history(history, monkeypatch, tmp_path):
    calls = []

    class Ticker:
        def __init__(self, symbol):
            pass

        def history(self, **kwargs):
            calls.append(kwargs)
            return history

    monkeypatch.setattr(sector_data.yf, "Ticker", Ticker)
    monkeypatch.setattr(sector_data.yf, "set_tz_cache_location", lambda _: None)
    root = sector_data.download(tmp_path, date(2024, 1, 2), date(2024, 5, 1))
    assert len(calls) == 8
    assert all(
        c["end"] == "2024-05-02" and not c["auto_adjust"] and c["actions"]
        for c in calls
    )
    manifest = json.loads((root / "manifest.json").read_bytes())
    assert manifest["status"] == "complete"
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest


def test_provider_failure_marks_archive_failed(monkeypatch, tmp_path):
    class Ticker:
        def __init__(self, symbol):
            pass

        def history(self, **kwargs):
            raise RuntimeError("unavailable")

    monkeypatch.setattr(sector_data.yf, "Ticker", Ticker)
    monkeypatch.setattr(sector_data.yf, "set_tz_cache_location", lambda _: None)
    with pytest.raises(RuntimeError):
        sector_data.download(tmp_path, date(2024, 1, 2), date(2024, 5, 1))
    manifest = json.loads(next(tmp_path.glob("*/manifest.json")).read_bytes())
    assert manifest["status"] == "failed"


def test_exact_close_boundaries_survive_adjustment_roundoff(history):
    history["Close"] = 34.595001
    history["High"] = history["Close"]
    history["Low"] = history["Close"]
    history["Adj Close"] = 31.85108
    frame = sector_data.normalize(history, date(2024, 1, 2), date(2024, 5, 1))
    assert (frame.high == frame.close).all()
    assert (frame.low == frame.close).all()

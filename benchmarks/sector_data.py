"""Freeze Yahoo Finance stock/sector history for local development, without API keys."""

import argparse
import hashlib
import importlib.metadata
import json
import uuid
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import exchange_calendars as xcals
import pandas as pd
import yfinance as yf

from backend.evaluation import session_frame

SYMBOLS = ("AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "SPY", "XLK", "XLY")


def normalize(history: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Normalize a Yahoo library response; fail on gaps, including boundaries."""
    if start > end or history.empty:
        raise ValueError("Empty history or invalid date window.")
    dates = pd.DatetimeIndex(history.index)
    labels = (
        dates.tz_localize(None).normalize()
        if dates.tz is not None
        else dates.normalize()
    )
    selected = history.loc[(labels.date >= start) & (labels.date <= end)].copy()
    selected["date"] = labels[(labels.date >= start) & (labels.date <= end)].strftime(
        "%Y-%m-%d"
    )
    factor = selected["Adj Close"] / selected["Close"]
    frame = pd.DataFrame(
        {
            "date": selected["date"],
            "close": selected["Adj Close"],
            # Preserve exact close-boundary equality despite float roundoff.
            "high": (selected["High"] * factor).where(
                selected["High"] != selected["Close"], selected["Adj Close"]
            ),
            "low": (selected["Low"] * factor).where(
                selected["Low"] != selected["Close"], selected["Adj Close"]
            ),
            "volume": selected["Volume"],
        }
    ).reset_index(drop=True)
    frame = session_frame(frame)
    calendar = xcals.get_calendar("XNYS", start=str(start), end=str(end))
    expected = calendar.sessions_in_range(str(start), str(end))
    actual = pd.DatetimeIndex(frame.date).tz_localize(None)
    if not actual.equals(expected.tz_localize(None)):
        raise ValueError(
            "History does not cover every requested session, including boundaries."
        )
    frame["date"] = frame.date.dt.strftime("%Y-%m-%d")
    return frame


def download(output_root: Path, start: date, end: date) -> Path:
    if start > end or end >= datetime.now(UTC).date():
        raise ValueError("Use an ordered historical window ending before today.")
    cache = output_root / ".yfinance-cache"
    cache.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache.resolve()))
    directory = (
        output_root
        / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex[:8]}"
    )
    (directory / "raw").mkdir(parents=True, exist_ok=False)
    (directory / "prices").mkdir()
    manifest = {
        "status": "downloading",
        "provider": "Yahoo Finance / yfinance",
        "yfinance_version": importlib.metadata.version("yfinance"),
        "request": {
            "interval": "1d",
            "auto_adjust": False,
            "actions": True,
            "repair": False,
        },
        "symbols": list(SYMBOLS),
        "start": str(start),
        "end": str(end),
        "price_basis": "split_and_dividend_adjusted_close",
        "volume_basis": "Yahoo-provided volume, preserved without additional adjustment",
        "membership_status": "Current sector proxy assignments require separate dated evidence; no historical memberships inferred",
        "raw_scope": "Library-returned history requested only for the development window; adjusted historical values can be revised by provider",
        "retrieved_at": {},
    }
    try:
        for symbol in SYMBOLS:
            history = yf.Ticker(symbol).history(
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                interval="1d",
                auto_adjust=False,
                actions=True,
                repair=False,
                keepna=True,
                raise_errors=True,
                timeout=30,
            )
            # This is the library-returned table, not the original HTTP response.
            history.to_csv(directory / "raw" / f"{symbol}.csv")
            frame = normalize(history, start, end)
            frame.to_csv(directory / "prices" / f"{symbol}.csv", index=False)
            manifest["retrieved_at"][symbol] = datetime.now(UTC).isoformat()
            print(f"{symbol}: {len(frame)} exported sessions", flush=True)
        manifest["status"] = "complete"
    except Exception:
        manifest["status"] = "failed"
        raise
    finally:
        (directory / "ingestion.py").write_bytes(Path(__file__).read_bytes())
        manifest["files"] = {
            p.relative_to(directory).as_posix(): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in directory.rglob("*")
            if p.is_file()
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2024, 9, 12))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 9, 11))
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/sector-datasets")
    )
    args = parser.parse_args()
    print(download(args.output_root, args.start, args.end))


if __name__ == "__main__":
    main()

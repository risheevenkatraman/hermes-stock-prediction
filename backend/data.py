"""Market data ingestion adapters used by the prediction API."""

from __future__ import annotations

from datetime import datetime, timezone

import exchange_calendars as xcals
import pandas as pd
import yfinance as yf


class MarketDataError(RuntimeError):
    """Raised when the configured market-data provider cannot return prices."""


def fetch_daily_prices(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Fetch split-adjusted daily OHLCV history from Yahoo Finance.

    Yahoo Finance is used as the initial zero-key provider. A production
    deployment should add a licensed provider behind this same interface.
    """
    symbol = ticker.strip().upper()
    try:
        history = yf.Ticker(symbol).history(
            period=period,
            interval="1d",
            auto_adjust=True,
            actions=False,
        )
    except Exception as error:
        raise MarketDataError(
            f"Market data provider request failed for {symbol}."
        ) from error

    if history.empty:
        raise MarketDataError(f"No daily market data was returned for {symbol}.")

    history = history.rename(
        columns={
            "Close": "close",
            "High": "high",
            "Low": "low",
            "Volume": "volume",
        }
    )
    required = ["close", "high", "low", "volume"]
    if any(column not in history.columns for column in required):
        raise MarketDataError(f"Market data for {symbol} did not include OHLCV fields.")

    prices = history[required].reset_index()
    date_column = "Date" if "Date" in prices.columns else "Datetime"
    prices = prices.rename(columns={date_column: "date"})
    prices["date"] = pd.to_datetime(prices["date"], utc=True, errors="coerce")
    prices["date"] = prices["date"].dt.strftime("%Y-%m-%d")
    prices = prices.dropna(subset=required + ["date"])
    prices = completed_daily_prices(prices)
    if len(prices) < 60:
        raise MarketDataError(
            f"Only {len(prices)} valid daily rows were returned for {symbol}; 60 are required."
        )
    return prices


def completed_daily_prices(
    prices: pd.DataFrame, now: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Keep only US sessions whose scheduled closing time has passed."""
    # This application forecasts US exchange sessions from completed daily bars.
    dates = pd.DatetimeIndex(pd.to_datetime(prices["date"]))
    if not len(dates):
        raise MarketDataError("No dated price rows were returned.")
    calendar = xcals.get_calendar(
        "XNYS", start=str(dates.min().date()), end=str(dates.max().date())
    )
    closes = calendar.schedule["close"].reindex(dates)
    if closes.isna().any():
        raise MarketDataError(
            "Only US exchange-session daily price history is supported."
        )
    prices = prices.loc[
        (closes <= (pd.Timestamp.now(tz="UTC") if now is None else now)).to_numpy()
    ].reset_index(drop=True)
    return prices


def latest_market_metadata(prices: pd.DataFrame) -> dict[str, str | float]:
    """Return serializable metadata for the latest ingested bar."""
    latest = prices.iloc[-1]
    return {
        "as_of": pd.Timestamp(latest["date"]).date().isoformat(),
        "latest_price": round(float(latest["close"]), 2),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "source": "Yahoo Finance",
        "price_type": "adjusted daily close",
    }

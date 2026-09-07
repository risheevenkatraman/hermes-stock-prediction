"""Trader-flow ingestion and recommendation scoring.

The three vendors expose different products and account tiers.  This module
therefore consumes user-owned JSON endpoints configured through
environment variables instead of pretending that a common undocumented API
exists. Each adapter normalizes its response into :class:`TraderTrade`.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from .data import fetch_daily_prices
from .model import Forecast, forecast


class TraderProviderError(RuntimeError):
    """Raised when a configured trader-data provider cannot be read."""


@dataclass(frozen=True)
class TraderTrade:
    source: str
    trader: str
    ticker: str
    action: str
    trade_date: str
    reported_return: float | None = None
    portfolio_weight: float | None = None
    source_url: str | None = None


@dataclass(frozen=True)
class TraderRecommendation:
    ticker: str
    action: str
    score: float
    trader_signal: float
    trader_return: float | None
    trader_count: int
    sources: tuple[str, ...]
    recent_trades: tuple[dict[str, object], ...]
    price_forecast: dict[str, object] | None


def _as_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_date(value: object) -> str:
    today = datetime.now(tz=timezone.utc).date()
    if value is None or value == "":
        return today.isoformat()
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    return today.isoformat() if pd.isna(parsed) else parsed.date().isoformat()


def normalize_records(source: str, payload: object) -> list[TraderTrade]:
    """Normalize common vendor field names without changing provider data."""
    if isinstance(payload, dict):
        rows = payload.get("data", payload.get("results", payload.get("trades", [])))
    else:
        rows = payload
    if not isinstance(rows, list):
        raise TraderProviderError(f"{source} response must contain a list of trades.")

    records: list[TraderTrade] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker", row.get("symbol", ""))).strip().upper()
        if not ticker:
            continue
        action = str(row.get("action", row.get("type", "unknown"))).strip().lower()
        records.append(
            TraderTrade(
                source=source,
                trader=str(row.get("trader", row.get("investor", "unknown"))),
                ticker=ticker,
                action=action,
                trade_date=_as_date(
                    row.get("trade_date", row.get("date", row.get("transaction_date")))
                ),
                reported_return=_as_float(
                    row.get(
                        "reported_return", row.get("return", row.get("performance"))
                    )
                ),
                portfolio_weight=_as_float(
                    row.get("portfolio_weight", row.get("weight"))
                ),
                source_url=(str(row["url"]) if row.get("url") else None),
            )
        )
    return records


def _configured_provider(
    source: str, url_name: str, key_name: str
) -> Callable[[], list[TraderTrade]] | None:
    url = os.getenv(url_name, "").strip()
    if not url:
        return None
    api_key = os.getenv(key_name, "").strip()

    def fetch() -> list[TraderTrade]:
        headers = {"Accept": "application/json", "User-Agent": "Hermes/0.1"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            headers["X-API-Key"] = api_key
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            raise TraderProviderError(f"{source} request failed.") from error
        return normalize_records(source, payload)

    return fetch


def configured_providers() -> dict[str, Callable[[], list[TraderTrade]]]:
    """Return only providers that have been deliberately configured."""
    provider_specs = (
        ("quiver_quantitative", "QUIVER_QUANT_API_URL", "QUIVER_QUANT_API_KEY"),
        ("stockcircle", "STOCKCIRCLE_API_URL", "STOCKCIRCLE_API_KEY"),
        ("tradingview", "TRADINGVIEW_API_URL", "TRADINGVIEW_API_KEY"),
    )
    return {
        source: fetch
        for source, url_name, key_name in provider_specs
        if (fetch := _configured_provider(source, url_name, key_name)) is not None
    }


def _recency_weight(trade_date: str, today: date) -> float:
    age = max(0, (today - date.fromisoformat(trade_date)).days)
    return max(0.05, 1.0 / (1.0 + age / 30.0))


def _trade_direction(action: str) -> float:
    if any(word in action for word in ("sell", "short", "reduce", "close")):
        return -1.0
    if any(word in action for word in ("buy", "long", "add", "purchase")):
        return 1.0
    return 0.0


def _recommendation(
    ticker: str,
    records: list[TraderTrade],
    price_forecast: Forecast | None,
) -> TraderRecommendation:
    today = datetime.now(tz=timezone.utc).date()
    weighted = [
        (
            _trade_direction(record.action),
            _recency_weight(record.trade_date, today),
            record,
        )
        for record in records
    ]
    denominator = sum(weight for direction, weight, _ in weighted if direction)
    trader_signal = (
        sum(direction * weight for direction, weight, _ in weighted) / denominator
        if denominator
        else 0.0
    )
    returns = [
        record.reported_return
        for record in records
        if record.reported_return is not None
    ]
    trader_return = sum(returns) / len(returns) if returns else None
    model_signal = (
        max(-1.0, min(1.0, price_forecast.five_day_return / 0.05))
        if price_forecast
        else 0.0
    )
    score = 0.65 * trader_signal + 0.35 * model_signal
    action = "buy" if score >= 0.2 else "sell" if score <= -0.2 else "watch"
    recent = tuple(
        asdict(record)
        for record in sorted(records, key=lambda item: item.trade_date, reverse=True)[
            :5
        ]
    )
    model_data = (
        {
            "direction": price_forecast.five_day_direction,
            "predicted_return": price_forecast.five_day_return,
            "confidence": price_forecast.confidence,
        }
        if price_forecast
        else None
    )
    return TraderRecommendation(
        ticker=ticker,
        action=action,
        score=round(score, 4),
        trader_signal=round(trader_signal, 4),
        trader_return=round(trader_return, 6) if trader_return is not None else None,
        trader_count=len({record.trader for record in records}),
        sources=tuple(sorted({record.source for record in records})),
        recent_trades=recent,
        price_forecast=model_data,
    )


def build_recommendations(
    records: Iterable[TraderTrade],
    *,
    price_fetcher: Callable[[str], pd.DataFrame] = fetch_daily_prices,
    include_price_model: bool = True,
    limit: int = 20,
) -> list[TraderRecommendation]:
    """Score normalized trades and optionally blend the existing price model."""
    grouped: dict[str, list[TraderTrade]] = {}
    for record in records:
        grouped.setdefault(record.ticker, []).append(record)
    recommendations: list[TraderRecommendation] = []
    for ticker, ticker_records in grouped.items():
        price_result = None
        if include_price_model:
            try:
                price_result = forecast(price_fetcher(ticker))
            except (ValueError, RuntimeError):
                # A trader signal remains useful when a price provider is stale.
                price_result = None
        recommendations.append(_recommendation(ticker, ticker_records, price_result))
    return sorted(recommendations, key=lambda item: item.score, reverse=True)[:limit]


class TraderPipeline:
    """In-memory refreshable pipeline; persistence can be added independently."""

    def __init__(self) -> None:
        self.records: list[TraderTrade] = []
        self.last_refresh: str | None = None
        self.provider_errors: dict[str, str] = {}

    def refresh(self) -> list[TraderTrade]:
        records: list[TraderTrade] = []
        errors: dict[str, str] = {}
        for source, provider in configured_providers().items():
            try:
                records.extend(provider())
            except TraderProviderError as error:
                errors[source] = str(error)
        self.records = records
        self.provider_errors = errors
        self.last_refresh = datetime.now(timezone.utc).isoformat()
        return records

    def status(self) -> dict[str, object]:
        configured = tuple(sorted(configured_providers()))
        return {
            "configured_sources": configured,
            "records": len(self.records),
            "last_refresh": self.last_refresh,
            "provider_errors": self.provider_errors,
            "refresh_required": not bool(configured),
        }


pipeline = TraderPipeline()

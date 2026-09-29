"""Historical snapshot features; no claim of point-in-time provider price vintages."""

from datetime import date, datetime
from typing import Literal

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .evaluation import session_frame
from .model import FEATURE_COLUMNS, _build_feature_frame

WINDOWS = (21, 63, 126, 252)
CONTEXT_COLUMNS = [f"return_{w}" for w in WINDOWS] + [
    "quarter_to_date",
    "year_to_date",
    "volatility_21",
]


class Membership(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ticker: str = Field(pattern=r"^[A-Z0-9][A-Z0-9.-]{0,9}$")
    proxy: str = Field(pattern=r"^[A-Z0-9][A-Z0-9.-]{0,9}$")
    effective_from: date
    available_at: datetime
    source: str = Field(min_length=1)

    @model_validator(mode="after")
    def timezone_required(self):
        if self.available_at.tzinfo is None:
            raise ValueError("Membership availability requires a timezone.")
        return self


class ContextSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    mode: Literal["market_only", "sector", "industry", "fixed_sector"]
    market_proxy: str = Field(pattern=r"^[A-Z0-9][A-Z0-9.-]{0,9}$")
    price_basis: Literal["split_and_dividend_adjusted_close"]
    price_vintage: Literal["revised_historical_snapshot"]
    source: str = Field(min_length=1)
    memberships: list[Membership] = Field(default_factory=list)

    fixed_sector_proxies: dict[str, str] = Field(default_factory=dict)

    @property
    def group_name(self) -> str:
        return "sector" if self.mode == "fixed_sector" else self.mode

    @model_validator(mode="after")
    def check_memberships(self):
        import re

        if self.mode == "fixed_sector":
            if self.memberships or not self.fixed_sector_proxies:
                raise ValueError(
                    "Fixed-sector research requires explicit proxies and no dated memberships."
                )
            if any(
                not re.fullmatch(r"[A-Z0-9][A-Z0-9.-]{0,9}", value)
                for pair in self.fixed_sector_proxies.items()
                for value in pair
            ):
                raise ValueError("Fixed proxies require normalized ticker symbols.")
            return self
        if self.fixed_sector_proxies:
            raise ValueError(
                "Fixed proxies are only permitted in fixed_sector research mode."
            )
        if (self.mode != "market_only") != bool(self.memberships):
            raise ValueError(
                "Sector/industry modes require memberships; market-only mode must have none."
            )
        keys = [(m.ticker, m.effective_from, m.available_at) for m in self.memberships]
        if len(set(keys)) != len(keys):
            raise ValueError("Ambiguous duplicate membership events.")
        return self


def context_returns(frame: pd.DataFrame) -> pd.DataFrame:
    close = frame.close
    result = pd.DataFrame({f"return_{w}": close / close.shift(w) - 1 for w in WINDOWS})
    dates = frame.date.dt.tz_localize(None)
    for period, name in (("Q", "quarter_to_date"), ("Y", "year_to_date")):
        groups = dates.dt.to_period(period)
        # Use the prior period's last close, not the first close of this period.
        start = close.shift(1).where(groups != groups.shift(1)).ffill()
        result[name] = close / start - 1
    result["volatility_21"] = close.pct_change(fill_method=None).rolling(21).std()
    return result


def build_context_features(
    prices: pd.DataFrame,
    ticker: str,
    proxies: dict[str, pd.DataFrame],
    spec: ContextSpec,
):
    frame = session_frame(prices)
    features = _build_feature_frame(frame)[FEATURE_COLUMNS].join(
        context_returns(frame).add_prefix("stock_")
    )
    price_columns = list(features.columns)
    dates = pd.DatetimeIndex(frame.date).tz_localize(None)
    aligned = {}
    needed = {spec.market_proxy} | {
        m.proxy for m in spec.memberships if m.ticker == ticker
    }
    if spec.mode == "fixed_sector":
        if ticker not in spec.fixed_sector_proxies:
            raise ValueError(f"Missing fixed sector proxy for {ticker}")
        needed.add(spec.fixed_sector_proxies[ticker])
    for symbol in sorted(needed):
        if symbol not in proxies:
            raise ValueError(f"Missing proxy prices: {symbol}")
        proxy = session_frame(proxies[symbol])
        values = context_returns(proxy)
        values.index = pd.DatetimeIndex(proxy.date).tz_localize(None)
        aligned[symbol] = values.reindex(dates).set_axis(frame.index)
    market = aligned[spec.market_proxy]
    features = features.join(market.add_prefix("market_"))
    for column in CONTEXT_COLUMNS:
        features[f"relative_market_{column}"] = (
            features[f"stock_{column}"] - market[column]
        )
    used = pd.Series(None, index=frame.index, dtype=object)
    if spec.mode == "fixed_sector":
        proxy = spec.fixed_sector_proxies[ticker]
        group = aligned[proxy]
        used[:] = proxy
    elif spec.mode in {"sector", "industry"}:
        memberships = sorted(
            (m for m in spec.memberships if m.ticker == ticker),
            key=lambda m: (m.effective_from, m.available_at),
        )
        if not memberships:
            raise ValueError(f"No historical {spec.mode} memberships for {ticker}")
        calendar = xcals.get_calendar("XNYS", start=dates.min(), end=dates.max())
        group = pd.DataFrame(np.nan, index=frame.index, columns=CONTEXT_COLUMNS)
        for i, session in enumerate(dates):
            eligible = [
                m
                for m in memberships
                if m.effective_from <= session.date()
                and pd.Timestamp(m.available_at) <= calendar.session_close(session)
            ]
            if eligible:
                member = eligible[-1]
                group.loc[i] = aligned[member.proxy].loc[i]
                used.loc[i] = member.proxy
    if spec.mode != "market_only":
        features = features.join(group.add_prefix(f"{spec.group_name}_"))
        for column in CONTEXT_COLUMNS:
            features[f"relative_{spec.group_name}_{column}"] = (
                features[f"stock_{column}"] - group[column]
            )
    return frame, features.replace([np.inf, -np.inf], np.nan), price_columns, used

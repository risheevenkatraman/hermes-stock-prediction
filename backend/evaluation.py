"""Price-only, close-to-close forecast evaluation. Never reports tradable P&L.

All horizons share prediction origins. A label may enter training only once its
endpoint is observed; missing exchange sessions are errors, not shorter horizons.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .model import (
    FEATURE_COLUMNS,
    _build_feature_frame,
    _regression_model,
    _validate_prices,
)


class EvaluationProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(pattern=r"^[a-z0-9-]+$")
    symbols: list[str] = Field(min_length=1)
    horizons: list[int] = Field(default_factory=lambda: [1, 2, 3, 4, 5])
    development_start: date
    development_end: date
    holdout_start: date
    holdout_end: date
    min_train_rows: int = Field(default=252, ge=40)
    step: int = Field(default=1, ge=1)
    calendar: Literal["XNYS"] = "XNYS"
    candidate: Literal["price_gbt_v1"] = "price_gbt_v1"
    primary_metric: Literal["return_mae"] = "return_mae"
    notes: str = ""

    @model_validator(mode="after")
    def check_protocol(self):
        import re

        if len(set(self.symbols)) != len(self.symbols) or any(
            not re.fullmatch(r"[A-Z0-9][A-Z0-9.-]{0,9}", symbol)
            for symbol in self.symbols
        ):
            raise ValueError("Symbols must be unique normalized US ticker symbols.")
        if self.horizons != [1, 2, 3, 4, 5]:
            raise ValueError("This protocol requires every horizon from 1 to 5.")
        if (
            not self.development_start
            <= self.development_end
            < self.holdout_start
            <= self.holdout_end
        ):
            raise ValueError(
                "Development and holdout dates must be ordered and disjoint."
            )
        return self


def session_frame(prices: pd.DataFrame, calendar_name: str = "XNYS") -> pd.DataFrame:
    frame = _validate_prices(prices)
    if "date" not in frame or frame.date.isna().any():
        raise ValueError("Evaluation requires a valid date for every daily bar.")
    dates = pd.DatetimeIndex(frame.date).tz_convert(None)
    if not dates.equals(dates.normalize()):
        raise ValueError("Daily bar dates must be session dates at midnight.")
    calendar = xcals.get_calendar(calendar_name, start=dates.min(), end=dates.max())
    expected = calendar.sessions_in_range(dates.min(), dates.max()).tz_localize(None)
    if not dates.equals(expected):
        raise ValueError(
            "History must contain exactly one bar for every exchange session; missing or non-session dates found."
        )
    return frame


def evaluate_symbol(
    prices: pd.DataFrame, symbol: str, protocol: EvaluationProtocol
) -> pd.DataFrame:
    if symbol not in protocol.symbols:
        raise ValueError("Symbol is not in the frozen protocol universe.")
    # Reject holdout input, rather than silently using any part of it for development.
    frame = session_frame(prices, protocol.calendar)
    if frame.date.max().date() >= protocol.holdout_start:
        raise ValueError("Holdout rows are not permitted in a development evaluation.")
    frame = frame.loc[frame.date.dt.date <= protocol.development_end].reset_index(
        drop=True
    )
    if frame.empty or frame.date.max().date() < protocol.development_end:
        # End can fall on a holiday/weekend; require the last scheduled session.
        calendar = xcals.get_calendar(protocol.calendar)
        expected_end = calendar.date_to_session(
            str(protocol.development_end), direction="previous"
        )
        if frame.empty or frame.date.max().date() < expected_end.date():
            raise ValueError("Snapshot does not cover the complete development period.")
    features = _build_feature_frame(frame)
    valid = features.notna().all(axis=1)
    close = frame.close
    targets = {h: close.shift(-h) / close - 1 for h in protocol.horizons}
    candidates = frame.index[(frame.date.dt.date >= protocol.development_start) & valid]
    origins = [i for i in candidates if i + max(protocol.horizons) < len(frame)]
    if not origins:
        raise ValueError("No common forecast origins in this development period.")
    records = []
    for origin in origins[:: protocol.step]:
        for horizon in protocol.horizons:
            known = valid & (frame.index + horizon <= origin)
            train_features = features.loc[known, FEATURE_COLUMNS]
            train_target = targets[horizon].loc[known]
            if len(train_features) < protocol.min_train_rows:
                raise ValueError(
                    f"Insufficient observable training history for {symbol}, horizon {horizon}, {frame.date.iloc[origin].date()}."
                )
            model = _regression_model()
            model.fit(train_features, train_target)
            prediction = float(
                model.predict(features.iloc[[origin]][FEATURE_COLUMNS])[0]
            )
            baseline_predictions = {
                protocol.candidate: prediction,
                "zero_return": 0.0,
                "historical_mean": float(train_target.mean()),
                "previous_horizon": float(
                    close.iloc[origin] / close.iloc[origin - horizon] - 1
                ),
                "always_up": None,
            }
            for name, predicted in baseline_predictions.items():
                if predicted is not None and not np.isfinite(predicted):
                    raise ValueError("Model produced a non-finite forecast.")
                records.append(
                    {
                        "ticker": symbol,
                        "horizon": horizon,
                        "model": name,
                        "origin_date": frame.date.iloc[origin].date().isoformat(),
                        "target_date": frame.date.iloc[origin + horizon]
                        .date()
                        .isoformat(),
                        "training_rows": len(train_features),
                        "training_label_end": frame.date.iloc[
                            train_features.index[-1] + horizon
                        ]
                        .date()
                        .isoformat(),
                        "predicted_return": predicted,
                        "predicted_direction": 1
                        if name == "always_up"
                        else int(np.sign(predicted)),
                        "actual_return": float(targets[horizon].iloc[origin]),
                    }
                )
    return pd.DataFrame(records)


def summarize_predictions(predictions: pd.DataFrame) -> dict:
    def scores(rows):
        actual = rows.actual_return.to_numpy(dtype=float)
        predicted = rows.predicted_return.to_numpy(dtype=float)
        direction = rows.predicted_direction.to_numpy(dtype=int)
        errors = predicted - actual
        has_return = bool(np.isfinite(predicted).all())
        return {
            "observations": len(rows),
            "return_mae": float(np.abs(errors).mean()) if has_return else None,
            "return_rmse": float(np.sqrt(np.square(errors).mean()))
            if has_return
            else None,
            "directional_accuracy": float((direction == np.sign(actual)).mean()),
            "up_fraction": float((actual > 0).mean()),
        }

    per_symbol = {}
    for (ticker, horizon, model), rows in predictions.groupby(
        ["ticker", "horizon", "model"], sort=True
    ):
        per_symbol.setdefault(ticker, {}).setdefault(str(horizon), {})[model] = scores(
            rows
        )
    pooled = {}
    for (horizon, model), rows in predictions.groupby(["horizon", "model"], sort=True):
        pooled.setdefault(str(horizon), {})[model] = scores(rows)
    return {"per_symbol": per_symbol, "pooled": pooled}

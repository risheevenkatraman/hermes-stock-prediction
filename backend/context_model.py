"""Separate return/direction heads for development-only context comparisons."""

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .compact_model import select_compact
from .context_features import ContextSpec, build_context_features
from .evaluation import EvaluationProtocol, session_frame, summarize_predictions
from .model import FEATURE_COLUMNS, _regression_model


def evaluate_context(
    prices,
    ticker: str,
    proxies: dict,
    spec: ContextSpec,
    protocol: EvaluationProtocol,
    *,
    compact_selection: bool = False,
):
    if compact_selection and spec.mode != "fixed_sector":
        raise ValueError(
            "Compact selection currently requires fixed_sector research mode."
        )
    if ticker not in protocol.symbols:
        raise ValueError("Ticker outside protocol universe.")
    for data in [prices, *proxies.values()]:
        checked = session_frame(data)
        if checked.date.max().date() >= protocol.holdout_start:
            raise ValueError("Holdout input is forbidden in development.")
        if checked.date.max().date() < protocol.development_end:
            raise ValueError("Input does not cover development end.")
    prices = prices.loc[
        pd.to_datetime(prices.date, utc=True).dt.date <= protocol.development_end
    ].reset_index(drop=True)
    proxies = {
        k: v.loc[
            pd.to_datetime(v.date, utc=True).dt.date <= protocol.development_end
        ].reset_index(drop=True)
        for k, v in proxies.items()
    }
    frame, features, price_columns, used = build_context_features(
        prices, ticker, proxies, spec
    )
    market_columns = [
        c
        for c in features.columns
        if c in price_columns or c.startswith(("market_", "relative_market_"))
    ]
    variants = [("ridge_price", price_columns)]
    if spec.mode == "market_only":
        variants.append(("ridge_context", market_columns))
    else:
        variants.extend(
            [
                ("ridge_market", market_columns),
                (
                    "ridge_sector" if spec.group_name == "sector" else "ridge_industry",
                    list(features.columns),
                ),
            ]
        )
    valid = features.notna().all(axis=1)
    origins = [
        i
        for i in frame.index
        if valid[i]
        and frame.date.iloc[i].date() >= protocol.development_start
        and i + max(protocol.horizons) < len(frame)
    ]
    if not origins:
        raise ValueError("No common origins after feature warmup.")
    records = []
    for origin in origins[:: protocol.step]:
        for horizon in protocol.horizons:
            targets = frame.close.shift(-horizon) / frame.close - 1
            known = valid & (frame.index + horizon <= origin)
            y = targets.loc[known]
            if len(y) < protocol.min_train_rows:
                raise ValueError(
                    f"Insufficient known labels for {ticker} at {frame.date.iloc[origin]}; need {protocol.min_train_rows}, got {len(y)}"
                )
            labels = (y > 0).astype(int)
            estimates = {}
            for name, columns in variants:
                train, latest = (
                    features.loc[known, columns],
                    features.loc[[origin], columns],
                )
                regressor = make_pipeline(StandardScaler(), Ridge(alpha=100.0))
                predicted = float(regressor.fit(train, y).predict(latest)[0])
                if labels.nunique() == 1:
                    probability = float(labels.iloc[0])
                else:
                    classifier = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(C=0.01, max_iter=1000, random_state=42),
                    )
                    probability = float(
                        classifier.fit(train, labels).predict_proba(latest)[0, 1]
                    )
                estimates[name] = (
                    predicted,
                    probability,
                    1 if probability >= 0.5 else -1,
                )
            gbt = _regression_model().fit(features.loc[known, FEATURE_COLUMNS], y)
            predicted = float(gbt.predict(features.loc[[origin], FEATURE_COLUMNS])[0])
            estimates.update(
                {
                    "price_gbt_v1": (predicted, None, int(np.sign(predicted))),
                    "zero_return": (0.0, None, 0),
                    "historical_mean": (
                        float(y.mean()),
                        float(labels.mean()),
                        1 if labels.mean() >= 0.5 else -1,
                    ),
                    "always_up": (None, 1.0, 1),
                }
            )
            selection = None
            if compact_selection:
                predicted, probability, selection = select_compact(
                    features.loc[known], y, features.loc[[origin]], horizon
                )
                estimates["compact_selected"] = (
                    predicted,
                    probability,
                    1 if probability >= 0.5 else -1,
                )
            for name, (predicted, probability, direction) in estimates.items():
                records.append(
                    {
                        "ticker": ticker,
                        "horizon": horizon,
                        "model": name,
                        "selection": json.dumps(selection)
                        if name == "compact_selected"
                        else None,
                        "origin_date": frame.date.iloc[origin].date().isoformat(),
                        "target_date": frame.date.iloc[origin + horizon]
                        .date()
                        .isoformat(),
                        "training_rows": len(y),
                        "training_label_end": frame.date.iloc[y.index[-1] + horizon]
                        .date()
                        .isoformat(),
                        "context_mode": spec.mode,
                        "sector_proxy"
                        if spec.group_name == "sector"
                        else "industry_proxy": used.iloc[origin],
                        "predicted_return": predicted,
                        "up_probability": probability,
                        "predicted_direction": direction,
                        "actual_return": float(targets.iloc[origin]),
                    }
                )
    return pd.DataFrame(records)


def context_metrics(predictions: pd.DataFrame) -> dict:
    result = summarize_predictions(predictions)
    for keys, rows in predictions.groupby(["ticker", "horizon", "model"]):
        ticker, horizon, model = keys
        result["per_symbol"][ticker][str(horizon)][model].update(
            probability_metrics(rows)
        )
    for (horizon, model), rows in predictions.groupby(["horizon", "model"]):
        result["pooled"][str(horizon)][model].update(probability_metrics(rows))
    return result


def probability_metrics(rows: pd.DataFrame) -> dict:
    probability = rows.up_probability.to_numpy(dtype=float)
    actual = (rows.actual_return.to_numpy() > 0).astype(int)
    if not np.isfinite(probability).all():
        return {"brier_score": None, "binary_direction_accuracy": None}
    return {
        "brier_score": float(np.mean((probability - actual) ** 2)),
        "binary_direction_accuracy": float(np.mean((probability >= 0.5) == actual)),
    }

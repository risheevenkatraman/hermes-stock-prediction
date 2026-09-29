"""Rolling development blocks with fixed models throughout each assessment block."""

import numpy as np
import pandas as pd
from pydantic import Field

from .compact_model import SEARCH, compact_groups, predict_candidate
from .context_features import build_context_features
from .evaluation import EvaluationProtocol, session_frame


class RollingProtocol(EvaluationProtocol):
    validation_rows: int = Field(default=126, ge=15)
    assessment_rows: int = Field(default=63, ge=5)


def block_indices(valid_indices, start, horizon, protocol):
    indices = np.asarray(valid_indices, dtype=int)
    # Common validation/final-fit rows across horizons. No label reaches assessment.
    known = indices[indices + max(protocol.horizons) < start]
    if len(known) < protocol.validation_rows:
        raise ValueError("Insufficient validation history.")
    validation = known[-protocol.validation_rows :]
    training = indices[(indices < validation[0]) & (indices + horizon <= validation[0])]
    if len(training) < protocol.min_train_rows:
        raise ValueError("Insufficient initial training history before validation.")
    return training, validation, known


def evaluate_rolling(prices, ticker, proxies, spec, protocol):
    if (
        spec.mode != "fixed_sector"
        or ticker not in protocol.symbols
        or protocol.step != 1
    ):
        raise ValueError(
            "Rolling evaluation requires fixed-sector mode, protocol ticker and daily step=1."
        )
    for data in [prices, *proxies.values()]:
        checked = session_frame(data)
        if checked.date.max().date() >= protocol.holdout_start:
            raise ValueError("Holdout input is forbidden in development.")
        if checked.date.max().date() < protocol.development_end:
            raise ValueError("Input does not cover development end.")
    trim = lambda f: f.loc[
        pd.to_datetime(f.date, utc=True).dt.date <= protocol.development_end
    ].reset_index(drop=True)
    frame, features, _, used = build_context_features(
        trim(prices), ticker, {k: trim(v) for k, v in proxies.items()}, spec
    )
    valid = features.index[features.notna().all(axis=1)].to_numpy()
    origins = valid[
        (frame.date.iloc[valid].dt.date >= protocol.development_start)
        & (valid + max(protocol.horizons) < len(frame))
    ]
    if not len(origins):
        raise ValueError("No complete assessment origins.")
    rows, audits = [], []
    for offset in range(0, len(origins), protocol.assessment_rows):
        assessment = origins[offset : offset + protocol.assessment_rows]
        start = int(assessment[0])
        block = frame.date.iloc[start].date().isoformat()
        for horizon in protocol.horizons:
            train, validation, known = block_indices(valid, start, horizon, protocol)
            y = frame.close.shift(-horizon) / frame.close - 1
            choices, outputs = {}, {}
            audit = {
                "ticker": ticker,
                "block": block,
                "horizon": horizon,
                "assessment_end": str(frame.date.iloc[assessment[-1]].date()),
                "assessment_rows": len(assessment),
                "fit_rows": len(train),
                "refit_rows": len(known),
                "fit_label_end": str(frame.date.iloc[train[-1] + horizon].date()),
                "validation_start": str(frame.date.iloc[validation[0]].date()),
                "validation_label_end": str(
                    frame.date.iloc[validation[-1] + horizon].date()
                ),
                "refit_label_end": str(frame.date.iloc[known[-1] + horizon].date()),
            }
            for classification in (False, True):
                candidates = (
                    [("prior", None)]
                    if classification
                    else [("zero", None), ("mean", None)]
                )
                grid = (
                    SEARCH["logistic_cs"]
                    if classification
                    else sorted(SEARCH["ridge_alphas"], reverse=True)
                )
                candidates += [(g, v) for g in compact_groups() for v in grid]
                actual = (
                    (y.iloc[validation].to_numpy() > 0).astype(int)
                    if classification
                    else y.iloc[validation].to_numpy()
                )
                losses = []
                for candidate in candidates:
                    pred = predict_candidate(
                        candidate,
                        features.iloc[train],
                        y.iloc[train],
                        features.iloc[validation],
                        classification,
                    )
                    losses.append(
                        float(
                            np.mean(
                                (pred - actual) ** 2
                                if classification
                                else np.abs(pred - actual)
                            )
                        )
                    )
                chosen = {
                    group: min(
                        (i for i, c in enumerate(candidates) if c[0] == group),
                        key=lambda i: losses[i],
                    )
                    for group in compact_groups()
                }
                chosen["selected"] = int(np.argmin(losses))
                for group, index in chosen.items():
                    outputs[(group, classification)] = predict_candidate(
                        candidates[index],
                        features.iloc[known],
                        y.iloc[known],
                        features.iloc[assessment],
                        classification,
                    )
                key = "classification" if classification else "regression"
                choices[key] = {group: candidates[i] for group, i in chosen.items()}
                audit[key] = {
                    "chosen": choices[key],
                    "candidates": [
                        {"group": g, "regularization": v, "loss": loss}
                        for (g, v), loss in zip(candidates, losses, strict=True)
                    ],
                }
            audits.append(audit)
            mean, prior = float(y.iloc[known].mean()), float((y.iloc[known] > 0).mean())
            for position, origin in enumerate(assessment):
                estimates = {
                    f"compact_{g}": (
                        outputs[(g, False)][position],
                        outputs[(g, True)][position],
                    )
                    for g in [*compact_groups(), "selected"]
                }
                estimates.update(
                    zero_return=(0.0, None),
                    historical_mean=(mean, prior),
                    always_up=(None, 1.0),
                )
                for model, (predicted, probability) in estimates.items():
                    rows.append(
                        {
                            "ticker": ticker,
                            "horizon": horizon,
                            "block": block,
                            "model": model,
                            "origin_date": str(frame.date.iloc[origin].date()),
                            "target_date": str(
                                frame.date.iloc[origin + horizon].date()
                            ),
                            "training_rows": len(known),
                            "training_label_end": audit["refit_label_end"],
                            "predicted_return": predicted,
                            "up_probability": probability,
                            "predicted_direction": 0
                            if probability is None
                            else 1
                            if probability >= 0.5
                            else -1,
                            "actual_return": float(y.iloc[origin]),
                            "context_mode": spec.mode,
                            "sector_proxy": used.iloc[origin],
                        }
                    )
    result = pd.DataFrame(rows)
    result.attrs["selection_audit"] = audits
    return result

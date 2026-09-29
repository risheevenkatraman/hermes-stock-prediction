"""Evaluate six/twelve-month research directions on a verified cached snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from backend.context_features import ContextSpec, build_context_features
from backend.long_term import predict_long_term
from backend.research_heads import LONG_FEATURES, ResearchHead, long_term_signal


def evaluate(prices, ticker, proxies, spec, protocol):
    for data in [prices, *proxies.values()]:
        dates = pd.to_datetime(data.date, utc=True)
        if dates.max().date().isoformat() >= protocol["protected_start"]:
            raise ValueError("Protected holdout inputs are forbidden")
        if dates.max().date().isoformat() != protocol["development_end"]:
            raise ValueError("Snapshot must end at the declared development end")
    frame, features, _, _ = build_context_features(prices, ticker, proxies, spec)
    valid = features[LONG_FEATURES].notna().all(axis=1)
    origins = [
        i
        for i in frame.index
        if valid[i]
        and str(frame.date.iloc[i].date()) >= protocol["development_start"]
        and i + max(protocol["horizons"]) < len(frame)
    ]
    records = []
    for origin in origins[:: protocol["step_sessions"]]:
        for h in protocol["horizons"]:
            target = frame.close.shift(-h) / frame.close - 1
            known = valid & (frame.index + h <= origin)
            if int(known.sum()) < protocol["minimum_fit_rows"]:
                raise ValueError("Insufficient matured long-term labels")
            train, y = features.loc[known], target.loc[known]
            model = ResearchHead("long_term").fit(train, y)
            probability = float(model.predict(features.iloc[[origin]])[0])
            records.append(
                {
                    "ticker": ticker,
                    "horizon": h,
                    "origin_date": str(frame.date.iloc[origin].date()),
                    "target_date": str(frame.date.iloc[origin + h].date()),
                    "fit_rows": len(y),
                    "fit_label_end": str(frame.date.iloc[train.index[-1] + h].date()),
                    "probability": probability,
                    "prior_probability": model.training_prior,
                    "actual_positive": int(target.iloc[origin] > 0),
                    "signal": long_term_signal(
                        probability, h, str(frame.date.iloc[origin].date())
                    )["signal"],
                }
            )
    if not records:
        raise ValueError("No common long-term assessment origins")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/sector-datasets/20260928T220147197512Z-37e2dbec"),
    )
    parser.add_argument(
        "--protocol", type=Path, default=Path("benchmarks/protocols/long_term_v1.json")
    )
    parser.add_argument(
        "--context",
        type=Path,
        default=Path("benchmarks/protocols/sector_rolling_context.json"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/long-term-experiments")
    )
    args = parser.parse_args()
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    if (
        manifest["status"] != "complete"
        or manifest["price_basis"] != "split_and_dividend_adjusted_close"
    ):
        raise ValueError("A completed adjusted-price snapshot is required")
    for name, digest in manifest["files"].items():
        path = (args.dataset / name).resolve()
        if (
            not path.is_relative_to(args.dataset.resolve())
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise ValueError("Dataset integrity check failed")
    protocol = json.loads(args.protocol.read_text())
    spec = ContextSpec.model_validate_json(args.context.read_text())
    prices = {
        s: pd.read_csv(args.dataset / "prices" / (s + ".csv"))
        for s in protocol["symbols"] + ["SPY", "XLK", "XLY"]
    }
    records = []
    for ticker in protocol["symbols"]:
        records.extend(
            evaluate(
                prices[ticker],
                ticker,
                {s: prices[s] for s in ("SPY", "XLK", "XLY")},
                spec,
                protocol,
            )
        )
        print("Evaluated " + ticker, flush=True)
    frame = pd.DataFrame(records)
    scores = []
    for (ticker, h), group in frame.groupby(["ticker", "horizon"]):
        scores.append(
            {
                "ticker": ticker,
                "horizon": int(h),
                "observations": len(group),
                "brier": float(
                    brier_score_loss(group.actual_positive, group.probability)
                ),
                "prior_brier": float(
                    brier_score_loss(group.actual_positive, group.prior_probability)
                ),
                "accuracy": float(
                    np.mean((group.probability >= 0.5) == group.actual_positive)
                ),
                "always_up_accuracy": float(group.actual_positive.mean()),
                "uncertain_fraction": float((group.signal == "uncertain").mean()),
            }
        )
    report = {
        "status": "development_only_not_promoted",
        "metrics": scores,
        "overlapping_outcomes": True,
        "news_used": False,
        "fundamentals_used": False,
        "notes": "This is not an independent forward test or a customer recommendation. No current signal is published.",
    }
    run = args.output_root / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8]
    )
    run.mkdir(parents=True)
    frame.to_csv(run / "predictions.csv", index=False)
    outlooks = [
        predict_long_term(
            prices[t], t, {s: prices[s] for s in ("SPY", "XLK", "XLY")}, spec
        )
        for t in protocol["symbols"]
    ]
    (run / "snapshot-outlooks.json").write_text(json.dumps(outlooks, indent=2))
    (run / "report.json").write_text(json.dumps(report, indent=2))
    (run / "protocol.json").write_bytes(args.protocol.read_bytes())
    (run / "context.json").write_bytes(args.context.read_bytes())
    (run / "input-manifest.json").write_bytes(
        (args.dataset / "manifest.json").read_bytes()
    )
    (run / "inputs").mkdir()
    for ticker, data in prices.items():
        data.to_csv(run / "inputs" / (ticker + ".csv"), index=False)
    (run / "source").mkdir()
    for source in [Path(__file__), *Path("backend").glob("*.py")]:
        (run / "source" / source.name).write_bytes(source.read_bytes())
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "files": {
                    str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(run.rglob("*"))
                    if p.is_file()
                }
            },
            indent=2,
        )
    )
    print(run)


if __name__ == "__main__":
    main()

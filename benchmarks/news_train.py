"""Run the predeclared news comparison only after verified coverage is ready."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from backend.context_features import ContextSpec
from benchmarks.news_model import evaluate_news, score_predictions
from benchmarks.news_progress import verify_files


def paired_primary(predictions, symbols, horizon=1):
    rows = predictions[
        (predictions.phase == "assessment") & (predictions.horizon == horizon)
    ].copy()
    rows["loss"] = (rows.probability - (rows.actual_return > 0).astype(int)) ** 2
    paired = rows.pivot(index=["origin_date", "ticker"], columns="model", values="loss")
    expected = {"price_sector_plus_news", "price_sector_only", "historical_class_prior"}
    if not expected <= set(paired.columns) or paired[list(expected)].isna().any().any():
        raise ValueError("Primary comparison lacks matched forecasts")
    dates = sorted(rows.origin_date.unique())
    if len(dates) != 63 or any(
        set(rows.loc[rows.origin_date == d, "ticker"]) != set(symbols) for d in dates
    ):
        raise ValueError("Primary assessment requires 63 matched five-company sessions")
    result = {}
    for comparator in ("price_sector_only", "historical_class_prior"):
        difference = paired["price_sector_plus_news"] - paired[comparator]
        by_date = (
            difference.groupby(level="origin_date").mean().reindex(dates).to_numpy()
        )
        rng = np.random.default_rng(42)
        samples = []
        for _ in range(2000):
            starts = rng.integers(0, len(dates) - 4, size=int(np.ceil(len(dates) / 5)))
            indices = np.concatenate([np.arange(s, s + 5) for s in starts])[
                : len(dates)
            ]
            samples.append(float(by_date[indices].mean()))
        result[comparator] = {
            "mean_loss_difference": float(by_date.mean()),
            "paired_95_interval": np.quantile(samples, [0.025, 0.975]).tolist(),
            "improved_tickers": int(
                (difference.groupby(level="ticker").mean() < 0).sum()
            ),
            "improved_blocks": sum(
                float(by_date[i : i + 21].mean()) < 0 for i in range(0, 63, 21)
            ),
        }
    result["research_gate_passed"] = (
        all(
            result[c]["mean_loss_difference"] < 0
            and result[c]["paired_95_interval"][1] < 0
            for c in ("price_sector_only", "historical_class_prior")
        )
        and result["price_sector_only"]["improved_tickers"] >= 3
        and result["price_sector_only"]["improved_blocks"] >= 2
    )
    result["production_promotion"] = False
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", required=True, type=Path)
    parser.add_argument("--assembly", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/news-experiments")
    )
    args = parser.parse_args()
    verify_files(args.progress)
    readiness = json.loads((args.progress / "report.json").read_text())
    if not readiness["observation_coverage_ready"]:
        parser.exit(
            2, "News coverage is not ready. Continue collecting; no model trained.\n"
        )
    verify_files(args.assembly)
    if (
        hashlib.sha256((args.assembly / "manifest.json").read_bytes()).hexdigest()
        != readiness["assembly_manifest_sha256"]
    ):
        raise ValueError("Progress belongs to another assembly")
    protocol = json.loads((args.progress / "protocol.json").read_text())
    if protocol["name"] != "news-prospective-v1" or protocol["horizons"] != [1, 5]:
        raise ValueError("Unsupported news training protocol")
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    if (
        manifest["status"] != "complete"
        or manifest["price_basis"] != "split_and_dividend_adjusted_close"
    ):
        raise ValueError("Verified adjusted prices are required")
    for name, digest in manifest["files"].items():
        p = (args.dataset / name).resolve()
        if (
            not p.is_relative_to(args.dataset.resolve())
            or hashlib.sha256(p.read_bytes()).hexdigest() != digest
        ):
            raise ValueError("Price dataset integrity failure")
    phases = {
        name: [
            r["date"]
            for r in readiness["session_monitoring"]
            if r["qualifies"] and info["first_date"] <= r["date"] <= info["last_date"]
        ]
        for name, info in readiness["phases"].items()
    }
    spec = ContextSpec(
        mode="fixed_sector",
        market_proxy="SPY",
        price_basis="split_and_dividend_adjusted_close",
        price_vintage="revised_historical_snapshot",
        source="Verified news comparison price snapshot",
        fixed_sector_proxies=protocol["fixed_sector_proxies"],
    )
    prices = {
        s: pd.read_csv(args.dataset / "prices" / (s + ".csv"))
        for s in protocol["symbols"] + ["SPY", "XLK", "XLY"]
    }
    events = json.loads((args.assembly / "events.json").read_text())
    predictions = pd.concat(
        [
            evaluate_news(
                prices[t],
                t,
                {s: prices[s] for s in ("SPY", "XLK", "XLY")},
                spec,
                events,
                phases,
                protocol,
            )
            for t in protocol["symbols"]
        ],
        ignore_index=True,
    )
    run = args.output_root / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8]
    )
    run.mkdir(parents=True)
    predictions.to_csv(run / "predictions.csv", index=False)
    (run / "report.json").write_text(
        json.dumps(
            {
                "metrics": score_predictions(predictions),
                "primary_comparison": paired_primary(predictions, protocol["symbols"]),
                "status": "research_only",
                "model_probabilities_calibrated": False,
            },
            indent=2,
        )
    )
    (run / "protocol.json").write_bytes((args.progress / "protocol.json").read_bytes())
    (run / "readiness.json").write_bytes((args.progress / "report.json").read_bytes())
    (run / "events.json").write_bytes((args.assembly / "events.json").read_bytes())
    (run / "input-manifest.json").write_bytes(
        (args.dataset / "manifest.json").read_bytes()
    )
    for ticker, data in prices.items():
        data.to_csv(run / (ticker + "-prices.csv"), index=False)
    (run / "source").mkdir()
    for source in [
        Path(__file__),
        Path("benchmarks/news_model.py"),
        Path("benchmarks/news_assemble.py"),
        *Path("backend").glob("*.py"),
    ]:
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

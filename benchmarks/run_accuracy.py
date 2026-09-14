"""Run with python -m benchmarks.run_accuracy; --cached reuses saved prices."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

# Configure before importing numerical libraries; production functions stay untouched.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pandas as pd

from backend.backtest import serialize_metrics, walk_forward_backtest
from backend.data import fetch_daily_prices
from backend.model import _training_data, _validate_prices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached", action="store_true")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "TSLA"],
    )
    args = parser.parse_args()
    output = Path("benchmarks/results")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "period": "2y",
        "min_train_rows": 252,
        "step": 1,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "torch", "scikit-learn", "yfinance"]
        },
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path("backend").glob("*.py")
        },
        "results": {},
    }
    for symbol in args.tickers:
        if not symbol.isalnum():
            raise ValueError("Benchmark symbols must be alphanumeric.")
        start = time.perf_counter()
        cache = output / f"{symbol}.csv"
        prices = (
            pd.read_csv(cache)
            if args.cached
            else fetch_daily_prices(symbol, period="2y")
        )
        if not args.cached:
            prices.to_csv(cache, index=False)
        frame = _validate_prices(prices)
        x, _ = _training_data(frame, 5)
        result = {
            "data_start": str(frame["date"].iloc[0]),
            "data_end": str(frame["date"].iloc[-1]),
            "first_prediction_date": str(frame["date"].iloc[x.index[252]]),
            "last_prediction_date": str(frame["date"].iloc[x.index[-1]]),
            "data_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
            "strategies": serialize_metrics(
                walk_forward_backtest(prices, min_train_rows=252)
            ),
            "seconds": round(time.perf_counter() - start, 2),
        }
        report["results"][symbol] = result
        (output / "accuracy.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(symbol, json.dumps(result), flush=True)
    strategies = next(iter(report["results"].values()))["strategies"]
    report["pooled"] = {}
    for strategy in strategies:
        values = [r["strategies"][strategy] for r in report["results"].values()]
        observations = sum(v["observations"] for v in values)
        report["pooled"][strategy] = {
            "observations": observations,
            **{
                metric: sum(v[metric] * v["observations"] for v in values)
                / observations
                for metric in ["mae", "directional_accuracy"]
            },
        }
    (output / "accuracy.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("POOLED", json.dumps(report["pooled"]), flush=True)


if __name__ == "__main__":
    main()

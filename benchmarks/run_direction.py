"""Development-only check of the direction head on saved price/news snapshots."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import pandas as pd
from backend.direction_model import predict_direction
from backend.news import NewsArticle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--news-json",
        type=Path,
        help="Normalized JSON object with an articles list and explicit historical available_at timestamps.",
    )
    args = parser.parse_args()
    articles = []
    if args.news_json:
        articles = [
            NewsArticle(**row)
            for row in json.loads(args.news_json.read_text())["articles"]
        ]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scikit-learn", "exchange-calendars"]
        },
        "purpose": "development check; previously inspected price data; no claimed unbiased improvement",
        "method": "60% chronological fitting, 20% policy selection, 20% evaluation; final refit for inference",
        "news_articles": len(articles),
        "news_sha256": (
            hashlib.sha256(args.news_json.read_bytes()).hexdigest()
            if args.news_json
            else None
        ),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path("backend").glob("*.py")
        },
        "results": {},
    }
    for symbol in ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "TSLA"]:
        data_path = Path(f"benchmarks/results/{symbol}.csv")
        data = pd.read_csv(data_path)
        result = predict_direction(
            data, [a for a in articles if a.ticker == symbol], symbol
        )
        report["results"][symbol] = {
            "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            **result,
        }
        metrics = result["evaluation"]
        print(
            f"{symbol}: {metrics['accuracy']:.2%}, always-up {metrics['always_up_accuracy']:.2%}, policy {result['policy']}, news {result['news_status']}",
            flush=True,
        )
    observations = sum(
        r["evaluation"]["observations"] for r in report["results"].values()
    )
    report["pooled"] = {
        "observations": observations,
        **{
            metric: sum(
                r["evaluation"][metric] * r["evaluation"]["observations"]
                for r in report["results"].values()
            )
            / observations
            for metric in ["accuracy", "always_up_accuracy", "brier_score"]
        },
    }
    Path("benchmarks/results/direction_development.json").write_text(
        json.dumps(report, indent=2)
    )


if __name__ == "__main__":
    main()

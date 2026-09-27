"""Run explicitly as a scheduled worker, never from a dashboard request.

Usage: python -m backend.publish --tickers AAPL,MSFT,NVDA
"""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path

from .data import fetch_daily_prices, latest_market_metadata
from .model import forecast
from .news import NewsStore, ticker_symbol
from .storage import initialize, research, save_record
from dataclasses import asdict


def publish(ticker: str):
    symbol = ticker_symbol(ticker)
    prices = fetch_daily_prices(symbol)
    result = forecast(prices)
    now = datetime.now(timezone.utc)
    # Version includes research source content so code changes produce new versions.
    source = b"".join(
        path.read_bytes() for path in sorted(Path(__file__).parent.glob("*.py"))
    )
    payload = {
        "ticker": symbol,
        "published_at": now.isoformat(),
        "expires_at": (now + timedelta(hours=24)).isoformat(),
        "model_version": hashlib.sha256(source).hexdigest()[:12],
        "data_hash": hashlib.sha256(prices.to_csv(index=False).encode()).hexdigest(),
        "market_data": latest_market_metadata(prices),
        "prices": [
            {"date": str(row.date), "close": float(row.close)}
            for row in prices.itertuples()
        ],
        "forecasts": [
            {
                "horizon": 1,
                "expected_price": result.expected_price,
                "predicted_return": result.predicted_return,
            },
            {
                "horizon": 5,
                "expected_price": round(
                    float(prices.iloc[-1].close) * (1 + result.five_day_return), 2
                ),
                "predicted_return": result.five_day_return,
            },
        ],
        "news": [
            asdict(article) for article in NewsStore().articles(symbol)[-6:][::-1]
        ],
        "status": "experimental",
        "limitations": "Price-only experimental model. Calibrated intervals and horizons 2–4 are not available. No demonstrated baseline advantage.",
    }
    # Optional immutable S3 snapshot. AWS credentials come from the worker's IAM role.
    bucket = os.getenv("RESEARCH_BUCKET")
    if bucket:
        import boto3

        prefix = f"research/{symbol}/{now.strftime('%Y%m%dT%H%M%S%fZ')}"
        client = boto3.client("s3")
        client.put_object(
            Bucket=bucket,
            Key=f"{prefix}/snapshot.json",
            Body=json.dumps(payload).encode(),
            ContentType="application/json",
        )
        client.put_object(
            Bucket=bucket,
            Key=f"{prefix}/prices.csv",
            Body=prices.to_csv(index=False).encode(),
            ContentType="text/csv",
        )
    save_record(research, symbol, payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", required=True)
    args = parser.parse_args()
    initialize()
    failed = []
    for ticker in dict.fromkeys(args.tickers.split(",")):
        try:
            publish(ticker)
            print(f"Published {ticker}")
        except Exception:
            failed.append(ticker)
            print(f"Could not publish {ticker}; previous snapshot retained.")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Generate five-horizon candidate forecasts without changing customer snapshots.

Use --data-dir for offline snapshots; --fetch explicitly requests fresh daily bars.
Old snapshots are recorded but marked ineligible for prospective evaluation.
"""

import argparse
import hashlib
import os
from datetime import UTC, datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pandas as pd

from backend.data import (
    completed_daily_prices,
    fetch_daily_prices,
    latest_market_metadata,
)
from backend.evaluation import EvaluationProtocol, session_frame
from backend.forecast_archive import archive_forecast
from backend.model import (
    FEATURE_COLUMNS,
    _build_feature_frame,
    _regression_model,
    _training_data,
)


def shadow_forecast(
    prices: pd.DataFrame,
    symbol: str,
    protocol: EvaluationProtocol,
    output_root: Path,
    *,
    data_source: str = "Supplied CSV snapshot; provider unverified",
) -> Path:
    if symbol not in protocol.symbols:
        raise ValueError("Symbol is not in the protocol universe.")
    frame = session_frame(prices, protocol.calendar)
    frame["date"] = frame.date.dt.strftime("%Y-%m-%d")
    frame = session_frame(completed_daily_prices(frame), protocol.calendar)
    features = _build_feature_frame(frame)
    inference = features.iloc[[-1]][FEATURE_COLUMNS]
    if inference.isna().any().any():
        raise ValueError("Latest daily bar has incomplete features.")
    models, forecasts = {}, []
    for horizon in protocol.horizons:
        x, y = _training_data(frame, horizon, features)
        if len(x) < protocol.min_train_rows:
            raise ValueError("More observed training labels are required.")
        model = _regression_model().fit(x, y)
        predicted = float(model.predict(inference)[0])
        models[horizon] = model
        forecasts.append(
            {
                "horizon": horizon,
                "predicted_return": predicted,
                "expected_price": float(frame.close.iloc[-1]) * (1 + predicted),
                "baselines": {
                    "zero_return": 0.0,
                    "historical_mean": float(y.mean()),
                    "previous_horizon": float(
                        frame.close.iloc[-1] / frame.close.iloc[-1 - horizon] - 1
                    ),
                    "always_up_direction": 1,
                },
                "training_rows": len(x),
                "training_label_end": frame.date.iloc[x.index[-1] + horizon]
                .date()
                .isoformat(),
            }
        )
    source_root = Path(__file__).resolve().parent.parent
    paths = sorted((source_root / "backend").glob("*.py")) + [Path(__file__).resolve()]
    sources = {
        path.relative_to(source_root).as_posix(): path.read_bytes() for path in paths
    }
    source = b"".join(sources.values())
    protocol_bytes = protocol.model_dump_json().encode()
    payload = {
        "ticker": symbol,
        "published_at": datetime.now(UTC).isoformat(),
        "model_name": protocol.candidate,
        "model_version": hashlib.sha256(source + protocol_bytes).hexdigest(),
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "feature_columns": FEATURE_COLUMNS,
        "data_hash": hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest(),
        "market_data": {**latest_market_metadata(frame), "source": data_source},
        "forecasts": forecasts,
        "status": "shadow_only_not_customer_published",
        "limitations": "Experimental price-only candidate. No calibrated intervals or established baseline advantage.",
    }
    return archive_forecast(
        payload,
        frame,
        root=output_root,
        models=models,
        protocol=protocol.model_dump(mode="json"),
        sources=sources,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=Path("benchmarks/protocols/price_v1.json")
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--data-dir", type=Path)
    inputs.add_argument("--fetch", action="store_true")
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/shadow-forecasts")
    )
    args = parser.parse_args()
    protocol = EvaluationProtocol.model_validate_json(args.protocol.read_bytes())
    for symbol in protocol.symbols:
        prices = (
            fetch_daily_prices(symbol)
            if args.fetch
            else pd.read_csv(args.data_dir / f"{symbol}.csv")
        )
        source = (
            "Yahoo Finance / yfinance"
            if args.fetch
            else "Supplied CSV snapshot; provider unverified"
        )
        print(
            shadow_forecast(
                prices, symbol, protocol, args.output_root, data_source=source
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()

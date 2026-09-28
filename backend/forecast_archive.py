"""Append-only by convention forecast records, independent of the latest DB row.

These files are not tamper-proof. S3 object timestamps can provide additional
evidence when archives are uploaded. No claim of public delivery is made here.
"""

import hashlib
import importlib.metadata
import json
import os
import re
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import exchange_calendars as xcals
import pandas as pd


def archive_forecast(
    payload: dict,
    prices: pd.DataFrame,
    *,
    root: Path | None = None,
    models: dict | None = None,
    protocol: dict | None = None,
    sources: dict[str, bytes] | None = None,
) -> Path:
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9.-]{0,9}", payload["ticker"]):
        raise ValueError("Archive ticker must be a normalized symbol, not a path.")
    now = datetime.now(UTC)
    origin = pd.Timestamp(payload["market_data"]["as_of"])
    calendar = xcals.get_calendar("XNYS", start=origin, end=origin + timedelta(days=30))
    origin_session = calendar.date_to_session(origin, direction="none")
    next_session = calendar.next_session(origin_session)
    cutoff = calendar.session_open(next_session)
    generated = pd.Timestamp(payload["published_at"])
    forecasts = [
        {
            **row,
            "target_date": calendar.session_offset(origin_session, row["horizon"])
            .date()
            .isoformat(),
        }
        for row in payload["forecasts"]
    ]
    record = {
        "generated_at": payload["published_at"],
        "origin_date": origin.date().isoformat(),
        "ticker": payload["ticker"],
        "next_session_open": cutoff.isoformat(),
        "model_version": payload["model_version"],
        "data_hash": payload["data_hash"],
        "model_name": payload.get("model_name", "legacy_price_hybrid"),
        "input_source": payload["market_data"].get("source", "unspecified"),
        "feature_columns": payload.get("feature_columns"),
        "protocol_sha256": payload.get("protocol_sha256"),
        "forecasts": forecasts,
        "versions": {
            name: importlib.metadata.version(name)
            for name in [
                "numpy",
                "pandas",
                "scikit-learn",
                "torch",
                "exchange-calendars",
            ]
        },
        "status": "generated_not_delivery_confirmation",
        "limitations": "Local clock and writable files are not tamper-proof. Eligibility checks timing, not predictive accuracy or executable returns.",
    }
    root = root or Path(os.getenv("FORECAST_ARCHIVE_DIR", "data/forecast-history"))
    directory = (
        root
        / payload["ticker"]
        / f"{now.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex[:8]}"
    )
    directory.mkdir(parents=True, exist_ok=False)
    inputs = prices.to_csv(index=False).encode()
    if hashlib.sha256(inputs).hexdigest() != payload["data_hash"]:
        raise ValueError("Forecast archive input hash does not match publication.")
    (directory / "prices.csv").write_bytes(inputs)
    (directory / "snapshot.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    if protocol is not None:
        (directory / "protocol.json").write_text(
            json.dumps(protocol, indent=2, allow_nan=False), encoding="utf-8"
        )
    if models is not None:
        import joblib

        for horizon, model in models.items():
            if horizon not in range(1, 6):
                raise ValueError("Only model horizons 1 to 5 can be archived.")
            joblib.dump(model, directory / f"model-{horizon}.joblib")
        record["model_parameters"] = {
            str(h): model.get_params() for h, model in models.items()
        }
    for relative, content in (sources or {}).items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Source archive paths must stay relative to the archive.")
        destination = directory / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    # Write the manifest last: absent manifest denotes an incomplete archive.
    record["files"] = {
        path.relative_to(directory).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in directory.rglob("*")
        if path.is_file()
    }
    recorded = datetime.now(UTC)
    record["recorded_at"] = recorded.isoformat()
    record["eligible_for_prospective_scoring"] = bool(
        calendar.session_close(origin_session)
        <= generated
        <= pd.Timestamp(recorded)
        < cutoff
    )
    (directory / "record.json").write_text(
        json.dumps(record, indent=2, allow_nan=False), encoding="utf-8"
    )
    return directory

"""Audit shadow archives and score matured outcomes without fitting any model.

Default is an archive-only audit. Supplying --outcomes-dir explicitly opens
prospective outcomes; never use these results to tune the reserved holdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from backend.evaluation import EvaluationProtocol, summarize_predictions


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def utc(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError("An explicit timezone is required.")
    return timestamp.tz_convert("UTC")


def finite(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Forecast values must be numeric.")
    if not math.isfinite(value):
        raise ValueError("Forecast values must be finite.")
    return float(value)


def verify_archive(directory: Path) -> dict:
    """Verify bytes and cross-check saved metadata without unpickling models."""
    record = json.loads((directory / "record.json").read_bytes())
    files = record["files"]
    required = {"prices.csv", "snapshot.json", "protocol.json"} | {
        f"model-{h}.joblib" for h in range(1, 6)
    }
    if not required.issubset(files):
        raise ValueError("Incomplete shadow manifest.")
    root = directory.resolve()
    for name, expected in files.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(root) or path == root:
            raise ValueError("Manifest path escapes the archive.")
        if digest(path.read_bytes()) != expected:
            raise ValueError(f"Hash mismatch: {name}")
    snapshot = json.loads((directory / "snapshot.json").read_bytes())
    protocol = EvaluationProtocol.model_validate_json(
        (directory / "protocol.json").read_bytes()
    )
    for key in (
        "ticker",
        "model_version",
        "model_name",
        "data_hash",
    ):
        if snapshot[key] != record[key]:
            raise ValueError(f"Snapshot mismatch: {key}")
    if record["data_hash"] != files["prices.csv"]:
        raise ValueError("Input digest mismatch.")
    if record["ticker"] not in protocol.symbols:
        raise ValueError("Ticker is outside the saved protocol.")
    if record["model_name"] != protocol.candidate:
        raise ValueError("Candidate differs from saved protocol.")
    if snapshot["status"] != "shadow_only_not_customer_published":
        raise ValueError("Only shadow records are supported.")
    origin = pd.Timestamp(record["origin_date"])
    if origin.tzinfo is not None or origin != origin.normalize():
        raise ValueError("Origin must be a session date.")
    if snapshot["market_data"]["as_of"] != record["origin_date"]:
        raise ValueError("Origin differs from snapshot.")
    prices = pd.read_csv(directory / "prices.csv")
    if pd.Timestamp(prices.date.iloc[-1]).date() != origin.date():
        raise ValueError("Input endpoint differs from origin.")
    calendar = xcals.get_calendar("XNYS", start=origin, end=origin + timedelta(days=30))
    session = calendar.date_to_session(origin, direction="none")
    cutoff = calendar.session_open(calendar.next_session(session))
    generated, recorded = utc(record["generated_at"]), utc(record["recorded_at"])
    if generated != utc(snapshot["published_at"]) or cutoff != utc(
        record["next_session_open"]
    ):
        raise ValueError("Publication timing differs from snapshot/calendar.")
    eligible = bool(calendar.session_close(session) <= generated <= recorded < cutoff)
    if record["eligible_for_prospective_scoring"] is not eligible:
        raise ValueError("Saved eligibility differs from computed timing.")
    # Early offline rehearsals lack baseline/protocol metadata. Their hashes and
    # timing still get checked, but no ineligible record can enter scoring.
    if not eligible:
        return record
    if "protocol_sha256" not in record and "protocol_sha256" not in snapshot:
        # Original archive schema predates saved baselines and protocol digests.
        # Report it explicitly; never backfill predictions using today's model.
        return {**record, "unsupported_schema": True}
    if digest(protocol.model_dump_json().encode()) != record["protocol_sha256"]:
        raise ValueError("Protocol digest mismatch.")
    if snapshot["protocol_sha256"] != record["protocol_sha256"]:
        raise ValueError("Snapshot protocol digest mismatch.")
    if [f["horizon"] for f in record["forecasts"]] != protocol.horizons:
        raise ValueError("All five unique horizons are required.")
    original_forecasts = []
    for forecast in record["forecasts"]:
        target = (
            calendar.session_offset(session, forecast["horizon"]).date().isoformat()
        )
        if forecast["target_date"] != target:
            raise ValueError("Target date differs from exchange calendar.")
        original_forecasts.append(
            {k: v for k, v in forecast.items() if k != "target_date"}
        )
        finite(forecast["predicted_return"])
        for name in ("zero_return", "historical_mean", "previous_horizon"):
            finite(forecast["baselines"][name])
        if (
            forecast["baselines"]["zero_return"] != 0
            or forecast["baselines"]["always_up_direction"] != 1
        ):
            raise ValueError("Invalid fixed baseline.")
        if pd.Timestamp(forecast["training_label_end"]) > origin:
            raise ValueError("Training label extends beyond origin.")
    if original_forecasts != snapshot["forecasts"]:
        raise ValueError("Forecasts differ from hashed snapshot.")
    return record


def select_archives(root: Path, as_of: pd.Timestamp) -> tuple[list, list]:
    """Earliest eligible complete record per ticker/origin/version/protocol.

    Fail closed on corrupt complete archives: silently skipping one could select
    a later prediction after seeing the result. Directories without manifests
    are reported as incomplete and never used.
    """
    selected, audit, unsupported = {}, [], set()
    for directory in sorted(root.glob("*/*")):
        if not directory.is_dir():
            continue
        entry = {"archive": str(directory), "status": "incomplete"}
        audit.append(entry)
        if not (directory / "record.json").is_file():
            continue
        try:
            record = verify_archive(directory)
        except (ValueError, KeyError, OSError, TypeError, IndexError) as error:
            raise ValueError(f"Invalid archive {directory}: {error}") from error
        if not record["eligible_for_prospective_scoring"]:
            entry["status"] = "ineligible_timing"
            continue
        if utc(record["recorded_at"]) > as_of:
            entry["status"] = "recorded_after_as_of"
            continue
        if record.get("unsupported_schema"):
            entry.update(
                status="unsupported_legacy_schema",
                model_version=record["model_version"],
            )
            unsupported.add(
                tuple(record[k] for k in ("ticker", "origin_date", "model_version"))
            )
            continue
        key = tuple(
            record[k]
            for k in ("ticker", "origin_date", "model_version", "protocol_sha256")
        )
        rank = (utc(record["recorded_at"]), str(directory))
        entry["status"] = "superseded_by_earlier_record"
        if key not in selected or rank < selected[key][0]:
            selected[key] = (rank, directory, record, entry)
    rows = []
    for _, directory, record, entry in selected.values():
        if (
            tuple(record[k] for k in ("ticker", "origin_date", "model_version"))
            in unsupported
        ):
            raise ValueError(
                "An eligible legacy record blocks selection of a later record of the same version."
            )
        entry["status"] = "selected"
        rows.append((directory, record))
    return rows, audit


def load_outcomes(root: Path, as_of: pd.Timestamp) -> tuple[dict, dict]:
    metadata = json.loads((root / "metadata.json").read_bytes())
    if metadata["price_basis"] != "split_and_dividend_adjusted_close":
        raise ValueError("Outcomes must use split- and dividend-adjusted closes.")
    if not isinstance(metadata["source"], str) or not metadata["source"].strip():
        raise ValueError("Outcome source is required.")
    if utc(metadata["retrieved_at"]) > as_of:
        raise ValueError("Outcome snapshot was retrieved after as-of.")
    outcomes = {}
    for path in sorted(root.glob("*.csv")):
        if not re.fullmatch(r"[A-Z0-9][A-Z0-9.-]{0,9}", path.stem):
            raise ValueError("Outcome filename must be a normalized ticker.")
        frame = pd.read_csv(path)
        dates = pd.to_datetime(frame["date"], errors="raise")
        values = pd.to_numeric(frame["adjusted_close"], errors="raise")
        if (
            dates.isna().any()
            or dates.dt.tz is not None
            or not dates.equals(dates.dt.normalize())
            or dates.duplicated().any()
            or not np.isfinite(values).all()
            or (values <= 0).any()
        ):
            raise ValueError(f"Invalid outcome rows in {path.name}.")
        if len(dates):
            calendar = xcals.get_calendar(
                "XNYS", start=dates.min(), end=dates.max() + timedelta(days=1)
            )
            if not all(calendar.is_session(date) for date in dates):
                raise ValueError("Outcomes contain non-session dates.")
            if any(
                calendar.session_close(date) > utc(metadata["retrieved_at"])
                for date in dates
            ):
                raise ValueError("Outcome snapshot contains an unfinished session.")
        outcomes[path.stem] = dict(
            zip(dates.dt.strftime("%Y-%m-%d"), values, strict=True)
        )
    return outcomes, metadata


def score_records(
    selected: list, outcomes: dict, as_of: pd.Timestamp
) -> tuple[list, list, dict]:
    observations, predictions = [], []
    for directory, record in selected:
        origin = record["origin_date"]
        closes = outcomes.get(record["ticker"], {})
        calendar = xcals.get_calendar(
            "XNYS", start=origin, end=pd.Timestamp(origin) + timedelta(days=30)
        )
        for forecast in record["forecasts"]:
            row = {
                k: record[k]
                for k in ("ticker", "origin_date", "model_version", "protocol_sha256")
            }
            row.update(
                archive=str(directory),
                horizon=forecast["horizon"],
                target_date=forecast["target_date"],
            )
            if calendar.session_close(forecast["target_date"]) > as_of:
                row["status"] = "pending_session_close"
            elif origin not in closes or forecast["target_date"] not in closes:
                row["status"] = "pending_outcome_data"
            else:
                # Both endpoints MUST come from the same adjusted snapshot.
                # Dividing by the old archived price would invent split/dividend losses.
                actual = closes[forecast["target_date"]] / closes[origin] - 1
                row.update(
                    status="scored",
                    actual_return=actual,
                    outcome_origin_close=closes[origin],
                    outcome_target_close=closes[forecast["target_date"]],
                )
                models = {
                    record["model_name"]: forecast["predicted_return"],
                    **{
                        k: forecast["baselines"][k]
                        for k in ("zero_return", "historical_mean", "previous_horizon")
                    },
                    "always_up": None,
                }
                for name, predicted in models.items():
                    predictions.append(
                        {
                            **row,
                            "model": name,
                            "predicted_return": predicted,
                            "predicted_direction": 1
                            if predicted is None
                            else int(np.sign(predicted)),
                        }
                    )
            observations.append(row)
    metrics = {}
    if predictions:
        frame = pd.DataFrame(predictions)
        for (version, protocol), group in frame.groupby(
            ["model_version", "protocol_sha256"]
        ):
            metrics.setdefault(version, {})[protocol] = summarize_predictions(group)
    return observations, predictions, metrics


def run(
    archive_root: Path,
    output_root: Path,
    *,
    outcomes_dir: Path | None = None,
    as_of: pd.Timestamp | None = None,
) -> Path:
    now = pd.Timestamp.now(tz="UTC")
    as_of = utc(as_of if as_of is not None else now)
    if as_of > now:
        raise ValueError("As-of cannot be in the future.")
    if not archive_root.is_dir():
        raise ValueError("Archive root does not exist.")
    selected, audit = select_archives(archive_root, as_of)
    outcomes, metadata = (
        ({}, None) if outcomes_dir is None else load_outcomes(outcomes_dir, as_of)
    )
    observations, predictions, metrics = score_records(selected, outcomes, as_of)
    directory = (
        output_root
        / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex[:8]}"
    )
    directory.mkdir(parents=True, exist_ok=False)
    evidence = {}
    # Freeze the bytes needed to reproduce each score; do not deserialize model files.
    for index, (source, _) in enumerate(selected):
        target = directory / "archives" / str(index)
        target.mkdir(parents=True)
        evidence[str(source)] = str(target.relative_to(directory))
        for name in ("record.json", "snapshot.json", "prices.csv", "protocol.json"):
            (target / name).write_bytes((source / name).read_bytes())
    if outcomes_dir is not None:
        target = directory / "outcomes"
        target.mkdir()
        for source in [
            outcomes_dir / "metadata.json",
            *sorted(outcomes_dir.glob("*.csv")),
        ]:
            (target / source.name).write_bytes(source.read_bytes())
    report = {
        "mode": "archive_audit" if outcomes_dir is None else "prospective_scoring",
        "as_of": as_of.isoformat(),
        "outcome_metadata": metadata,
        "audit": audit,
        "selected_records": len(selected),
        "observations": observations,
        "predictions": predictions,
        "metrics_by_version_and_protocol": metrics,
        "evidence": evidence,
        "limitations": "Local mutable archives are not tamper-proof. Adjusted outcomes are provider revisions, not executable returns. Prospective results must not be used to tune the reserved holdout. No accuracy claim, calibrated intervals, or customer promotion.",
    }
    (directory / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    statuses = (
        pd.Series([r["status"] for r in observations], dtype=str)
        .value_counts()
        .to_dict()
    )
    (directory / "summary.md").write_text(
        "# Forward forecast evaluation\n\n"
        f"Mode: {report['mode']}. As of: {as_of.isoformat()}.\n\n"
        f"Selected archives: {len(selected)}. Forecast status counts: {json.dumps(statuses)}.\n\n"
        "Per-symbol and pooled metrics are separated by model version, protocol and horizon in report.json. "
        "Pending outcomes are excluded from metrics; all baselines use the same scored rows.\n\n"
        + report["limitations"]
        + "\n",
        encoding="utf-8",
    )
    (directory / "scorer.py").write_bytes(Path(__file__).read_bytes())
    (directory / "evaluation.py").write_bytes(
        (Path(__file__).resolve().parent.parent / "backend/evaluation.py").read_bytes()
    )
    hashes = {
        p.relative_to(directory).as_posix(): digest(p.read_bytes())
        for p in directory.rglob("*")
        if p.is_file()
    }
    (directory / "manifest.json").write_text(
        json.dumps(hashes, indent=2) + "\n", encoding="utf-8"
    )
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-root", type=Path, default=Path("data/shadow-forecasts")
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/forward-scores"))
    parser.add_argument("--outcomes-dir", type=Path)
    parser.add_argument(
        "--as-of", help="Timezone-aware cutoff; defaults to current UTC time."
    )
    args = parser.parse_args()
    print(
        run(
            args.archive_root,
            args.output_root,
            outcomes_dir=args.outcomes_dir,
            as_of=args.as_of,
        )
    )


if __name__ == "__main__":
    main()

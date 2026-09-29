"""Freeze a local context dataset and run paired development experiments."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import uuid
from datetime import UTC, datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pandas as pd

from backend.compact_model import SEARCH, compact_groups
from backend.context_features import ContextSpec, build_context_features
from backend.context_model import context_metrics, evaluate_context
from backend.evaluation import EvaluationProtocol, session_frame
from backend.rolling_model import RollingProtocol, evaluate_rolling


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def run(
    data_dir: Path,
    protocol_path: Path,
    context_path: Path,
    output_root: Path,
    *,
    compact_selection: bool = False,
    rolling: bool = False,
) -> Path:
    protocol_bytes, context_bytes = (
        protocol_path.read_bytes(),
        context_path.read_bytes(),
    )
    if rolling and compact_selection:
        raise ValueError("Choose rolling or compact selection, not both.")
    protocol = (RollingProtocol if rolling else EvaluationProtocol).model_validate_json(
        protocol_bytes
    )
    spec = ContextSpec.model_validate_json(context_bytes)
    directory = (
        output_root
        / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex[:8]}"
    )
    inputs = directory / "inputs"
    inputs.mkdir(parents=True, exist_ok=False)
    (directory / "protocol.json").write_bytes(protocol_bytes)
    (directory / "context.json").write_bytes(context_bytes)
    manifest = {
        "status": "preparing",
        "evaluation": "rolling_blocks" if rolling else "expanding_origins",
        "rolling_search": {
            "search": SEARCH,
            "features": compact_groups(),
            "validation_rows": protocol.validation_rows,
            "assessment_rows": protocol.assessment_rows,
            "refit": "once before each assessment block",
        }
        if rolling
        else None,
        "mode": spec.mode,
        "partition": "development",
        "classification_basis": "Fixed proxy research assumptions; not historical membership evidence"
        if spec.mode == "fixed_sector"
        else "Dated memberships or market-only context",
        "created_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "versions": {
            p: importlib.metadata.version(p)
            for p in (
                "pandas",
                "numpy",
                "scikit-learn",
                "exchange-calendars",
                "pydantic",
            )
        },
        "parameters": {
            "ridge_alpha": 100.0,
            "logistic_C": 0.01,
            "logistic_threshold": 0.5,
        },
        "compact_selection": {"search": SEARCH, "feature_groups": compact_groups()}
        if compact_selection
        else None,
        "calibration": "not calibrated",
        "inputs": {},
    }
    write_json(directory / "manifest.json", manifest)
    try:
        symbols = (
            set(protocol.symbols)
            | {spec.market_proxy}
            | {m.proxy for m in spec.memberships}
            | set(spec.fixed_sector_proxies.values())
        )
        frames = {}
        for symbol in sorted(symbols):
            content = (data_dir / f"{symbol}.csv").read_bytes()
            (inputs / f"{symbol}.csv").write_bytes(content)
            frame = pd.read_csv(inputs / f"{symbol}.csv")
            checked = session_frame(frame)
            if checked.date.max().date() >= protocol.holdout_start:
                raise ValueError("Holdout input is forbidden in development.")
            frames[symbol] = frame
            manifest["inputs"][symbol] = {
                "rows": len(frame),
                "start": str(checked.date.min().date()),
                "end": str(checked.date.max().date()),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        source_root = Path(__file__).resolve().parent.parent
        for path in [
            *sorted((source_root / "backend").glob("*.py")),
            Path(__file__).resolve(),
        ]:
            target = directory / "source" / path.relative_to(source_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
        manifest["status"] = "running"
        write_json(directory / "manifest.json", manifest)
        results = []
        for symbol in protocol.symbols:
            normalized, features, _, used = build_context_features(
                frames[symbol], symbol, frames, spec
            )
            export = features.copy()
            export.insert(0, "date", normalized.date.dt.strftime("%Y-%m-%d").to_numpy())
            export[
                "sector_proxy" if spec.group_name == "sector" else "industry_proxy"
            ] = used
            export.to_csv(directory / f"{symbol}-features.csv", index=False)
            if rolling:
                result = evaluate_rolling(
                    frames[symbol], symbol, frames, spec, protocol
                )
                write_json(
                    directory / f"{symbol}-selection.json",
                    result.attrs.pop("selection_audit"),
                )
            else:
                result = evaluate_context(
                    frames[symbol],
                    symbol,
                    frames,
                    spec,
                    protocol,
                    compact_selection=compact_selection,
                )
            results.append(result)
            print(
                f"{symbol}: {result.origin_date.nunique()} common origins, five horizons",
                flush=True,
            )
        predictions = pd.concat(results, ignore_index=True)
        predictions.to_csv(directory / "predictions.csv", index=False)
        report = {
            "mode": spec.mode,
            "metrics": context_metrics(predictions),
            "classification_basis": manifest["classification_basis"],
            "limitations": "Development-only comparison on revised historical snapshots. Market-only is not sector/industry evidence; sector proxies are not industry indices. Probabilities are uncalibrated; no intervals, long-term forecasts, personal recommendations or trading performance are established.",
        }
        if rolling:
            report["per_block"] = {
                str(block): context_metrics(group)
                for block, group in predictions.groupby("block")
            }
        write_json(directory / "report.json", report)
        manifest["status"] = "complete"
        manifest["completed_at"] = datetime.now(UTC).isoformat()
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error_type"] = type(error).__name__
        raise
    finally:
        manifest["files"] = {
            p.relative_to(directory).as_posix(): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in directory.rglob("*")
            if p.is_file() and p.name != "manifest.json"
        }
        write_json(directory / "manifest.json", manifest)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("benchmarks/results"))
    parser.add_argument(
        "--protocol", type=Path, default=Path("benchmarks/protocols/context_v1.json")
    )
    parser.add_argument(
        "--context", type=Path, default=Path("benchmarks/protocols/market_context.json")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/context-experiments")
    )
    parser.add_argument("--compact-selection", action="store_true")
    parser.add_argument("--rolling", action="store_true")
    args = parser.parse_args()
    print(
        run(
            args.data_dir,
            args.protocol,
            args.context,
            args.output_root,
            compact_selection=args.compact_selection,
            rolling=args.rolling,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

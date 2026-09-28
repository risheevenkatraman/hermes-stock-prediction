"""Reproducible offline development runs from an explicit CSV snapshot directory.

python -m benchmarks.experiment --data-dir benchmarks/results
"""

from __future__ import annotations

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

from backend.evaluation import (
    EvaluationProtocol,
    evaluate_symbol,
    summarize_predictions,
)
from backend.model import FEATURE_COLUMNS, _regression_model


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def write_json(path: Path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def markdown_report(report: dict) -> str:
    lines = [
        "# Price model development evaluation",
        "",
        f"Run: `{report['run_id']}`",
        "",
        "Historical development results, not an untouched holdout or live trading result.",
        "",
        "Returns are decimal fractions in JSON; the table reports MAE in percentage points.",
        "",
        "| Horizon | Model MAE (pp) | Zero-return MAE (pp) | Model direction | Always up | Observations |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for horizon, models in report["metrics"]["pooled"].items():
        candidate = models["price_gbt_v1"]
        lines.append(
            f"| {horizon} | {candidate['return_mae'] * 100:.4f} | {models['zero_return']['return_mae'] * 100:.4f} | {candidate['directional_accuracy']:.2%} | {models['always_up']['directional_accuracy']:.2%} | {candidate['observations']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Each horizon uses the same forecast origins; all target endpoints stay inside development.",
        "- Training includes a label only when its endpoint is known at the forecast origin.",
        "- Always-up is direction-only; its return MAE/RMSE are null, not misleading +/-1 return scores.",
        "- Zero return, historical mean, and previous-horizon return are separate baselines.",
        "- Pooled rows share dates/market exposure; no independence or statistical significance is claimed.",
        "- Adjusted historical prices are provider snapshots, not verified point-in-time histories.",
        "- No trading performance is reported: close-to-close forecasts are not executable at that same close.",
        "- No intervals or calibrated probabilities are claimed. Do not deploy a model on this report alone.",
        "- Candidate and baselines must be compared per ticker and horizon as well as in aggregate.",
        "",
        "The future holdout is reserved in protocol.json and has not been evaluated by this command.",
        "",
    ]
    return "\n".join(lines)


def run_experiment(protocol_path: Path, data_dir: Path, output_root: Path) -> Path:
    protocol_bytes = protocol_path.read_bytes()
    protocol = EvaluationProtocol.model_validate_json(protocol_bytes)
    now = datetime.now(UTC)
    run_id = f"{now.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex[:8]}"
    directory = output_root / run_id
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "inputs").mkdir()
    (directory / "source").mkdir()
    (directory / "protocol.json").write_bytes(protocol_bytes)
    source_root = Path(__file__).resolve().parent.parent
    source_paths = sorted((source_root / "backend").glob("*.py")) + [
        Path(__file__).resolve()
    ]
    source_hashes = {}
    for path in source_paths:
        relative = path.relative_to(source_root)
        content = path.read_bytes()
        source_hashes[relative.as_posix()] = digest(content)
        target = directory / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    manifest = {
        "run_id": run_id,
        "created_at": now.isoformat(),
        "status": "running",
        "partition": "development",
        "protocol_sha256": digest(protocol_bytes),
        "source_sha256": source_hashes,
        "versions": {
            name: importlib.metadata.version(name)
            for name in [
                "numpy",
                "pandas",
                "scikit-learn",
                "exchange-calendars",
                "torch",
                "pydantic",
            ]
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "features": FEATURE_COLUMNS,
        "candidate_parameters": _regression_model().get_params(),
        "inputs": {},
        "holdout_status": "Reserved dates only; no holdout scores or proof of prospective collection.",
    }
    write_json(directory / "manifest.json", manifest)
    results = []
    try:
        # Copy every input before fitting so an edited source CSV cannot alter a run midway.
        for symbol in protocol.symbols:
            content = (data_dir / f"{symbol}.csv").read_bytes()
            (directory / "inputs" / f"{symbol}.csv").write_bytes(content)
            manifest["inputs"][symbol] = {
                "sha256": digest(content),
                "bytes": len(content),
            }
        write_json(directory / "manifest.json", manifest)
        for symbol in protocol.symbols:
            frame = pd.read_csv(directory / "inputs" / f"{symbol}.csv")
            result = evaluate_symbol(frame, symbol, protocol)
            result.to_csv(directory / f"{symbol}-predictions.csv", index=False)
            results.append(result)
            print(
                f"{symbol}: {len(result) // (5 * 5)} common origins, five horizons complete",
                flush=True,
            )
        predictions = pd.concat(results, ignore_index=True)
        predictions.to_csv(directory / "predictions.csv", index=False)
        report = {"run_id": run_id, "metrics": summarize_predictions(predictions)}
        write_json(directory / "report.json", report)
        (directory / "report.md").write_text(markdown_report(report), encoding="utf-8")
        manifest["status"] = "complete"
        manifest["completed_at"] = datetime.now(UTC).isoformat()
        manifest["output_sha256"] = {
            path.name: digest(path.read_bytes())
            for path in directory.iterdir()
            if path.is_file() and path.name != "manifest.json"
        }
        write_json(directory / "manifest.json", manifest)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error_type"] = type(error).__name__
        write_json(directory / "manifest.json", manifest)
        raise
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=Path("benchmarks/protocols/price_v1.json")
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/experiments"))
    args = parser.parse_args()
    print(run_experiment(args.protocol, args.data_dir, args.output_root), flush=True)


if __name__ == "__main__":
    main()

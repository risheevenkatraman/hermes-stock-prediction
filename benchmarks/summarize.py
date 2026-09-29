"""Render saved benchmark measurements without rerunning training."""

import json
from pathlib import Path


def main():
    root = Path("benchmarks/results")
    report = json.loads((root / "accuracy.json").read_text())
    if "pooled" not in report:
        raise ValueError("Benchmark is not complete.")
    lines = [
        "# Current model accuracy benchmark",
        "",
        f"Run: {report['created_at']}",
        "",
        (
            "Yahoo Finance adjusted daily prices, two years per ticker. Expanding-window "
            "walk-forward evaluation starts after 252 usable training rows. All models "
            "refit for each prediction; neural blend weights use only the earlier "
            "selection portion of that training window. No hyperparameters were tuned "
            "against these benchmark results."
        ),
        "",
        "## Pooled next-day results",
        "",
        "| Model / baseline | Direction accuracy | Return MAE (percentage points) | Observations |",
        "|---|---:|---:|---:|",
    ]
    for key, name in [
        ("model", "Statistical"),
        ("deep_model", "Neural"),
        ("hybrid_model", "Hybrid"),
        ("previous_day", "Previous-day return"),
        ("buy_and_hold", "Always up / zero return"),
    ]:
        value = report["pooled"][key]
        lines.append(
            f"| {name} | {value['directional_accuracy']:.2%} | {value['mae'] * 100:.4f} | {value['observations']} |"
        )
    lines += [
        "",
        (
            "The last row uses always-up predictions for direction and zero-return "
            "predictions for MAE; these are two separate baseline definitions."
        ),
        "",
        "## Per-ticker results",
        "",
        "| Ticker | Statistical direction | Neural direction | Hybrid direction | Always up | Hybrid MAE (pp) | Zero-return MAE (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for symbol, result in report["results"].items():
        s = result["strategies"]
        lines.append(
            f"| {symbol} | {s['model']['directional_accuracy']:.2%} | {s['deep_model']['directional_accuracy']:.2%} | {s['hybrid_model']['directional_accuracy']:.2%} | {s['buy_and_hold']['directional_accuracy']:.2%} | {s['hybrid_model']['mae'] * 100:.4f} | {s['buy_and_hold']['mae'] * 100:.4f} |"
        )
    lines += ["", "## Coverage", ""]
    for symbol, result in report["results"].items():
        lines.append(
            f"- {symbol}: data {result['data_start'][:10]} through {result['data_end'][:10]}; prediction dates {result['first_prediction_date'][:10]} through {result['last_prediction_date'][:10]}; {result['strategies']['model']['observations']} forecasts."
        )
    lines += [
        "",
        "## Interpretation and limits",
        "",
        "- MAE measures the absolute next-day return error, not a percentage of correct predictions. Lower is better.",
        "- Direction counts exact sign matches, including flat actual returns.",
        "- The seven symbols share market exposure and dates. Pooled observations are correlated, not independent trials; no statistical significance is claimed.",
        "- This is one recent test window and a selected large-cap/index basket, not evidence across all securities or market regimes.",
        "- The current backtest uses the common one-day/five-day window, omitting the newest four otherwise evaluable one-day forecasts.",
        "- Five-day accuracy is a separate horizon and must not be compared directly with next-day MAE.",
        "- Classifier return MAE in the raw output uses +/-1 direction labels and is not a meaningful return forecast score; use direction accuracy and Brier score instead.",
        "- Raw portfolio returns exclude fees, slippage, and execution delays. Signals use closing-bar information and assume execution at that close; treat those returns as idealized, not a tradable performance estimate.",
        "- Saved CSVs reproduce this provider snapshot. Adjusted history can change later; package versions and source/data hashes are recorded in accuracy.json.",
        "",
        "## Reproduce",
        "",
        "```powershell",
        ".venv\\Scripts\\python.exe -m benchmarks.run_accuracy --cached",
        ".venv\\Scripts\\python.exe -m benchmarks.summarize",
        "```",
        "",
        "Omit `--cached` to fetch a new two-year snapshot. Raw results: `accuracy.json`.",
    ]
    docs_root = Path("docs/benchmarks")
    docs_root.mkdir(parents=True, exist_ok=True)
    (docs_root / "accuracy.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:32]))


if __name__ == "__main__":
    main()

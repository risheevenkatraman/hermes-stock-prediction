# Current model accuracy benchmark

Run: 2026-09-14T00:53:46.185629+00:00

Yahoo Finance adjusted daily prices, two years per ticker. Expanding-window walk-forward evaluation starts after 252 usable training rows. All models refit for each prediction; neural blend weights use only the earlier selection portion of that training window. No hyperparameters were tuned against these benchmark results.

## Pooled next-day results

| Model / baseline | Direction accuracy | Return MAE (percentage points) | Observations |
|---|---:|---:|---:|
| Statistical | 50.06% | 1.5827 | 1568 |
| Neural | 48.47% | 1.5121 | 1568 |
| Hybrid | 50.32% | 1.5351 | 1568 |
| Previous-day return | 50.96% | 2.0222 | 1568 |
| Always up / zero return | 52.10% | 1.4133 | 1568 |

The last row uses always-up predictions for direction and zero-return predictions for MAE; these are two separate baseline definitions.

## Per-ticker results

| Ticker | Statistical direction | Neural direction | Hybrid direction | Always up | Hybrid MAE (pp) | Zero-return MAE (pp) |
|---|---:|---:|---:|---:|---:|---:|
| SPY | 51.79% | 50.00% | 54.02% | 54.46% | 0.6729 | 0.6303 |
| QQQ | 49.11% | 53.57% | 50.89% | 54.46% | 1.0428 | 0.9822 |
| AAPL | 54.91% | 44.20% | 53.12% | 53.57% | 1.1991 | 1.1267 |
| MSFT | 49.11% | 47.77% | 49.11% | 49.55% | 1.5388 | 1.4446 |
| NVDA | 52.23% | 48.21% | 51.34% | 50.89% | 2.0409 | 1.8947 |
| AMZN | 47.32% | 50.00% | 48.21% | 50.89% | 1.7185 | 1.5704 |
| TSLA | 45.98% | 45.54% | 45.54% | 50.89% | 2.5326 | 2.2442 |

## Coverage

- SPY: data 2024-09-12 through 2026-09-11; prediction dates 2025-10-14 through 2026-09-03; 224 forecasts.
- QQQ: data 2024-09-12 through 2026-09-11; prediction dates 2025-10-14 through 2026-09-03; 224 forecasts.
- AAPL: data 2024-09-12 through 2026-09-11; prediction dates 2025-10-14 through 2026-09-03; 224 forecasts.
- MSFT: data 2024-09-12 through 2026-09-11; prediction dates 2025-10-14 through 2026-09-03; 224 forecasts.
- NVDA: data 2024-09-12 through 2026-09-11; prediction dates 2025-10-14 through 2026-09-03; 224 forecasts.
- AMZN: data 2024-09-12 through 2026-09-11; prediction dates 2025-10-14 through 2026-09-03; 224 forecasts.
- TSLA: data 2024-09-12 through 2026-09-11; prediction dates 2025-10-14 through 2026-09-03; 224 forecasts.

## Interpretation and limits

- MAE measures the absolute next-day return error, not a percentage of correct predictions. Lower is better.
- Direction counts exact sign matches, including flat actual returns.
- The seven symbols share market exposure and dates. Pooled observations are correlated, not independent trials; no statistical significance is claimed.
- This is one recent test window and a selected large-cap/index basket, not evidence across all securities or market regimes.
- The current backtest uses the common one-day/five-day window, omitting the newest four otherwise evaluable one-day forecasts.
- Five-day accuracy is a separate horizon and must not be compared directly with next-day MAE.
- Classifier return MAE in the raw output uses +/-1 direction labels and is not a meaningful return forecast score; use direction accuracy and Brier score instead.
- Raw portfolio returns exclude fees, slippage, and execution delays. Signals use closing-bar information and assume execution at that close; treat those returns as idealized, not a tradable performance estimate.
- Saved CSVs reproduce this provider snapshot. Adjusted history can change later; package versions and source/data hashes are recorded in accuracy.json.

## Reproduce

```powershell
.venv\Scripts\python.exe -m benchmarks.run_accuracy --cached
.venv\Scripts\python.exe -m benchmarks.summarize
```

Omit `--cached` to fetch a new two-year snapshot. Raw results: `accuracy.json`.

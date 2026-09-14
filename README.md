# Hermes Developer Log

Hermes is a stock intelligence application that combines market data, price
features, machine learning forecasts, and a dashboard UI. This document tracks
the implementation status and provides the commands needed to run the project.

## Implementation log

### Completed

- Created the responsive Hermes dashboard using HTML, CSS, and JavaScript.
- Added browser-persisted saved tickers with add/remove and quick analysis.
- Added a FastAPI prediction service.
- Added Pandas-based OHLCV validation and feature engineering:
  - One-, five-, and twenty-day returns.
  - Five- and twenty-day moving-average ratios.
  - Rolling volatility.
  - Daily high-low range.
  - Volume change.
  - Fourteen-day RSI.
- Added a scikit-learn gradient-boosting regression model for next-trading-day
  return estimates.
- Added a five-trading-day return target to reduce reliance on noisy one-day
  movement.
- Added a directional classifier probability and holdout directional-accuracy
  metric alongside regression metrics.
- Added chronological holdout validation, confidence scoring, and prediction
  metadata.
- Added Yahoo Finance daily market-data ingestion through `yfinance`.
- Switched ingestion to split-adjusted OHLCV history with `auto_adjust=True`.
- Added the `GET /predict/live/{ticker}` endpoint.
- Connected the dashboard's **Analyze** button to the live SPY
  prediction endpoint.
- Added ticker selection with free-form input and popular ticker presets.
- Connected the selected ticker to live prediction requests and the backtest
  basket.
- Added walk-forward backtesting that compares:
  - Hermes model predictions.
  - Previous-day-return baseline.
  - Buy-and-hold baseline.
- Added backtest metrics for mean absolute error, directional accuracy, and
  cumulative return.
- Added out-of-sample five-day regression and direction-classifier evaluation,
  including Brier score, maximum drawdown, and annualized volatility.
- Added automated tests for feature generation, forecasts, the live endpoint,
  and backtesting.
- Added multi-ticker, multi-period backtesting through
  `GET /backtest/live?tickers=SPY,QQQ&period=1y`.
- Added a dashboard model-performance section that loads aggregate and
  per-ticker backtest results from the batch endpoint.
- Added a deterministic PyTorch neural regressor over the existing price
  features. Live forecasts now expose both the neural prediction and a
  validation-gated statistical-neural hybrid prediction; the response also
  includes held-out neural validation metrics.
- Added a refreshable trader-flow pipeline that normalizes Quiver Quantitative,
  Stockcircle, and TradingView records, weights recent buy/sell consensus and
  reported trader returns, and optionally blends the existing price model into
  recommendations.
- Added `/trader-pipeline/status`, `/trader-pipeline/refresh`, and
  `/recommendations/live` endpoints. Set
  `TRADER_REFRESH_INTERVAL_MINUTES` to enable in-process periodic refreshes.
- Added out-of-sample walk-forward metrics for the statistical, deep-learning,
  and hybrid strategies so blend quality is measured on unseen observations.
- The hybrid chooses a neural weight from 0%, 10%, 20%, and 30% by
  evaluating the blend on an earlier chronological selection window. Later
  holdout observations remain reserved for evaluation.
- Formatted the frontend with Prettier and the Python code with Black.

## Remaining work

### Data and modeling

- Calibrate classifier probabilities and compare model performance across more
  market regimes.
- Add model versioning, experiment tracking, and persisted trained models.
- Add data caching and rate-limit handling for the market-data provider.
- Replace the initial zero-key provider with a licensed production market-data
  provider before deployment.

### Product features

- Expand the new company-news pipeline with verified historical coverage and
  independent out-of-sample evaluation.
- Add public trader and institutional-flow data as model features.
- Connect budget-aware recommendations to live predictions and portfolio rules.
- Add authentication, persistent watchlists, and user settings.
- Add production logging, monitoring, and error reporting.

## Project layout

```text
backend/
  backtest.py       Walk-forward evaluation and benchmark strategies
  data.py           Market-data provider adapter
  main.py           FastAPI routes and request validation
  model.py          Feature engineering and forecasting model
  trader_pipeline.py
                    Provider adapters, normalization, scoring, and refresh state
tests/
  test_model.py     Prediction and backtest tests
app.js              Dashboard interactions and API integration
index.html          Hermes dashboard markup
styles.css          Dashboard styling and responsive layout
requirements.txt    Runtime Python dependencies
requirements-dev.txt
                    Runtime and test dependencies
```

## Run the software

### 1. Create or select a Python environment

Python 3.11 or newer is recommended.

### 2. Install runtime dependencies

From the project root:

```bash
python -m pip install -r requirements.txt
```

For development and tests:

```bash
python -m pip install -r requirements-dev.txt
```

### 3. Start the prediction API

```bash
python -m uvicorn backend.main:app --reload
```

The API runs at `http://localhost:8000`.

- Health check: `GET /health`
- Manual prediction: `POST /predict`
- Live prediction: `GET /predict/live/{ticker}`
- Trader pipeline status: `GET /trader-pipeline/status`
- Refresh trader records: `POST /trader-pipeline/refresh`
- Recommendations: `GET /recommendations/live`
- Live backtest: `GET /backtest/live/{ticker}`
- Multi-ticker backtest: `GET /backtest/live?tickers=SPY,QQQ&period=5y`
- Interactive API docs: `http://localhost:8000/docs`

### 4. Open the dashboard

Open **http://127.0.0.1:8000/** after starting the API. FastAPI serves the
three frontend files explicitly; the repository and local news archive are
not exposed. No separate frontend build or static server is required.

The dashboard loads the selected ticker's completed daily prices, return
forecast, experimental direction analysis, and saved news independently.
Use **Analyze** to refresh or change tickers. **Save ticker** persists a ticker
in this browser; registration and cross-device accounts remain a clearly
labeled preview.

Use **Connection settings** if your API is at a different address. Opening
`index.html` directly is also supported, with `http://localhost:8000` as the
default API address. A connected API does not imply a configured news or
trader provider; each section reports its own availability and errors.

### Trader-flow provider configuration

The provider adapters are intentionally disabled until their licensed
endpoints are configured. Use the endpoint and credentials supplied by each
vendor; Hermes does not scrape private pages or ship credentials:

```text
QUIVER_QUANT_API_URL=...
QUIVER_QUANT_API_KEY=...
STOCKCIRCLE_API_URL=...
STOCKCIRCLE_API_KEY=...
TRADINGVIEW_API_URL=...
TRADINGVIEW_API_KEY=...
TRADER_REFRESH_INTERVAL_MINUTES=60
```

Each response should be a JSON list, or an object containing `data`, `results`,
or `trades`. Records should include a ticker/symbol, action/type, and date;
trader/investor and reported return/performance fields are used when present.
TradingView data must come from an account-approved export or API endpoint.

### 5. Run tests

```bash
python -m pytest tests -q
```

## Model implementation notes

Feature calculations are shared across forecast horizons. Tree models operate
on unscaled features and use explicit chronological validation. Input columns
are normalized before validation; malformed OHLCV values and duplicate or
partially invalid dates are rejected rather than silently dropping bars.
Omitting dates for every row preserves the supplied row order.

The direction probability estimates a positive **next-day** return. Confidence
is a heuristic score, not a calibrated probability. Forecast MAE, R?, and
directional accuracy now describe the selected hybrid on the final evaluation
window. Statistical and neural MAE use those same evaluation rows.

The neural regressor standardizes both features and targets using only its
training window, then refits preprocessing and weights on all available rows
for inference. It uses a private CPU random generator and leaves process RNG
and thread settings unchanged. For small CPU workloads, configure
`OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` before starting Python to avoid
excessive thread overhead; tune these for the deployment workload.

Both models train on the first 80% of usable history (at least 30 rows). The
first half of the remaining holdout selects a blend from 0%, 10%, 20%, and 30%
neural weight; the second half evaluates it without influencing selection.
Live predictions and walk-forward backtests share this selection logic. Small
histories yield very small evaluation windows, so these scores are noisy.
The feed-forward network consumes engineered daily features, not raw sequences;
these implementation changes do not establish improved market performance.

Five-day backtests train only on targets whose ending prices are already known.
Their MAE and directional accuracy evaluate five-day forecasts, while portfolio
returns use the signal to hold or exit for the next day. This avoids compounding
overlapping five-day returns. With `step > 1`, portfolio metrics describe only
sampled one-day trades, not continuous buy-and-hold performance.

## Current limitations

The current forecast and trader recommendations are educational research
features, not financial advice. Disclosed trades can be delayed, incomplete,
or survivorship-biased, and the pipeline does not infer that a trader will
repeat past returns. Yahoo Finance and configured trader providers are suitable
for this development stage, but production use requires appropriate licensing,
availability guarantees, credential storage, caching, and data-quality
monitoring.


## Accuracy benchmark

Run the reproducible seven-symbol walk-forward benchmark with:

```powershell
.venv\Scripts\python.exe -m benchmarks.run_accuracy
.venv\Scripts\python.exe -m benchmarks.summarize
```

The runner saves two years of provider data per symbol, uses 252 initial usable
training rows, and evaluates daily predictions without tuning on test outcomes.
Use `--cached` to reuse the downloaded CSVs. See
[the benchmark report](benchmarks/results/accuracy.md) for accuracy, baselines,
coverage, and limitations; `accuracy.json` records versions and source hashes.


## Company news and direction analysis

The dashboard now supports ticker-specific news scans and an experimental
classification head that learns subsequent market direction from prices,
financial sentiment, and headline/summary text. It reports held-out accuracy
against always-up and explicitly falls back when historical news is missing.
The original hybrid return forecast remains available alongside it.

See [setup, JSON import, timing rules, and the improvement plan](docs/news-direction.md).
Set `ALPHAVANTAGE_API_KEY` for live scans or import a verified historical archive.
News contribution has not yet been benchmarked on real historical articles.


## Dashboard verification

The dashboard contains no demonstration prices, sentiment scores, buy ideas,
portfolio balances, or fabricated provider activity. Account creation is the
only feature preview. See [the frontend guide](docs/frontend.md) for behavior,
source limitations, and browser-test commands.

Browser interaction tests use isolated API fixtures only inside the test suite;
the running app never falls back to those fixtures or saved benchmark data.
On Windows tests use installed Microsoft Edge. On other platforms install the
Playwright Chromium engine first:

```bash
python -m playwright install chromium
python -m pytest tests/test_frontend.py -q
```

# Hermes Developer Log

Hermes is a stock intelligence application that combines market data, price
features, machine learning forecasts, and a dashboard UI. This document tracks
the implementation status and provides the commands needed to run the project.

## Implementation log

### Completed

- Created the responsive Hermes dashboard using HTML, CSS, and JavaScript.
- Added editable investment budget and watchlist interactions.
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
- Connected the dashboard's **Refresh analysis** button to the live SPY
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

- Replace representative news cards with a news ingestion and sentiment
  pipeline.
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
- Live backtest: `GET /backtest/live/{ticker}`
- Multi-ticker backtest: `GET /backtest/live?tickers=SPY,QQQ&period=5y`
- Interactive API docs: `http://localhost:8000/docs`

### 4. Open the dashboard

Open [index.html](index.html) directly in a browser, or serve the project
directory with a static server:

```bash
npx serve .
```

With the API running, click **Refresh analysis** in the dashboard to fetch
current SPY history and run a live prediction.

### 5. Run tests

```bash
python -m pytest tests -q
```

## Current limitations

The current forecast is an educational research feature, not financial advice.
The model uses price and volume history only; news sentiment and trader-flow
data are not yet connected. Yahoo Finance is suitable for this development
stage, but production use requires a provider with appropriate licensing,
availability guarantees, and data-quality monitoring.

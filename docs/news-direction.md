# News sentiment and direction prediction

The existing return ensemble minimizes return MAE. A lower MAE does not imply
better up/down classification. Hermes now has a separate experimental direction
head, combining a regularized price classifier with a classifier trained on
price features, company-specific sentiment, and article text. Return forecasts
keep their existing meaning. `direction_analysis` describes the new direction
head and may disagree with the return forecast.

## How the pipeline works

1. `POST /news/refresh/AAPL` requests ticker-tagged news from Alpha Vantage.
   Company-specific relevance below 0.2 is discarded. Sentiment comes from the
   provider's ticker-specific score, not the article's overall score.
2. The local SQLite archive retains the first observed version of each URL and
   ticker. Tracking parameters are removed for deduplication, but content IDs
   are preserved. Identical headlines are counted once during aggregation.
3. Each historical row receives only news published and available by its actual
   US market close, including early closes and daylight-saving changes. The
   feature window covers the preceding three calendar days, with exponential
   age weighting and relevance weights. News count, positive/negative share,
   mean sentiment, and disagreement are numerical features.
4. The news-response classifier also reads headline and summary TF-IDF features.
   Vocabulary, scaling, and model weights are fit on the training window only.
   Its labels are actual next-trading-day close-to-close outcomes (`return > 0`).
   It therefore learns associations between earlier news and subsequent movement;
   positive tone is not automatically treated as a positive price forecast.
5. The first 60% of usable price rows fit the classifiers; the next 20% select
   news weight (0%, 25%, or 50%) and a direction threshold (0.45, 0.50, or 0.55).
   An always-up policy is an explicit candidate and wins ties. The final 20%
   evaluates that frozen choice against always-up. At least 120 usable rows
   and 30 news-covered days in the initial fit are required for news modeling.
6. For inference, eligible models refit on all known labels. No-news inference
   receives zero news weight; the chosen threshold remains fixed. Training and
   evaluation use the same no-news fallback. Policy selection is never rerun on
   evaluation outcomes. Probabilities are currently uncalibrated.

The provider analyzes news sentiment; the local text model learns market
response from the supplied headlines and summaries. Hermes does not download
full article bodies or implement a local FinBERT scorer in this version.
Text tone and next-day performance are different targets. Price responses may
already have happened before the forecast cutoff, and these associations are
not estimates of causal news impact or recommendations to buy/sell.

## Configure and run

Install updated requirements, set your own key in the server environment,
then start the API. No API key is stored in source or returned by status calls.

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:ALPHAVANTAGE_API_KEY = 'your-key'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

The archive defaults to `data/news.sqlite3` (ignored by Git). Override it with
`NEWS_DB_PATH`. In the dashboard, select a ticker and use **Scan latest news**.
**Recheck direction** uses the saved archive; **Analyze** loads this direction
head independently alongside the original return forecast. Refreshing prices
or analyzing direction does not repeatedly charge news-provider requests.

| Endpoint | Behavior |
|---|---|
| `POST /news/refresh/{ticker}` | Fetch and archive news; optional timezone-aware `time_from` and `time_to` |
| `POST /news/import` | Import normalized scored JSON articles |
| `GET /news/{ticker}?limit=20` | Read archive status and recent article sources/tone |
| `GET /predict/direction/live/{ticker}` | Analyze saved news and completed daily prices |
| `GET /predict/live/{ticker}?include_news=true` | Existing hybrid return response plus `direction_analysis` |

Provider errors or missing credentials yield a clear 503 for refresh. Cached
analysis still works without a key. There is no background news polling in this
version; schedule the refresh endpoint externally if desired. A response hitting
Alpha Vantage's 1000-article ceiling is rejected instead of silently treating an
incomplete historical window as complete; narrow the requested date interval.
Provider plan limits and historical coverage depend on the user's account.

## Import historical JSON

`POST /news/import` accepts this normalized shape. The example is illustrative,
not benchmark data:

```json
{
  "articles": [
    {
      "ticker": "AAPL",
      "title": "Example company earnings headline",
      "summary": "Example summary describing the earnings announcement.",
      "url": "https://example.com/earnings-story",
      "source": "Example archive",
      "published_at": "2025-01-06T18:00:00Z",
      "available_at": "2025-01-06T18:05:00Z",
      "sentiment": 0.6,
      "relevance": 0.9,
      "sentiment_model": "archive-model-version"
    }
  ]
}
```

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/news/import -ContentType application/json -InFile news.json
```

Sentiment must be between -1 and 1 and relevance between 0 and 1. Timestamp
values must include a timezone. Import up to 5000 articles per request; invalid
batches insert nothing. Existing stories remain immutable on repeated imports.

`available_at` means when this exact article text and sentiment were available,
not when an updated article claims it was originally published. Omit it to use
the import time. Alpha Vantage downloads always use retrieval time, including
historical queries. Do not substitute publication time for an unverified
historical availability timestamp: doing so invalidates point-in-time testing.
Imported timestamps are trusted archive assertions, not independently verified
by Hermes. Keep sentiment model versions and their historical availability too.

A first refresh will usually report `insufficient_point_in_time_history`. This
is expected: downloading old articles today does not create a valid historical
training archive. Likewise, news after the latest completed close appears in
the news list but is excluded from that close's forecast. Intraday/after-hours
forecasting needs a separately defined prediction cutoff and training target.

## Evaluation and recommended improvement sequence

The existing seven-ticker benchmark is now development data: it has already
influenced design decisions. A gain there is not fresh evidence of an edge.

1. Keep a new chronological test period and additional securities untouched.
   Use rolling training/selection windows on development history. Save paired
   per-date model and baseline decisions. Compare accuracy, down-day precision,
   balanced accuracy, and Brier score; use date-block resampling across the whole
   ticker basket when estimating uncertainty, because stocks move together.
2. Prioritize reliable detection of down days. Relative to always-up, a down
   override only helps when it is correct more often than it is wrong. Tune
   thresholds on selection data only. Calibrate probabilities on a separate
   chronological calibration window before presenting them as reliable odds.
3. Build a sufficiently long point-in-time article archive. Compare price-only,
   news-only, and combined models on identical dates. Measure coverage and source
   outages separately from genuine no-news periods. This version exposes coverage
   counts but cannot infer whether a historical provider feed was complete.
4. Add earnings surprises versus consensus, guidance changes, sector/index
   returns, and market volatility. Distinguish positive wording from a result
   beating expectations. For learning news-specific responses, also evaluate
   market/sector-relative returns; current labels use absolute stock returns.
5. Promote a new policy only after repeated held-out improvements. Keep the
   current return ensemble and always-up as baselines, and use realistic next-open
   execution and costs for trading evaluations. Classification accuracy alone
   does not establish trading profitability.

Run the development comparison against the saved price CSVs:

```powershell
.venv\Scripts\python.exe -m benchmarks.run_direction
.venv\Scripts\python.exe -m benchmarks.run_direction --news-json historical-news.json
```

The JSON file uses the import schema, with explicit `available_at` timestamps.
Outputs go to `benchmarks/results/direction_development.json` with input/source
hashes. This uses a 60/20/20 split and is not directly comparable to the earlier
expanding-window benchmark. Without an archive, it evaluates the price-only
classification path and cannot establish sentiment's contribution.

## Sources

- [Alpha Vantage news sentiment API](https://www.alphavantage.co/documentation/#news-sentiment): ticker filters, historical ranges, sentiment and result limits.
- [scikit-learn threshold tuning guidance](https://scikit-learn.org/1.5/modules/classification_threshold.html): separating fitting from decision-threshold selection.
- [Exchange calendars](https://github.com/gerrymanoim/exchange_calendars): actual exchange schedules.
- [FinBERT model card](https://huggingface.co/ProsusAI/finbert): possible future local financial-tone scorer, not a stock-return predictor.

# Frontend guide

Run the API and open http://127.0.0.1:8000/. No frontend build step is needed.
The API explicitly serves only index.html, app.js, and styles.css.

## Working features

- **Ticker analysis:** type a US equity/ETF ticker or use a quick-select button.
  Price history, return forecasts, direction validation, and news load
  independently. A failed optional source does not hide successful results.
- **Historical chart:** view two closes, five sessions, approximately one month,
  or three months of completed daily bars. Inspect values with the pointer or
  keyboard-accessible slider. The expandable table contains the same observations.
  Prices are provider-adjusted; they are not intraday quotes or currency conversions.
- **Return forecasts:** next-session price/return, the separate five-session
  return estimate, validation MAE in percentage points, sign accuracy, neural
  blend weight, and training counts all come from the API. There is no invented
  confidence or portfolio-risk score.
- **Direction validation:** the experimental classifier shows its selected
  policy, uncalibrated probability, held-out accuracy, always-up baseline,
  difference, evaluation dates, and news coverage. It is separate from the
  return ensemble and can disagree with it.
- **Saved tickers:** add, select, and remove up to 30 tickers using browser
  storage. Nothing is presented as an account portfolio. Storage failures
  explicitly limit persistence to the current visit.
- **News:** scan through the configured Alpha Vantage provider, view company
  sentiment and source links, or import scored JSON archives using the documented
  schema. Unconfigured providers show an empty state; historical availability
  rules still apply. The browser limits each import to 10 MB / 5000 records.
- **Backtests:** choose 1-20 tickers and 1, 2, or 5 years. Display per-ticker
  results, named statistical/neural/hybrid models, both simple baselines,
  separate five-session metrics, classifier Brier score, and individual source
  failures. Export the complete API results as JSON. Longer runs show elapsed
  time, not a fabricated progress percentage. Closing a browser request does
  not cancel backend training already in progress.
- **Trader disclosures:** refresh configured sources and display the actual
  recent disclosures, source names, trader counts, and recency-weighted
  buy/sell consensus. The UI does not translate these into buy recommendations.
  Records with absent, malformed, or future dates are omitted, not dated today.
- **Connection settings:** select the API server and check its actual health.
  The API indicator is not a market-open indicator or a data-provider guarantee.
- **Accounts:** an explicitly labeled preview explains that registration,
  sign-in, and cloud synchronization are not available. No fake identity,
  balance, or subscription is displayed.

## Accuracy and failure behavior

All financial values originate in API responses; there are no fallback sample
prices or production fixtures. Missing values remain unavailable, rather than
being formatted as zero. Dates are shown with the corresponding data. The chart
only draws observed historical closes, never an invented future price path.

Changing tickers clears the prior ticker's data and aborts pending panel
requests. Late responses cannot overwrite the current selection. Chart-range
requests use the same protection. Errors remain visible in their own panel,
while successful panels remain usable. News text, source errors, and ticker
labels are inserted as text; article links accept only HTTP(S).

Fetching a current quote or running a model requires network access to the
market-data provider. Alpha Vantage and trader feeds require their server-side
configuration. A missing provider is an unavailable feature, not a reason to
show sample data. Models remain research estimates without a demonstrated
consistent advantage over simple baselines.

## Validation

`tests/test_frontend.py` exercises real browser interactions with API fixtures
isolated to the tests: metrics, saved ticker persistence, changing tickers,
chart ranges, independent panel failures, stale response suppression, safe
article rendering, scanning, JSON import, backtests, export, mobile overflow,
connection settings, and the account preview. Screenshots go to the ignored
`tests/browser_artifacts/` directory. Backend tests cover static serving,
optional news failures, completed exchange closes, and disclosure validation.

The redesigned UI was also smoke-tested against the running API with actual
Yahoo Finance history. No stored benchmark CSV is used as application data.

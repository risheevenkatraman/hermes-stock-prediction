# Hermes

Hermes is a stock research dashboard combining daily market data, statistical
and neural forecasts, company-news sentiment, and historical model evaluation.

## Current state

**Working development application; deployment configuration prepared.** EC2 and
GitHub Pages setup is documented, but a live production deployment has not been
verified. The latest local test run passed **89 tests**, including browser
interactions. The Linux container build still needs verification in GitHub Actions.

The dashboard uses real API responses and explicit unavailable/error states.
Accounts are the only mockup. Forecasts remain experimental research estimates;
the model has not demonstrated a consistent advantage over simple baselines.

## Current features

| Area | Implemented behavior |
| --- | --- |
| Market data | Ticker selection and adjusted daily price charts using completed trading sessions |
| Forecasts | Statistical, neural, and hybrid return estimates; next-session and five-session horizons |
| Direction analysis | Separate experimental classifier with held-out results and an always-up comparison |
| Company news | Alpha Vantage sentiment ingestion, historical JSON import, and a persistent article archive |
| Backtesting | Multiple tickers and periods, per-model metrics, baseline comparisons, and JSON export |
| Trader disclosures | Configurable provider adapters, dated disclosures, and recency-weighted consensus |
| Dashboard | Responsive layout, independent panel errors, saved tickers, and API connection settings |
| Deployment | Pages publishing workflow, EC2 containers, HTTPS proxy, and backend validation workflow |

News scans and trader feeds require configured providers. Saved tickers remain
in the current browser. In deployment mode, news writes, provider refreshes, and
backtests require administrator access; public predictions and stored news remain
readable. User registration and cross-device synchronization are not implemented.

## Model performance

The saved seven-symbol benchmark evaluated **1,568 next-day predictions**:

| Direction prediction | Accuracy |
| --- | --- |
| Hybrid model | 50.32% |
| Always-up baseline | 52.10% |

The hybrid did not beat always-up in this evaluation. These are historical
benchmark results, not current live accuracy. The separate news classifier has
not yet been evaluated with a verified historical news archive. Full results and
evaluation limits are in the [benchmark report](benchmarks/results/accuracy.md).

## Development log

- **Model foundation:** shared price features, statistical and neural regressors,
  chronological blend selection, and walk-forward evaluation.
- **Validation corrections:** reduced leakage, separated selection from evaluation,
  and corrected five-day backtest return accounting.
- **News and direction:** added article availability timestamps, sentiment/text
  features, and a separate direction classifier with price-only fallback.
- **Functional dashboard:** replaced demonstration content with live API results,
  working controls, source status, and browser interaction coverage.
- **Deployment preparation:** added EC2/Pages configuration, administrator access
  controls, persistent storage, and deployment validation checks.

## To do

### Before the first hosted release

- [ ] Verify the Linux container build and backend workflow in GitHub Actions.
- [ ] Deploy EC2 and GitHub Pages; verify HTTPS, CORS, and cloud data-provider access.
- [ ] Configure required providers and verify archive persistence and backup recovery.
- [ ] Establish monitoring, error reporting, and an operating-cost baseline.

### Prediction quality and data

- [ ] Collect verified historical news and measure its incremental predictive value.
- [ ] Improve direction accuracy against always-up across multiple market regimes.
- [ ] Calibrate probabilities and maintain an untouched evaluation set.
- [ ] Add model versioning, experiment tracking, and persisted trained models.
- [ ] Add market-data caching and provider rate-limit handling.
- [ ] Confirm data licensing and reliability requirements for the intended release.

### Product and operations

- [ ] Implement real accounts, permissions, and synchronized watchlists/settings.
- [ ] Add per-client request limits and background jobs for expensive evaluations.
- [ ] Lock release dependencies and retain versioned deployment images.
- [ ] Automate backups and expand deployment monitoring before broader traffic.

Personal setup instructions and development notes are maintained locally and
are not included in this repository.

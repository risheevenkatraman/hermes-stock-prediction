# Hermes

A Next.js/React stock research workspace with a cream, forest-green, and soft
orange visual identity. The Python research foundation is retained while the
customer application is rebuilt for AWS.

## Implemented

- Next.js App Router dashboard, responsive layout, accessible dialogs and chart
  data table, ticker search, watchlists, and selectable chart ranges.
- Profiles: USD budget, 200-character goal, investment horizon, and risk preference.
- Cognito OAuth integration and backend access-token verification.
- User-scoped PostgreSQL-compatible workspace persistence.
- Cached research API with stale/error states and an offline publication worker.
- Short-term forecast controls and a separate long-term research explanation.
- Cognito/S3 CloudFormation foundation and Amplify build configuration.

Removed the static HTML/JavaScript frontend, mock accounts, GitHub Pages builder,
and obsolete browser tests. Retained model, news, backtest, and trader modules as
research tools. Expensive legacy endpoints require admin access in production and
are not used by the customer dashboard.

## Limits and next milestones

This is a working application foundation, not a customer-ready investment adviser.
No AWS deployment or real OAuth round-trip has been verified. Deployment, local
setup, and operating instructions live in the local-only
`docs/` directory, which is excluded from Git.

- Actual models publish 1- and 5-trading-day forecasts. Horizons 2-4 remain
  unavailable in research mode; synthetic preview has all five.
- Calibrated intervals are not implemented; the old heuristic confidence is not
  displayed. The saved hybrid benchmark measured **50.32%** direction accuracy
  versus **52.10%** always-up across 1,568 observations, with no baseline advantage.
  See the [benchmark report](benchmarks/results/accuracy.md).
- Discovery cards are static research starting points, not personalized purchase
  recommendations. Saved budgets/goals do not yet drive a validated ranker.
- News is separate context; the three-stream model and verified trader track
  records remain research work.
- Long-term fundamentals, licensed production data, model artifact persistence,
  migrations, rate limits, and production operations remain milestones.
- Research currently assumes US exchanges and USD.

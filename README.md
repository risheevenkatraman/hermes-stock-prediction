# Hermes

Hermes is a US-stock research application for people learning to invest and
experienced investors exploring additional insights. It combines a personalized
workspace with market research and aims to explain why a stock may rise or fall.
Its interface uses a cream, dark-green, and soft-orange palette inspired by Hermes.

Hermes is currently in development. Forecasts are experimental: evaluations have
not established a consistent advantage over simple baselines. The application
does not execute trades, connect bank accounts, or provide validated personalized
purchase recommendations.

## Planned product features

- Individual accounts with email and Google sign-in.
- A saved budget, financial goal of up to 200 characters, investment horizon,
  and risk preference.
- Searchable stocks, personal watchlists, current market data with clear freshness
  information, and interactive price-history charts.
- Short-term forecasts covering one to five trading days.
- Six- and twelve-month positive, negative-or-flat, or uncertain outlooks without
  implying an exact future price.
- Recommendations matched to a user's goals and budget, supported by validated
  predictions and concise educational explanations of up to five sentences.
- A combined research model using price and market/sector trends, company news
  and events, and verifiable publicly disclosed trader activity.
- US stocks first, with international-market support planned for a later release.
- Bank connections and trade execution as later milestones after research quality
  and operational readiness are established.

## Implementation checklist

Checked items describe implemented code, not a verified production rollout.

- [x] Responsive Next.js dashboard with Hermes branding, ticker search, watchlists,
      chart ranges, and accessible chart data.
- [x] Saved USD budget, financial goal, horizon, and risk preferences.
- [x] Cognito sign-in integration, Google OAuth support, token verification,
      and user-scoped workspace storage.
- [x] Cached research API, stale/error states, and an offline publication worker
      that preserves forecast history.
- [x] Experimental one- and five-trading-day price forecasts.
- [x] Price/sector datasets, chronological benchmarks, baseline comparisons,
      and archived research evidence.
- [x] Timestamped news collection, quality flags, article-version tracking,
      snapshot verification, and coverage reporting.
- [x] Short-term news-only, price/sector-only, and combined model code with a
      guarded evaluation runner; real news training awaits sufficient data.
- [x] Separate six- and twelve-month direction models, evaluated on historical
      development data; neither horizon beat the simple baselines overall.
- [x] Dashboard support for long-term outlooks, with an unavailable state when
      no result has been published.
- [x] AWS deployment configuration for the web application, API, authentication,
      database, and research archives.
- [ ] Validated forecasts across every one-to-five-day horizon and calibrated
      uncertainty estimates.
- [ ] Validated news-enhanced and long-term models suitable for customer use.
- [ ] Verified public-trader data and a tested three-stream model.
- [ ] Goal/budget-based stock ranking and evidence-linked educational explanations.
- [ ] International stocks, bank connections, and trade execution.

Discovery cards are currently research starting points, not personalized rankings.
The new research models do not automatically replace published forecasts or drive
purchase recommendations.

## Implementation stack

| Area           | Technology                                                              |
| -------------- | ----------------------------------------------------------------------- |
| Frontend       | Next.js App Router, React, TypeScript, CSS                              |
| Backend        | Python, FastAPI, Pydantic                                               |
| Modeling       | scikit-learn, PyTorch, pandas, NumPy, exchange-calendars                |
| Research data  | Yahoo Finance/yfinance for local price research; Alpha Vantage for news |
| Authentication | Amazon Cognito, Google OAuth, JWT verification                          |
| Storage        | SQLAlchemy, PostgreSQL/Amazon RDS, local SQLite, Amazon S3 archives     |
| Hosting        | AWS Amplify frontend; Docker/FastAPI and Caddy on Amazon EC2            |
| Infrastructure | AWS CloudFormation foundation, Docker Compose                           |
| Validation     | pytest, Ruff, TypeScript checks, Playwright, Prettier                   |

## Next steps

1. Collect prospective news consistently and finish the remaining evaluation
   diagnostics before training on real news data.
2. Investigate point-in-time company fundamentals for a separately evaluated
   long-term model improvement.
3. Build verified public-disclosure data, personalized ranking, and educational
   explanations as their supporting evidence becomes available.
4. Complete production data permissions, operational checks, and model validation
   before a customer launch.

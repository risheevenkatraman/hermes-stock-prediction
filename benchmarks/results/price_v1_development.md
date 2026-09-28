# Price model development evaluation

Run: `20260928T022935830776Z-310c4ade`

Historical development results, not an untouched holdout or live trading result.

Returns are decimal fractions in JSON; the table reports MAE in percentage points.

| Horizon | Model MAE (pp) | Zero-return MAE (pp) | Model direction | Always up | Observations |
| ------- | -------------: | -------------------: | --------------: | --------: | -----------: |
| 1       |         1.5912 |               1.4214 |          49.94% |    51.95% |         1540 |
| 2       |         2.2983 |               2.0396 |          52.08% |    52.34% |         1540 |
| 3       |         2.8748 |               2.4924 |          51.49% |    52.14% |         1540 |
| 4       |         3.3697 |               2.9181 |          52.66% |    52.47% |         1540 |
| 5       |         3.8047 |               3.2620 |          49.68% |    52.01% |         1540 |

## Interpretation

- Each horizon uses the same forecast origins; all target endpoints stay inside development.
- Training includes a label only when its endpoint is known at the forecast origin.
- Always-up is direction-only; its return MAE/RMSE are null, not misleading +/-1 return scores.
- Zero return, historical mean, and previous-horizon return are separate baselines.
- Pooled rows share dates/market exposure; no independence or statistical significance is claimed.
- Adjusted historical prices are provider snapshots, not verified point-in-time histories.
- No trading performance is reported: close-to-close forecasts are not executable at that same close.
- No intervals or calibrated probabilities are claimed. Do not deploy a model on this report alone.
- Candidate and baselines must be compared per ticker and horizon as well as in aggregate.

The future holdout is reserved in protocol.json and has not been evaluated by this command.

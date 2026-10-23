# Architecture

This page describes the whole STOCKA system in layers, then the excerpt in this repository in detail: its module map, the data flow of the demo, how the provider protocol feeds the engine, and the design principles that are visible in the code.

## 1. The whole system

STOCKA is a research and risk-discipline tool for one private investor. It explains and checks; it does not trade. Its main cycle is a weekly research run. On-demand and event-driven reports come on top.

For each layer the Status column says what is **in this excerpt** (code in this repository) and what stays **private**. Descriptions of private parts are summaries and cannot be checked against this repository.

| Layer | What it does | Status |
|---|---|---|
| Data ingestion | Pulls fundamentals and end-of-day prices from a commercial market-data API (EODHD), plus regulatory filings and public macro series. | **Private.** Replaced here by the provider protocol and two simple providers in `stocka/providers/`. |
| Storage | Keeps fundamentals with point-in-time versions, universe snapshots and the portfolio state in local stores. | **Private.** The state persistence layer is not part of this excerpt (`brain/state/__init__.py`). |
| Scoring | Uses the data-quality (DQ) score as a hard-veto gate before a downstream score. | **Partly in this excerpt:** `brain/scoring/`. The downstream score, screening stage and orchestrator are private (`dq_score.py` and `factor_model.py` refer to them as outside this excerpt). |
| Risk | Measures drawdown against a high-water mark and checks stages, sizing limits, stops and risk analytics. | **Partly in this excerpt:** `brain/risk/`, `brain/state/`. Equity measurement, a sector cap, stress tests and persistence are private. |
| Forecasting and calibration | Registers screening picks as forecasts and scores the resolved ones for calibration. | **Partly in this excerpt:** `brain/forecast/calibration.py`, `brain/forecast/episode_freeze.py`. The forecast log and its reports are private. |
| Research retrieval (RAG) | Searches a corpus of regulatory filings with combined full-text and vector retrieval. | **Private.** |
| Reporting | Renders weekly, event-driven and on-demand reports, with a mandatory external bear case where a judgement is supported. | **Private.** `brain/core/external_bear_guard.py` is the shared bear-case check. The demo report in `stocka/demo.py` is a public stand-in. |
| Operations | Runs the weekly cycle and data refreshes on a schedule. | **Private.** This repository has its own GitHub Actions workflow (`.github/workflows/ci.yml`), which runs the tests and the demo. |

## 2. The excerpt

### 2.1 Engine: `brain/`

| Module | Main entry points | What it does |
|---|---|---|
| `scoring/beneish.py` | `compute_beneish_m_score` | Eight-component Beneish M-score from two fiscal years. Red flag above -1.78, gray zone from -2.22 to -1.78. Returns `complete: False` and the missing inputs instead of a partial score. |
| `scoring/sloan.py` | `compute_sloan_accruals`, `compute_sloan_quintile` | Balance-sheet accruals scaled by average total assets; quintile of one ticker against a universe sample (needs at least 10 values). |
| `scoring/sloan_universe.py` | `compute_sloan_universe_snapshot` | Accruals ratios for all non-financial tickers of a universe. |
| `scoring/dq_score.py` | `compute_dq_score`, `get_dq_vetos` | Composite of seven components (report age, auditor tier, restatements, revenue-recognition restatement, Sloan quintile, Beneish M, going concern), bounded 0 to 100, with hard vetoes, caps and flags. Sector bypass for Financials and Utilities. |
| `scoring/factor_model.py` | `compute_idio_share_inmemory`, `classify_idio_share_quintile` | PCA on an exponentially weighted covariance; keeps factors up to 90 % explained variance (at most 8); returns each ticker's idiosyncratic share of variance. Audit only. |
| `risk/drawdown.py` | `max_drawdown_from_closes` | Level-based maximum and current drawdown, gap count against a computed NYSE holiday calendar, and a list of single-day moves above 50 %. |
| `risk/es_returns.py` | `bounded_returns` | Daily returns that never bridge a bad bar or a gap of two or more missing business days. |
| `risk/es_historical.py` | `compute_es_historical` | Historical-simulation VaR and expected shortfall at 97.5 % and 99 % over up to 756 days, with a low-confidence flag below 756 observations. |
| `risk/_tail_k.py` | `tail_k` | Shared rule for how many worst observations form the tail mean. |
| `risk/risk_metrics.py` | `compute_cvar_bundle` | Naive and EWMA volatility-adjusted VaR/CVaR at 95 %, 97.5 % and 99 %. |
| `risk/risk_profile.py` | `compute_risk_profile` | Annualised volatility, downside deviation and the longest underwater stretch. No reward measure by design. |
| `risk/single_asset_metrics.py` | `single_asset_metrics` | Risk measures per asset; Sharpe, Sortino and Calmar only with `include_reward=True`. |
| `risk/bootstrap_sim.py` | `bootstrap_dd_sim`, `check_p95_pass` | Stationary bootstrap of daily returns (arch, with a numpy fallback); P95/P99 maximum drawdown and conditional drawdown beyond P95. No production caller in the private system; here the demo calls it. |
| `risk/correlation.py` | `pairwise_correlation_matrix`, `find_correlation_clusters`, `pearson_rolling_correlation` | Pearson correlation over a 60-day window; pairs with \|rho\| above 0.75. Not a cluster alarm. Here the demo calls the matrix and the pair search. |
| `risk/stops.py` | `calc_atr`, `compute_hybrid_stop`, `compute_hybrid_stop_from_ohlc`, `update_trailing_soft_stop` | ATR-based hard and soft stops; the soft-stop multiplier depends on sector, industry and market cap; trailing soft stop. |
| `risk/position_sizing.py` | `check_sizing`, `book_is_empty` | Three sizing limits for a new position; every rule returns PASS, WARN, FAIL or N/A with a reason. |
| `backtest/metrics.py` | `compute_sharpe`, `compute_sortino`, `compute_max_drawdown`, and others | Series metrics. Degenerate input never raises: Sharpe and Sortino return `None`, max drawdown returns `(0.0, None, None)`, win rate, hit rate and period return return 0.0. Only an unknown frequency raises. |
| `backtest/result.py` | `BacktestResult` | Result record with a point-in-time caveat as a default field. |
| `state/portfolio_state.py` | `PortfolioState` | Positions, high-water-mark anchor, drawdown stage, cooling-off window, cash, flow ledger and an append-only audit log; JSON round trip. |
| `state/dd_state_machine.py` | `get_dd_tier_raw`, `apply_tier_transition`, `ratchet_peak_to_measured_equity` | Drawdown stages at -22 %, -32 % and -42 %, a 3-point cushion before stepping down, and a 168-hour cooling-off after entering the -32 % zone. Every change goes through the audit log. |
| `state/dd_multiplier.py` | `get_dd_multiplier` | Maps stages 0 to 3 to sizing multipliers 1.0, 0.5, 0.0, 0.0. Dormant: not wired into any sizing path, neither here nor in the private system. |
| `forecast/calibration.py` | `brier_score`, `ece` | Calibration metrics with input validation. |
| `forecast/episode_freeze.py` | `lag_gate_rows`, `window_elapsed` | Shows a resolved forecast as open until its original window has elapsed, so an early resolution cannot leak into reported numbers. |
| `analytics/concentration.py` | `concentration_metrics` | HHI, effective number of names and top-n weight, always with input, priced and missing counts. |
| `core/external_bear_guard.py` | `require_external_bear_case` | Raises unless a bear case has an external source type, a URL and a source date. |

### 2.2 Public frame: `stocka/`

| Module | What it does |
|---|---|
| `providers/base.py` | `Ticker` and the `DataProvider` protocol (four methods, see section 4). |
| `providers/synthetic.py` | `SyntheticProvider`: seeded prices from a market, sector and noise model with one stress window and one single-name crash, on business days (weekdays, no holiday calendar); fundamentals in the EODHD shape; the first twenty tickers follow fixed profiles, some of them built to trip the engine's guards. |
| `providers/local_files.py` | `LocalFilesProvider` reads `prices.csv` and `fundamentals/<TICKER>.json`; `write_local_files` writes any provider in that layout. |
| `config.py` | Frozen dataclasses for demo settings and screen thresholds, loaded from TOML; wrong types, out-of-range values and unknown keys raise `ConfigError`. |
| `screen.py` | `screen_ticker`: engine vetoes, a minimum DQ score, a stricter Beneish limit, optional Sloan Q5 exclusion and optional exclusion of unverified audit inputs, each with a stated reason. |
| `demo.py` | Aligns the price series on common dates, builds the daily rebalanced equal-weight basket series and its sector weights, draws the synthetic calibration events, calls the engine and renders the Markdown report. |
| `cli.py` | `stocka demo` and `stocka --version`; configuration and input errors exit with code 2. |

## 3. Data flow of the demo

```text
stocka demo [--seed --tickers --config --data-dir --out]
  |
  |  load_config + overrides                                   stocka/config.py
  v
DataProvider  (SyntheticProvider or LocalFilesProvider)        stocka/providers/
  |  universe(), prices(), fundamentals(), audit_signals()
  v
collect()                                                      stocka/demo.py
  |- Sloan samples   compute_sloan_universe_snapshot           (only used with >= 10 values)
  |- per ticker      compute_beneish_m_score
  |                  compute_sloan_accruals
  |                  compute_dq_score   -> DQContractError = ticker rejected
  |                  compute_risk_profile, max_drawdown_from_closes
  |- screen          screen_ticker (illustrative thresholds)   stocka/screen.py
  |- passing basket  equal weight, daily rebalanced levels
  |                  compute_risk_profile, max_drawdown_from_closes
  |                  bounded_returns -> compute_es_historical  (fixed 97.5 %, needs >= 20 returns)
  |                  bootstrap_dd_sim                          (seeded, optional)
  |                  pairwise_correlation_matrix -> find_correlation_clusters (correlated pairs)
  |                  concentration_metrics (sector weights, market caps)
  |- calibration     brier_score, ece on synthetic forecasts
  v
render_report() -> Markdown on stdout or in --out
```

Steps in order:

1. The CLI loads the config, applies `--seed` and `--tickers`, and picks the synthetic provider or, with `--data-dir`, the local-file provider.
2. `collect()` reads the universe, all prices and all fundamentals. The scoring date is the latest price date.
3. Sloan accruals for the universe are passed to the DQ score only if at least 10 values exist. With fewer, the DQ score adds a soft flag instead of vetoing every ticker.
4. Each ticker is scored. If the audit signals break the contract, `compute_dq_score` raises `DQContractError` and the ticker is reported as rejected, not scored.
5. The illustrative screen turns engine results and config thresholds into pass or fail with reasons.
6. The passing tickers form an equal-weight basket. Prices are aligned on common dates, and the basket's risk measures come from the engine functions shown above.
7. A separate calibration section scores three synthetic forecasters with known probabilities.
8. The report is rendered. It contains data-quality checks, risk measures and the calibration example, and no performance figures.

## 4. How the provider protocol feeds the engine

`DataProvider` in `stocka/providers/base.py` has four methods:

| Method | Returns | Used by |
|---|---|---|
| `universe()` | list of `Ticker(symbol, sector)` in a stable order | the demo loop, the screen table, sector concentration |
| `prices(symbol)` | daily closes as `[(ISO date, close)]`, oldest first | `risk_profile`, `drawdown`, the basket, `es_returns`, `bootstrap_sim`, `correlation` |
| `fundamentals(symbol)` | dict in the shape of the EODHD fundamentals response: `General`, `Highlights`, `Financials` with yearly income statement, balance sheet and cash flow; at least two fiscal years; numbers may be strings | `beneish`, `sloan`, `sloan_universe`, `dq_score` (report age, sector bypass), market-cap concentration |
| `audit_signals(symbol)` | dict with required `going_concern` and `restatements_36m`, optional `auditor_tier`, `revenue_recognition_restatement`, `filing_date`; or `None` | `dq_score` |

The protocol is a feeding point, not a full data abstraction. A source with a different schema needs an adapter that produces the EODHD shape. Missing audit signals are allowed: the engine then rejects the ticker with a contract error instead of scoring it.

## 5. Design principles visible in the code

**Pure functions.** Many engine modules say in their docstring that they are pure functions without I/O or provider imports (for example `stops.py`, `drawdown.py`, `es_returns.py`, `correlation.py`, `dq_score.py`, `calibration.py`, `position_sizing.py`). The callers pass all inputs in. `test_position_sizing.py` checks that the sizing module contains no file or network calls. The exceptions are `PortfolioState` and the state machine that mutates it, and the SQLite read helpers in `concentration.py`, which use a connection supplied by the caller.

**Fail-closed inputs.** Missing or invalid input never turns into a plausible number:

- `compute_dq_score` requires the audit signals; there is no soft mode. `None` is an explicit "unverified" state that is flagged, never scored as clean.
- `max_drawdown_from_closes` returns `None` figures plus an error text, never a silent 0. `compute_risk_profile` does the same for volatility and downside deviation (its underwater bar counts are 0 on an empty or unusable series, as the module documents).
- `compute_es_historical` refuses NaN input; `bounded_returns` breaks the return chain at a bad bar.
- `get_dd_tier_raw` rejects NaN and positive drawdowns, so a broken equity curve cannot map to stage 0.
- `check_sizing` returns FAIL, with the name of the bad input, for every rule that a missing or invalid input affects.
- `compute_hybrid_stop` rejects stops at or below zero and enforces soft < hard < entry.

**Explicit caveats.** Results carry their own limits:

- `BacktestResult` has the point-in-time caveat as a default field.
- `concentration_metrics` always reports how many inputs were priced and how many were missing.
- `compute_es_historical` and `compute_cvar_bundle` set a low-confidence flag on short histories.
- `compute_dq_score` flags an unresolvable filing date and an unknown auditor tier, because both would otherwise look like measured bad values.

**Known deviations are documented, not silently fixed.** Where the code departs from a textbook method or has a known weakness, a comment says so and explains why the behaviour was kept until a deliberate decision. Examples: the simple-mean ATR instead of Wilder's smoothing (`stops.py`), the floor+1 tail size (`_tail_k.py`), the procyclical 756-day expected-shortfall window (`es_historical.py`), the short bootstrap block length (`bootstrap_sim.py`), flat series and negative correlations (`correlation.py`), the restatement grading that the downstream veto makes unreachable (`dq_score.py`), and three maximum-drawdown conventions with different signs and scales: negative percent, negative fraction, positive fraction (`drawdown.py`).

**Explain, do not act.** `dd_state_machine.py` documents its stages as stages of review depth, not trading actions. `dd_multiplier.py` would map stages 2 and 3 to a sizing multiplier of 0.0, but it is dormant and wired into no sizing path, so the stages never stop new purchases. The trade-block half of `es_historical.py` is deliberately not activated; its triggers would be prompts to the investor, not orders. `bootstrap_sim.py` has no production caller in the private system. Reward metrics in `single_asset_metrics.py` are off by default. `brain/analytics` describes the data and feeds no score.

**Engine rules versus frame settings.** The thresholds of the illustrative screen are configuration (`stocka/config.py`, `config/example.toml`). The engine's own rules (vetoes, stop multipliers, sizing limits, drawdown stages) are constants in code and cannot be configured.

## 6. What the excerpt does not contain

- No data ingestion, database, persistence layer or scheduling. The SQLite helpers in `concentration.py` only query a connection that the caller provides.
- No downstream score, candidate list or recommendation logic.
- No backtest harness and no forward test.
- No order, broker or execution code, and no network calls.
- No research retrieval and no language-model calls.
- No report rendering beyond the demo and two small text formatters (`format_concentration_lines` in `concentration.py`, `to_trade_frontmatter` in `portfolio_state.py`).

## 7. Private repository

**Not verifiable from this repository:** development since May 2026, about 2,700 commits, and 583 test files in the private suite.

# stocka-core

STOCKA is a private research tool for one private investor. It supports fundamental research and risk discipline: it checks the data quality of company fundamentals, measures risk, tracks drawdown stages and scores its own forecasts. This repository is an excerpt of it: a set of engine modules plus a small public frame that runs them on synthetic data. It is a research tool, not investment advice and not an auto-trader. The excerpt contains no order or broker code and makes no network calls.

## What is in this excerpt

Engine (`brain/`): 25 modules in 7 packages, copied from the private repository, plus tests (from the private suite and newly written, see [Tests](#tests)). The package keeps its private-system name `brain`, so the files match the originals.

| Package | Modules | What it does |
|---|---|---|
| `brain/scoring` | `beneish`, `sloan`, `sloan_universe`, `dq_score`, `factor_model` | Beneish M-score, Sloan balance-sheet accruals and quintiles, a 0-100 data-quality (DQ) score with hard vetoes, idiosyncratic return share from a PCA factor model |
| `brain/risk` | `drawdown`, `es_returns`, `es_historical`, `_tail_k`, `risk_metrics`, `risk_profile`, `single_asset_metrics`, `bootstrap_sim`, `correlation`, `stops`, `position_sizing` | Level-based drawdown, gap-bounded daily returns, historical expected shortfall, VaR/CVaR, volatility and downside deviation, bootstrap drawdown simulation, rolling correlation, ATR stops, position-size limits |
| `brain/backtest` | `metrics`, `result` | Sharpe, Sortino and max-drawdown helpers; a result record that carries a point-in-time caveat by default |
| `brain/state` | `portfolio_state`, `dd_state_machine`, `dd_multiplier` | Portfolio state with an append-only audit log; drawdown stages with a re-entry cushion and a cooling-off window; a dormant stage-to-sizing multiplier |
| `brain/forecast` | `calibration`, `episode_freeze` | Brier score and expected calibration error (ECE); a lag gate that counts a forecast as open until its window has elapsed |
| `brain/analytics` | `concentration` | Herfindahl index, effective number of names and top-n weight, with coverage counts |
| `brain/core` | `external_bear_guard` | Rejects a bear case that is not externally sourced |
| `brain/tests` | 22 test files | Tests for the modules above |

Public frame, written for this excerpt:

| Path | What it does |
|---|---|
| `stocka/providers/` | `DataProvider` protocol, a deterministic synthetic provider and a local-file provider |
| `stocka/config.py` | Demo settings and thresholds of the example screen, loaded from TOML and validated |
| `stocka/screen.py` | Illustrative pass/fail screen on top of the engine results |
| `stocka/demo.py`, `stocka/cli.py` | The `stocka demo` command: runs the engine and prints a Markdown report |
| `tests/` | Tests for the frame |
| `config/example.toml`, `pyproject.toml`, `Makefile`, `.github/workflows/ci.yml` | Example config, packaging, shortcuts, CI |

## Architecture

The private system has layers for data ingestion, storage, scoring, risk, forecasting and calibration, research retrieval, reporting and operations. This excerpt contains parts of the scoring, risk and forecasting layers, the portfolio-state logic, and a few helpers. Most engine functions do no I/O and work on plain Python data. The exceptions are the mutable portfolio state, the drawdown state machine that updates it, and two SQLite read helpers in `concentration` that take a connection from the caller. The frame feeds the engine through a small `DataProvider` protocol, so the same engine code runs on synthetic data or on local files. Details, a module map and the demo data flow are in [docs/architecture.md](docs/architecture.md).

## Quickstart

Requires Python 3.12 or newer.

```bash
pip install -e ".[dev]"     # installs numpy, arch (which pulls in pandas, scipy, statsmodels) and pytest
make demo                   # same as: python -m stocka demo
make test                   # pytest
make test-unittest          # standard-library unittest runner
```

After installation the command is also available as `stocka demo`. Options: `--seed`, `--tickers` (1 to 99), `--config config/example.toml`, `--data-dir DIR`, `--out FILE`.

Shortened output of `python -m stocka demo --seed 1` (about 1.6 s CPU time on one machine):

```text
# stocka-core demo report: synthetic data, not market data

Research output, not investment advice. The report shows data-quality checks and risk measures only; it deliberately contains no performance figures.

- Data: 20 tickers, 780 business days (weekdays, no holiday calendar) from 2022-07-05 to 2025-06-30, seed 1
- Scores as of: 2025-06-30
- Screen settings (illustrative): DQ score >= 70, Beneish M <= -2.22, Sloan Q5 excluded: yes, unverified audit inputs excluded: yes

## 1. Data-quality screen

| Ticker | Sector | DQ | Beneish M | Sloan ratio | Sloan Q | Vol p.a. | Max DD | Screen | Why |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| SYN01 | Information Technology | 85 | -2.64 | -0.038 | 2 | 26 % | -39 % | pass |  |
| SYN03 | Industrials | 25 | -1.13 | 0.077 | 5 | 26 % | -54 % | fail | engine veto: sloan_q5_anomaly; engine veto: beneish_red_flag; DQ score 25 < 70 |
| SYN09 | Financials | 55 | n/a | -0.042 | n/a | 20 % | -26 % | fail | DQ score 55 < 70; note: Beneish not applied (sector bypass); note: Sloan not applied (sector bypass) |
| SYN10 | Materials | n/a | -2.62 | -0.033 | n/a | 31 % | -35 % | fail | audit input rejected: missing restatements_36m |
...

11 of 20 tickers pass the screen.
...

## 2. Basket risk (equal weight, 11 passing tickers)

Daily rebalanced equal weights over 780 common price dates (2022-07-05 to 2025-06-30). Risk measures only.

| Measure | Value | Engine function |
|---|---:|---|
| Annualised volatility | 15.7 % | `risk_profile.compute_risk_profile` |
| Maximum drawdown | -21.5 % (2023-05-02 to 2023-11-08) | `drawdown.max_drawdown_from_closes` |
| Expected shortfall 97.5 %, 1 day | 2.17 % | `es_historical.compute_es_historical` |
| Bootstrap max drawdown, P95 / P99 | -45.2 % / -52.2 % | `bootstrap_sim.bootstrap_dd_sim` |
...

## 3. Forecast calibration (synthetic forecasts)
...

| Forecaster | Brier score | ECE (10 bins) |
|---|---:|---:|
| states the true probability | 0.170 | 0.030 |
| overconfident (distance from 0.5 stretched x1.6) | 0.186 | 0.125 |
| underconfident (distance from 0.5 shrunk x0.4) | 0.199 | 0.145 |
```

Measured with numpy 2.5.2 and arch 8.0.0; the bootstrap row depends on arch's random stream and may differ under another arch build.

The synthetic universe contains tickers built to trip the engine's guards (a Beneish red-zone case, high accruals, a bank, missing or unverified audit data, a restatement, a going-concern opinion). The bootstrap P95 drawdown (about -45 %) is far deeper than the realised basket drawdown (-21.5 %), mainly because it is the 95th percentile over 2,000 resampled paths, compared with one realised path. The report also lists correlated pairs and concentration figures, and it never prints returns, Sharpe, Sortino, win rates or hit rates.

To run the demo on your own data, write prices and fundamentals in the layout described in `stocka/providers/local_files.py`. The quickest way to see a complete example is to export the synthetic data:

```bash
python -c "from stocka.providers import SyntheticProvider, write_local_files; write_local_files(SyntheticProvider(1, 20), 'data')"
python -m stocka demo --data-dir data
```

## Using the engine

The engine functions take plain Python data. Example: level-based drawdown on a list of `(ISO date, close)` pairs.

```python
from brain.risk.drawdown import max_drawdown_from_closes

closes = [
    ("2025-06-02", 100.0),
    ("2025-06-03", 104.0),
    ("2025-06-04", 97.0),
    ("2025-06-05", 91.0),
    ("2025-06-06", 95.0),
    ("2025-06-09", 102.0),
]
result = max_drawdown_from_closes(closes)
print(result["max_dd_pct"], result["peak_date"], result["trough_date"])
# -12.5 2025-06-03 2025-06-05
print(result["current_dd_pct"], result["gap_count"])
# -1.9231 0

bad = max_drawdown_from_closes([("2025-06-02", 100.0), ("2025-06-03", 0.0)])
print(bad["error"], bad["max_dd_pct"])
# nicht berechenbar (Datenfehler) None
```

Invalid input does not produce a number: the result carries an `error` text ("not computable (data error)", in German) and the figures are `None`.

## Tests

The excerpt has 374 tests in 26 files, 22 in `brain/tests/` and 4 in `tests/`:

| Origin | Tests |
|---|---:|
| Private test suite, 14 files (code unchanged apart from the differences listed under "Why a single commit") | 162 |
| `brain/tests/test_external_bear_guard.py`, rewritten as a `unittest.TestCase` with the scenarios of the private original | 5 |
| New behaviour tests for `brain/scoring/dq_score.py` on synthetic data, replacing two private tests that depended on licensed vendor data | 48 |
| New behaviour tests for six engine modules that had no exportable tests (`test_tail_k.py`, `test_dd_multiplier.py`, `test_correlation.py`, `test_risk_metrics.py`, `test_sloan_universe.py`, `test_factor_model.py`) | 85 |
| Public frame (`tests/`) | 74 |

Every engine module is imported directly by at least one test file. Tests that construct a scenario (correlated or uncorrelated data, boundary hits, sector bypass) assert that premise explicitly before checking the outcome.

How to run them: `make test` runs pytest over `brain/tests` and `tests`. `make test-unittest` runs the same tests with the standard-library unittest runner, for environments without pytest.

Runs before publication:

| Interpreter | Runner | Result |
|---|---|---|
| Python 3.14.9, numpy 2.5.2, arch 8.0.0 | pytest 8.3.4 | 374 passed |
| Python 3.14.9 | `make test-unittest` | 374 passed (300 + 74) |
| Python 3.12.15 without numpy | pytest 8.3.4, skipping the four test files that need numpy (`tests/test_providers.py`, `tests/test_demo.py`, `brain/tests/test_numeric_regression.py`, `brain/tests/test_factor_model.py`) | 301 passed |
| Python 3.12.15 | syntax parse of all Python files | no errors |

`pip install -e .` and Python 3.13 were not run before publication. The GitHub Actions workflow installs the package with `pip install -e ".[dev]"` and runs pytest and the demo on Python 3.12 and 3.13 on every push and pull request.

## Limitations and caveats

- **Excerpt only.** The data layer, the strategy layer, reporting, scheduling and the rest of the system stay private. Several modules mention callers that are not part of this excerpt.
- **Prompts, not orders.** The expected-shortfall triggers ("trade block", "hard stop") are written as prompts to the investor and are deliberately not activated, and the drawdown stages only ask for a deeper review. Nothing in the code places orders.
- **Synthetic data, no performance claims.** The demo runs on generated data. The report deliberately prints no performance figures, and a test fails if return, Sharpe, Sortino, win-rate or hit-rate vocabulary appears in it.
- **Hard-coded personal rules.** Thresholds in the engine reflect one personal configuration and are not recommendations: stop multipliers (hard stop 1.5 x ATR, soft stop 2.0 to 3.75 x ATR by sector, industry and market cap), position-size limits (risk at most 2 % of capital, notional at most 10 %, a first position in an empty book at most 3 %), drawdown stages (-22 %, -32 %, -42 % from the high-water mark), and a bootstrap P95 drawdown floor of -25 %. They are fixed in code; the TOML config only covers the demo and the illustrative screen.
- **Data shape.** The scoring functions expect fundamentals in the shape of the EODHD fundamentals API response. Another source needs an adapter that produces this shape (see `stocka/providers/base.py`).
- **Pinned numerics.** `brain/tests/test_numeric_regression.py` pins reference values measured with numpy 2.5.2 and arch 8.0.0 under Python 3.14.7 (per the test's comment; the runs in the table above used Python 3.14.9), so `pyproject.toml` restricts numpy to `>=2.5,<2.6`. A numpy feature release may shift the values; they then have to be checked and re-pinned.
- **Language and names.** Code comments and some runtime messages are in German, with umlauts partly written as "ae", "oe", "ue". Veto, flag and note names ending in `_phase1` are historical names from the first build phase; they were kept because downstream code and tests match on them.
- **Adjusted-close caveat.** As stated in `brain/backtest/result.py`: backtest returns would use today's adjusted-close basis, adjustment factors are not point-in-time versioned, and no end-to-end point-in-time fundamentals provider is wired. A backtest built on these records is therefore not a restatement-clean fundamental track record. The caveat is a default field of every `BacktestResult`. No backtest harness is part of this excerpt.
- **Documented method limits.** More than a dozen known method gaps are documented in place and deliberately left open; the rule they follow is no behaviour change in a risk path without an explicit decision. Examples: the procyclical 756-day window of the expected shortfall (`brain/risk/es_historical.py`), the short bootstrap block length (`brain/risk/bootstrap_sim.py`) and the handling of flat series and negative correlations (`brain/risk/correlation.py`).

## Decisions visible in the code

- **Audit gaps:** `compute_dq_score` treats a missing audit value (`None`) as a third, flagged "unverified" state instead of offering a soft mode or a silent default, because a gap in the audit data must neither crash the score nor count as a clean result.
- **Real Estate:** the DQ score keeps the Beneish check for Real Estate instead of exempting the sector like Financials and Utilities, because the author evaluated such an exemption and judged that the M-score works as a signal for REITs; the evidence for that judgement is not part of this excerpt.
- **Forecast lag gate:** `episode_freeze` shows an early-resolved forecast as open until its original window has elapsed, a fixed time lag instead of a freeze triggered by the forecast's state, because a state-triggered freeze would itself leak one bit about the outcome.
- **First-position rule:** sizing rule S3 (a first position of at most 3 %) applies whenever the book holds no other open position, not only to the first position ever, because the "first ever" reading would be used up after one trade and never apply again.
- **Unknown cash:** `PortfolioState.cash_eur` defaults to `None` instead of `0.0`, because 0.0 is a real "fully invested" value and would silently distort the drawdown as if the cash balance were known.
- **ATR smoothing:** `calc_atr` keeps a simple mean of the true range instead of switching to Wilder's smoothing, because the stop rules do not fix the method and a switch would change every stop distance by the same factor (the two ATR variants differed by about 8 % on 60 synthetic bars, as the docstring records).

## How this was built

The author defined the goals, rules and constraints, made the architecture and trade-off decisions, and judged the results: test runs, reports and review findings. The code, the tests and most of the documentation, including this README, were written with Claude Code (Anthropic's AI coding agent) under that direction. The checks were the automated tests, review passes by separate AI agents (a second model checking the first one's output for privacy, for statements against the code and for the impression on a reader) and the author's own checks. The author did not review every line.

## Why a single commit

This repository is a snapshot excerpt, published as one commit. The development history since May 2026 is not included. A few figures about the private repository are in [docs/architecture.md](docs/architecture.md).

The engine code is unchanged from the private originals apart from edited comments and docstrings and the following differences, checked with an AST comparison that ignores docstrings:

- 8 removals: 6 test cases or classes that needed private files or private position data, plus one unused helper function and one unused constant left over from them,
- 33 string replacements: 17 in runtime messages of engine modules (process labels and one script name removed) and 16 in test literals (neutral tickers, dates, a neutral FX rate and round example numbers instead of private values, one print label, one assertion string),
- 6 test identifiers renamed,
- 8 unused imports removed from three test files,
- 1 private test file (`test_external_bear_guard.py`) rewritten from scratch as a `unittest.TestCase` with the same scenarios.

Everything in `stocka/`, `tests/` and `config/`, the new test files in `brain/tests/`, and the packaging and CI files were written for this excerpt.

## License

MIT, see [LICENSE](LICENSE).

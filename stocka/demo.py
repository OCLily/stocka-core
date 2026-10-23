"""`stocka demo`: engine results for one provider, rendered as a Markdown report.

Every figure comes from a function in `brain`. This module aligns the price
series on common dates, builds the daily rebalanced equal-weight basket
series and the basket's sector weights, draws the synthetic calibration
events, calls the engine and formats text. The report holds data-quality
checks and risk measures only and never prints performance figures, not
even for synthetic data.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date

import numpy as np

from brain.analytics.concentration import concentration_metrics
from brain.forecast.calibration import brier_score, ece
from brain.risk.bootstrap_sim import bootstrap_dd_sim
from brain.risk.correlation import (
    CLUSTER_THRESHOLD_DEFAULT,
    CORRELATION_WINDOW_DEFAULT,
    find_correlation_clusters,
    pairwise_correlation_matrix,
)
from brain.risk.drawdown import max_drawdown_from_closes
from brain.risk.es_historical import (
    ES_LOOKBACK_DAYS,
    ES_MIN_OBSERVATIONS,
    ES_PRIMARY_ALPHA,
    compute_es_historical,
)
from brain.risk.es_returns import bounded_returns
from brain.risk.risk_profile import compute_risk_profile
from brain.scoring.beneish import BENEISH_THRESHOLD_GRAY_LOW, BENEISH_THRESHOLD_RED, compute_beneish_m_score
from brain.scoring.dq_score import (
    AUDIT_SIGNALS_REQUIRED_KEYS,
    SLOAN_Q5_ABSOLUTE_VETO_THRESHOLD,
    DQContractError,
    compute_dq_score,
)
from brain.scoring.sloan import compute_sloan_accruals
from brain.scoring.sloan_universe import compute_sloan_universe_snapshot
from stocka.config import Config
from stocka.providers.base import DataProvider, Ticker
from stocka.providers.synthetic import SyntheticProvider
from stocka.screen import EngineResult, ScreenDecision, run_screen

SLOAN_MIN_SAMPLES = 10
CALIBRATION_EVENTS = 500
SNAPSHOT_META_KEYS = ("quintiles", "snapshot_date")

PriceSeries = list[tuple[str, float]]


@dataclass(frozen=True)
class TickerRisk:
    vol_annual: float | None
    max_dd_pct: float | None


@dataclass(frozen=True)
class DemoRun:
    source: str
    data_line: str
    as_of: date
    config: Config
    results: list[EngineResult]
    decisions: list[ScreenDecision]
    risks: dict[str, TickerRisk]
    basket: dict | None
    calibration: list[tuple[str, float, float]]


def sloan_samples(fundamentals_by_symbol: dict[str, dict]) -> list[float] | None:
    snapshot = compute_sloan_universe_snapshot(fundamentals_by_symbol)
    samples = [v for k, v in snapshot.items() if k not in SNAPSHOT_META_KEYS]
    # compute_dq_score vetoes every ticker as "incomplete" when handed a list
    # shorter than the quintile minimum; None gives a soft flag instead.
    return samples if len(samples) >= SLOAN_MIN_SAMPLES else None


def _audit_error(audit: dict | None) -> str:
    if not isinstance(audit, dict):
        return "no audit signals"
    missing = sorted(AUDIT_SIGNALS_REQUIRED_KEYS - set(audit))
    return f"missing {', '.join(missing)}" if missing else "invalid audit signals"


def score_ticker(
    ticker: Ticker, fundamentals: dict, audit: dict | None,
    samples: list[float] | None, as_of: date,
) -> EngineResult:
    m_score = compute_beneish_m_score(fundamentals)["m_score"]
    ratio = compute_sloan_accruals(fundamentals)["accruals_ratio"]
    try:
        dq = compute_dq_score(
            ticker=ticker.symbol, fundamentals=fundamentals, audit_signals=audit,
            sloan_universe=samples, as_of=as_of,
        )
    except DQContractError:
        return EngineResult(ticker.symbol, ticker.sector, None, _audit_error(audit), m_score, ratio, None)
    quintile = dq["components"].get("sloan_quintile", {}).get("quintile")
    return EngineResult(ticker.symbol, ticker.sector, dq, None, m_score, ratio, quintile)


def ticker_risk(closes: PriceSeries) -> TickerRisk:
    return TickerRisk(compute_risk_profile(closes)["vol_annual"], max_drawdown_from_closes(closes)["max_dd_pct"])


def aligned_daily_changes(series: dict[str, PriceSeries]) -> tuple[list[str], np.ndarray]:
    lookups = [dict(s) for s in series.values()]
    common = sorted(set.intersection(*(set(d) for d in lookups)))
    closes = np.array([[d[day] for day in common] for d in lookups], dtype=float).T
    return common, closes[1:] / closes[:-1] - 1.0


def _basket_levels(dates: list[str], changes: np.ndarray) -> PriceSeries:
    levels = 100.0 * np.cumprod(1.0 + changes.mean(axis=1))
    return [(dates[0], 100.0)] + [(d, float(v)) for d, v in zip(dates[1:], levels)]


def _correlation_pairs(symbols: list[str], changes: np.ndarray) -> list | None:
    if len(symbols) < 2 or len(changes) < CORRELATION_WINDOW_DEFAULT:
        return None
    matrix = pairwise_correlation_matrix({s: changes[:, i].tolist() for i, s in enumerate(symbols)})
    return find_correlation_clusters(matrix)


def _concentration(passing: list[EngineResult], fundamentals: dict[str, dict]) -> dict:
    caps = [_market_cap(fundamentals[r.ticker]) for r in passing]
    sectors = Counter(r.sector for r in passing)
    return {
        "sector": concentration_metrics([c / len(passing) for c in sectors.values()], top_n=1),
        "cap": concentration_metrics(caps, top_n=min(5, len(caps))),
    }


def _market_cap(fundamentals: dict) -> float | None:
    try:
        return float((fundamentals.get("Highlights") or {}).get("MarketCapitalization"))
    except (TypeError, ValueError):
        return None


def basket_risk(
    series: dict[str, PriceSeries], *, bootstrap_sims: int, seed: int,
) -> dict:
    dates, changes = aligned_daily_changes(series)
    levels = _basket_levels(dates, changes)
    daily = bounded_returns(levels)["returns"]
    enough_for_es = len(daily) >= ES_MIN_OBSERVATIONS
    return {
        "n_days": len(dates), "first": dates[0], "last": dates[-1],
        "profile": compute_risk_profile(levels),
        "drawdown": max_drawdown_from_closes(levels),
        "es": compute_es_historical(daily) if enough_for_es else None,
        "bootstrap": (
            bootstrap_dd_sim(changes, n_sims=bootstrap_sims, rng_seed=seed)
            if bootstrap_sims and len(changes) >= 2 else None
        ),
        "pairs": _correlation_pairs(list(series), changes),
    }


def calibration_demo(seed: int, n: int = CALIBRATION_EVENTS) -> list[tuple[str, float, float]]:
    rng = np.random.default_rng([seed, 9])
    truth = rng.uniform(0.02, 0.98, n)
    outcomes = (rng.random(n) < truth).astype(int).tolist()
    forecasters = {
        "states the true probability": truth,
        "overconfident (distance from 0.5 stretched x1.6)": np.clip(0.5 + 1.6 * (truth - 0.5), 0.0, 1.0),
        "underconfident (distance from 0.5 shrunk x0.4)": 0.5 + 0.4 * (truth - 0.5),
    }
    return [
        (name, brier_score(p.tolist(), outcomes), ece(p.tolist(), outcomes, n_bins=10))
        for name, p in forecasters.items()
    ]


def _data_line(provider: DataProvider, n_tickers: int, n_days: int, first: str, last: str) -> str:
    if isinstance(provider, SyntheticProvider):
        return (
            f"{n_tickers} tickers, {n_days} business days (weekdays, no holiday calendar) "
            f"from {first} to {last}, seed {provider.seed}"
        )
    return f"{n_tickers} tickers, {n_days} price dates from {first} to {last}, read from local files"


def _passing_basket(
    passing: list[EngineResult], prices: dict[str, PriceSeries],
    fundamentals: dict[str, dict], config: Config,
) -> dict | None:
    if not passing:
        return None
    basket = basket_risk(
        {r.ticker: prices[r.ticker] for r in passing},
        bootstrap_sims=config.demo.bootstrap_sims, seed=config.demo.seed,
    )
    basket["concentration"] = _concentration(passing, fundamentals)
    return basket


def collect(provider: DataProvider, config: Config) -> DemoRun:
    universe = provider.universe()
    if not universe:
        raise ValueError("the provider returned an empty universe")
    prices = {t.symbol: provider.prices(t.symbol) for t in universe}
    fundamentals = {t.symbol: provider.fundamentals(t.symbol) for t in universe}
    as_of = date.fromisoformat(max(p[-1][0] for p in prices.values()))
    samples = sloan_samples(fundamentals)
    results = [
        score_ticker(t, fundamentals[t.symbol], provider.audit_signals(t.symbol), samples, as_of)
        for t in universe
    ]
    decisions = run_screen(results, config.screen)
    passing = [r for r, d in zip(results, decisions) if d.passed]
    basket = _passing_basket(passing, prices, fundamentals, config)
    all_days = sorted({d for p in prices.values() for d, _ in p})
    synthetic = isinstance(provider, SyntheticProvider)
    return DemoRun(
        source="synthetic data, not market data" if synthetic else "user-supplied data, not checked by this tool",
        data_line=_data_line(provider, len(universe), len(all_days), all_days[0], all_days[-1]),
        as_of=as_of, config=config, results=results, decisions=decisions,
        risks={s: ticker_risk(p) for s, p in prices.items()},
        basket=basket, calibration=calibration_demo(config.demo.seed),
    )


def run_demo(provider: DataProvider, config: Config) -> str:
    return render_report(collect(provider, config))


def _num(value: float | None, fmt: str, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:{fmt}}{suffix}"


def _header(run: DemoRun) -> list[str]:
    s = run.config.screen
    yes_no = {True: "yes", False: "no"}
    return [
        f"# stocka-core demo report: {run.source}",
        "",
        "Research output, not investment advice. The report shows data-quality checks and "
        "risk measures only; it deliberately contains no performance figures.",
        "",
        f"- Data: {run.data_line}",
        f"- Scores as of: {run.as_of.isoformat()}",
        f"- Screen settings (illustrative): DQ score >= {s.min_dq_score}, "
        f"Beneish M <= {s.max_beneish_m:.2f}, Sloan Q5 excluded: {yes_no[s.exclude_sloan_q5]}, "
        f"unverified audit inputs excluded: {yes_no[s.require_verified_audit]}",
        "",
    ]


def _screen_row(r: EngineResult, d: ScreenDecision, risk: TickerRisk) -> str:
    dq = "n/a" if r.dq is None else str(r.dq["composite_score"])
    why = "; ".join(list(d.reasons) + [f"note: {n}" for n in d.notes])
    cells = [
        r.ticker, r.sector, dq, _num(r.beneish_m, ".2f"), _num(r.sloan_ratio, ".3f"),
        _num(r.sloan_quintile, "d"), _num(None if risk.vol_annual is None else risk.vol_annual * 100, ".0f", " %"),
        _num(risk.max_dd_pct, ".0f", " %"), "pass" if d.passed else "fail", why,
    ]
    return "| " + " | ".join(cells) + " |"


def _screen_section(run: DemoRun) -> list[str]:
    lines = [
        "## 1. Data-quality screen",
        "",
        "| Ticker | Sector | DQ | Beneish M | Sloan ratio | Sloan Q | Vol p.a. | Max DD | Screen | Why |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    lines += [_screen_row(r, d, run.risks[r.ticker]) for r, d in zip(run.results, run.decisions)]
    n_pass = sum(d.passed for d in run.decisions)
    lines += ["", f"{n_pass} of {len(run.decisions)} tickers pass the screen.", ""]
    flagged = [(r.ticker, r.dq["flags"]) for r in run.results if r.dq is not None and r.dq["flags"]]
    if flagged:
        lines += ["Engine flags (not vetos; the screen may still use them):", ""]
        lines += [f"- {t}: {', '.join(flags)}" for t, flags in flagged]
        lines.append("")
    lines += [
        "How to read it: DQ is the 0-100 composite of `brain.scoring.dq_score`; any engine "
        f"veto fails the screen. Beneish M above {BENEISH_THRESHOLD_RED} is the engine's red zone, "
        f"{BENEISH_THRESHOLD_GRAY_LOW} to {BENEISH_THRESHOLD_RED} the gray zone. The Sloan quintile "
        "ranks balance-sheet accruals within this universe (Q5 = highest) and needs at least "
        f"{SLOAN_MIN_SAMPLES} non-financial tickers; the engine vetoes Q5 only above an accruals "
        f"ratio of {SLOAN_Q5_ABSOLUTE_VETO_THRESHOLD}. Vol and Max DD describe each price series "
        "and do not enter the screen.",
        "",
    ]
    return lines


def _risk_table(basket: dict) -> list[str]:
    dd = basket["drawdown"]
    es = basket["es"]
    vol = basket["profile"]["vol_annual"]
    rows = [
        ("Annualised volatility", _num(None if vol is None else vol * 100, ".1f", " %"),
         "risk_profile.compute_risk_profile"),
        ("Maximum drawdown", f"{_num(dd['max_dd_pct'], '.1f', ' %')} ({dd['peak_date']} to {dd['trough_date']})",
         "drawdown.max_drawdown_from_closes"),
        (f"Expected shortfall {ES_PRIMARY_ALPHA * 100:g} %, 1 day", "n/a" if es is None else f"{es['es_975_pct']:.2f} %",
         "es_historical.compute_es_historical"),
    ]
    boot = basket["bootstrap"]
    if boot is not None:
        rows.append((
            "Bootstrap max drawdown, P95 / P99",
            f"{boot['p95_dd'] * 100:.1f} % / {boot['p99_dd'] * 100:.1f} %",
            "bootstrap_sim.bootstrap_dd_sim",
        ))
    lines = ["| Measure | Value | Engine function |", "|---|---:|---|"]
    return lines + [f"| {a} | {b} | `{c}` |" for a, b, c in rows]


def _risk_notes(basket: dict) -> list[str]:
    es = basket["es"]
    notes = []
    if es is not None:
        notes.append(
            f"- Expected shortfall: at the engine's fixed {ES_PRIMARY_ALPHA * 100:g} % level, historical "
            f"simulation over the last {min(es['n_obs'], ES_LOOKBACK_DAYS)} daily basket changes, mean of "
            f"the worst {es['worst_k_975']}, shown as a loss of basket value."
        )
    boot = basket["bootstrap"]
    if boot is not None:
        notes.append(
            f"- Bootstrap: {boot['n_sims']} stationary-bootstrap paths of the daily basket changes, "
            f"mean block length {boot['block_len']} days ({boot['method']}); CDaR beyond P95: "
            f"{boot['cdar_95plus'] * 100:.1f} %."
        )
    return notes


def _pairs_section(basket: dict) -> list[str]:
    lines = [
        "### Correlated pairs",
        "",
        f"Pairs with |rho| > {CLUSTER_THRESHOLD_DEFAULT} over the last {CORRELATION_WINDOW_DEFAULT} "
        "daily observations (`correlation.pairwise_correlation_matrix` + `find_correlation_clusters`). "
        "These are pairwise edges, not grouped clusters.",
        "",
    ]
    pairs = basket["pairs"]
    if pairs is None:
        return lines + ["Not computed: needs two tickers and a full window.", ""]
    if not pairs:
        return lines + [f"No pair above {CLUSTER_THRESHOLD_DEFAULT}.", ""]
    lines += ["| Pair | rho |", "|---|---:|"]
    return lines + [f"| {a} / {b} | {rho:.2f} |" for a, b, rho in pairs] + [""]


def _concentration_section(basket: dict) -> list[str]:
    sector, cap = basket["concentration"]["sector"], basket["concentration"]["cap"]
    lines = [
        "### Concentration",
        "",
        "`analytics.concentration.concentration_metrics` on the sector weights of the "
        "equal-weight basket and, for context, on the market caps of the same tickers.",
        "",
        "| View | HHI | Effective count | Largest weight |",
        "|---|---:|---:|---:|",
        f"| Sectors (equal weight) | {sector['hhi']:.3f} | {sector['effective_stocks']:.1f} sectors "
        f"| {sector['top_n_weight'] * 100:.0f} % (largest sector) |",
    ]
    if cap["hhi"] is not None:
        lines.append(
            f"| Market cap | {cap['hhi']:.3f} | {cap['effective_stocks']:.1f} names "
            f"| {cap['top_n_weight'] * 100:.0f} % (top {cap['top_n']}) |"
        )
    return lines + [""]


def _basket_section(run: DemoRun) -> list[str]:
    basket = run.basket
    if basket is None:
        return ["## 2. Basket risk", "", "No ticker passed the screen, so no basket was formed.", ""]
    n = sum(d.passed for d in run.decisions)
    lines = [
        f"## 2. Basket risk (equal weight, {n} passing tickers)",
        "",
        f"Daily rebalanced equal weights over {basket['n_days']} common price dates "
        f"({basket['first']} to {basket['last']}). Risk measures only.",
        "",
    ]
    lines += _risk_table(basket) + [""] + _risk_notes(basket) + [""]
    return lines + _pairs_section(basket) + _concentration_section(basket)


def _calibration_section(run: DemoRun) -> list[str]:
    lines = [
        "## 3. Forecast calibration (synthetic forecasts)",
        "",
        f"{CALIBRATION_EVENTS} binary events drawn with known probabilities, scored with "
        "`forecast.calibration.brier_score` and `ece`. The events are unrelated to the price "
        "data; the point is to show how both scores separate a calibrated forecaster from "
        "over- and underconfident ones. Lower is better for both.",
        "",
        "| Forecaster | Brier score | ECE (10 bins) |",
        "|---|---:|---:|",
    ]
    return lines + [f"| {name} | {b:.3f} | {e:.3f} |" for name, b, e in run.calibration] + [""]


def render_report(run: DemoRun) -> str:
    lines = _header(run) + _screen_section(run) + _basket_section(run) + _calibration_section(run)
    return "\n".join(lines).rstrip() + "\n"

"""Per-series performance metrics: Sharpe, Sortino, max drawdown, win rate.

Pure-compute (no DB-access). Degenerate inputs never raise:
compute_sharpe/compute_sortino return None (empty, single-value, zero
deviation), compute_max_drawdown returns (0.0, None, None), and
compute_win_rate/compute_hit_rate/compute_period_return return 0.0.
Only an unknown freq raises BacktestComputeError. compute_max_drawdown is used
by brain/risk/single_asset_metrics.py; compute_sharpe/compute_sortino are
only reported there with include_reward=True (off by default).
"""
from __future__ import annotations

import math
import statistics
from typing import Optional


# Public: this is the house annualization convention, so it must be importable
# rather than re-typed. Two copies still live outside it —
# brain/risk/single_asset_metrics.py (mean * 252) and brain/risk/risk_profile.py
# (_SQRT252) — numerically identical but not derived from here. Keep all three
# in sync.
PERIODS_PER_YEAR = {
    "daily": 252,
    "weekly": 52,
    "monthly": 12,
    "quarterly": 4,
    "yearly": 1,
}


class BacktestComputeError(ValueError):
    """Metric compute received malformed input."""


def periods_per_year(freq: str) -> int:
    if freq not in PERIODS_PER_YEAR:
        raise BacktestComputeError(
            f"unknown freq={freq!r}, expected one of {list(PERIODS_PER_YEAR.keys())}"
        )
    return PERIODS_PER_YEAR[freq]


def compute_sharpe(
    returns: list[float],
    *,
    freq: str = "daily",
    risk_free_rate: float = 0.0,
) -> Optional[float]:
    """Annualized Sharpe ratio. Sample-stdev (N-1).

    sharpe_annual = (mean(r) - rf) / stdev(r) * sqrt(periods_per_year).
    Raises BacktestComputeError on unknown freq. Returns None if N < 2 or stdev == 0.
    """
    # freq validated first, before the degenerate-input returns: otherwise a typo in
    # freq is swallowed exactly when the data is thin.
    ppy = periods_per_year(freq)
    if not returns or len(returns) < 2:
        return None
    sdev = statistics.stdev(returns)
    if sdev <= 0:
        return None
    mean_r = statistics.mean(returns)
    return (mean_r - risk_free_rate) / sdev * math.sqrt(ppy)


def compute_sortino(
    returns: list[float],
    *,
    freq: str = "daily",
    target: float = 0.0,
) -> Optional[float]:
    """Annualized Sortino ratio.

    Downside-deviation = sqrt(mean(min(0, r - target)^2)) over ALL N.
    Raises BacktestComputeError on unknown freq. Returns None if N < 2 or
    downside-deviation == 0.
    """
    ppy = periods_per_year(freq)
    if not returns or len(returns) < 2:
        return None
    downside_sq = [min(0.0, r - target) ** 2 for r in returns]
    dd = math.sqrt(sum(downside_sq) / len(downside_sq))
    if dd <= 0:
        return None
    mean_r = statistics.mean(returns)
    return (mean_r - target) / dd * math.sqrt(ppy)


def compute_max_drawdown(
    equity_curve: list[tuple[str, float]],
) -> tuple[float, Optional[str], Optional[str]]:
    """Largest peak-to-trough drawdown as positive fraction.

    Returns (dd_pct, peak_date, trough_date).
    Empty/single-point/no-drawdown: (0.0, None, None).
    """
    if not equity_curve or len(equity_curve) < 2:
        return 0.0, None, None
    peak_val = equity_curve[0][1]
    peak_date = equity_curve[0][0]
    max_dd = 0.0
    max_dd_peak = peak_date
    max_dd_trough = peak_date
    for d, v in equity_curve:
        if v > peak_val:
            peak_val = v
            peak_date = d
        if peak_val > 0:
            dd = (peak_val - v) / peak_val
            if dd > max_dd:
                max_dd = dd
                max_dd_peak = peak_date
                max_dd_trough = d
    if max_dd == 0.0:
        return 0.0, None, None
    return max_dd, max_dd_peak, max_dd_trough


# compute_win_rate, compute_hit_rate and compute_period_return below have no
# production caller (their consumer was the walk-forward harness, which is not
# part of this package). Kept deliberately: reusable for a future backtest
# harness, near-zero upkeep cost.
def compute_win_rate(returns: list[float]) -> float:
    """Fraction of returns > 0. Empty list returns 0.0."""
    if not returns:
        return 0.0
    wins = sum(1 for r in returns if r > 0)
    return wins / len(returns)


def compute_hit_rate(ticker_attribution: dict) -> float:
    """Fraction of tickers with total_pnl_pct > 0."""
    if not ticker_attribution:
        return 0.0
    wins = sum(
        1 for v in ticker_attribution.values()
        if isinstance(v, dict) and v.get("total_pnl_pct", 0.0) > 0
    )
    return wins / len(ticker_attribution)


def compute_period_return(
    start_prices: dict,
    end_prices: dict,
    weights: dict,
) -> float:
    """Weighted period return given start/end prices and ticker weights.

    Per-ticker return: (end - start) / start.
    Tickers missing in either start, end, or weights, or with zero start,
    are excluded (used-weight basket renormalized).

    COVERAGE IS NOT REPORTED. The scalar from 1-of-20 priced tickers is
    indistinguishable from a fully covered period return — the caller cannot tell.
    Returns 0.0 if no usable tickers (rather than raise).
    """
    # The house pattern for exactly this is brain/analytics/concentration.py,
    # which always returns n_input/n_priced/n_missing alongside the metric. Adding
    # it here changes the return contract (scalar -> mapping) and is deferred, not
    # dismissed: the function has no production caller, so the shape is best
    # decided when a backtest harness revives it. Do not put this into production
    # use as a bare scalar.
    if not weights:
        return 0.0
    contributions = []
    used_weight = 0.0
    for ticker, w in weights.items():
        sp = start_prices.get(ticker)
        ep = end_prices.get(ticker)
        if sp is None or ep is None or sp == 0:
            continue
        r = (ep - sp) / sp
        contributions.append(r * w)
        used_weight += w
    if used_weight <= 0:
        return 0.0
    return sum(contributions) / used_weight

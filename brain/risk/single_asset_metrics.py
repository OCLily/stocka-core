"""Per-asset risk/reward metrics as a calibrated base-rate description.

Return-FREE axes (recovery, max-drawdown, vol) carry no reward in the numerator and are
always on. Reward axes (Sharpe/Sortino/Calmar, reward in numerator) are gated behind
`include_reward`: they are admitted only as explicitly requested, two-sided valuation
context, which is why risk_profile.py leaves them out entirely. rf=0 is the declared,
built-in convention. Pure function, no I/O, no verdict, no cross-ticker comparison,
no threshold/action, no ranking.
"""
from __future__ import annotations

import statistics

from brain.risk.es_returns import bounded_returns
from brain.risk.risk_profile import compute_risk_profile
from brain.backtest.metrics import compute_sharpe, compute_sortino, compute_max_drawdown

RF = 0.0   # risk-free rate declared = 0; no rate provider


def single_asset_metrics(dated_closes, *, include_reward: bool = False) -> dict:
    pairs = list(dated_closes)
    # compute_risk_profile already exposes underwater_longest/current_bars + vol/downside-dev
    # (no reward in numerator) — reuse its public dict, don't import the private
    # _underwater_runs helper.
    rp = compute_risk_profile(pairs)                       # vol/downside-dev/underwater, reward-free
    equity = [(d, c) for d, c in pairs]                     # (date, adj_close) IS the equity curve
    max_dd, dd_peak, dd_trough = compute_max_drawdown(equity)

    out = {
        "max_drawdown": max_dd, "max_drawdown_peak": dd_peak, "max_drawdown_trough": dd_trough,
        "recovery_underwater_bars_longest": rp["underwater_longest_bars"],
        "recovery_underwater_bars_current": rp["underwater_current_bars"],
        "vol_annual": rp["vol_annual"], "downside_dev_annual": rp["downside_dev_annual"],
        "n": rp["n"], "n_excluded": rp["n_excluded"],
        "notes": rp.get("vol_note"),
        "rf": RF,
        "sharpe": None, "sortino": None, "calmar": None,   # gated by include_reward below
    }
    if include_reward:
        r = bounded_returns(pairs)["returns"]
        out["sharpe"] = compute_sharpe(r, risk_free_rate=RF)      # rf=0
        out["sortino"] = compute_sortino(r, target=RF)            # target=0
        # Calmar = annualized return / MaxDD; None if MaxDD==0 or too little history
        out["calmar"] = _calmar(r, max_dd)
    return out


def _calmar(returns, max_dd):
    # Annualisierung = arithm. Mittel * 252 (NICHT CAGR/geometrisch) — bewusst die
    # einfache Konvention, konsistent mit compute_sharpe/compute_sortino (mean-basiert).
    # Eine Anzeige sollte das kenntlich machen, etwa als "(rf=0; Calmar: arithm. Mittel*252)".
    if not returns or max_dd <= 0:
        return None
    ann_return = statistics.mean(returns) * 252
    return ann_return / max_dd

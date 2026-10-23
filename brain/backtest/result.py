"""BacktestResult dataclass — output format of a backtest run.

JSON-serializable. None-values → JSON null.

The run harness that produced these records is not part of this package; the
dataclass is kept deliberately because its shape and the
PIT_ADJUSTED_CLOSE_CAVEAT honesty-guard field are reusable for a future
backtest harness.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional

# Honesty guard: backtest returns use today's adjusted-close basis — adjustment
# factors are NOT point-in-time versioned — and no end-to-end fundamental PIT
# provider is wired. This caveat rides on EVERY BacktestResult by default so no
# consumer can present a backtest as a restatement-PIT-clean fundamental
# track-record. Enforced by construction (a default field), not by convention.
PIT_ADJUSTED_CLOSE_CAVEAT = (
    "Backtest-Renditen nutzen die HEUTIGE adjusted-close-Basis (Splits/Dividenden "
    "rückwirkend); Adjustment-Faktoren sind NICHT point-in-time versioniert "
    "und es ist kein End-to-End-Fundamental-PIT-Provider verdrahtet. "
    "→ Dies ist KEIN restatement-PIT-sauberer Fundamental-Backtest-Track-Record."
)


@dataclass
class BacktestResult:
    strategy_name: str
    universe_id: str
    start_date: str
    end_date: str
    rebalance_freq: str
    top_n: int
    initial_capital: float
    final_capital: float
    total_return_pct: float
    annualized_return_pct: float
    sharpe: Optional[float]
    sortino: Optional[float]
    max_drawdown_pct: float
    max_drawdown_start: Optional[str]
    max_drawdown_end: Optional[str]
    win_rate: float
    hit_rate: float
    n_rebalances: int
    n_unique_positions: int
    returns_series: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)
    ticker_attribution: dict = field(default_factory=dict)
    flags: list = field(default_factory=list)
    pit_caveat: str = PIT_ADJUSTED_CLOSE_CAVEAT

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, default=str)

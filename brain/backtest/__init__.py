"""brain.backtest — metrics + result building-blocks for backtest analysis.

The walk-forward harness itself is not part of this package. The metrics are
used by brain/risk/single_asset_metrics.py: compute_max_drawdown always,
compute_sharpe/compute_sortino only with include_reward=True (off by
default). BacktestResult is kept as a result shape that carries a
point-in-time honesty caveat by construction.

Public API:
  - BacktestResult dataclass
  - BacktestComputeError (ValueError) — metrics-compute edge-cases
"""
from brain.backtest.metrics import BacktestComputeError
from brain.backtest.result import BacktestResult

__all__ = [
    "BacktestResult",
    "BacktestComputeError",
]

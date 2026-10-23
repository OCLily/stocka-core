"""Tests for brain.backtest.metrics and brain.backtest.result.

Covers Sharpe/Sortino annualization, max drawdown, win rate, weighted
period return, and JSON serialization of BacktestResult (incl. None values).

Run: python3 -m brain.tests.test_backtest_metrics
"""
from __future__ import annotations

import json
import math
import statistics
import unittest


# ─── Metrics Tests (6) ─────────────────────────────────────────────


class TestMetrics(unittest.TestCase):
    def test_metrics_sharpe_known_series_daily(self):
        from brain.backtest.metrics import compute_sharpe
        returns = [0.01, 0.02, -0.005, 0.015]
        sharpe = compute_sharpe(returns, freq="daily")
        expected = (statistics.mean(returns) / statistics.stdev(returns)) * math.sqrt(252)
        self.assertIsNotNone(sharpe)
        self.assertAlmostEqual(sharpe, expected, places=9)

    def test_metrics_sharpe_quarterly_annualization(self):
        from brain.backtest.metrics import compute_sharpe
        returns = [0.01, 0.02, -0.005, 0.015]
        sharpe_q = compute_sharpe(returns, freq="quarterly")
        sharpe_d = compute_sharpe(returns, freq="daily")
        self.assertIsNotNone(sharpe_q)
        self.assertIsNotNone(sharpe_d)
        self.assertGreater(sharpe_d, sharpe_q)
        self.assertAlmostEqual(sharpe_d / sharpe_q, math.sqrt(63), places=6)

    def test_metrics_sortino_downside_only(self):
        from brain.backtest.metrics import compute_sortino, compute_sharpe
        returns = [0.05, -0.02, 0.03, 0.01]
        sortino = compute_sortino(returns, freq="daily")
        sharpe = compute_sharpe(returns, freq="daily")
        self.assertIsNotNone(sortino)
        self.assertIsNotNone(sharpe)
        self.assertGreater(sortino, sharpe)

    def test_metrics_max_drawdown(self):
        from brain.backtest.metrics import compute_max_drawdown
        equity = [("d1", 100.0), ("d2", 120.0), ("d3", 90.0), ("d4", 100.0)]
        dd_pct, dd_start, dd_end = compute_max_drawdown(equity)
        self.assertAlmostEqual(dd_pct, 0.25, places=9)
        self.assertEqual(dd_start, "d2")
        self.assertEqual(dd_end, "d3")

    def test_metrics_win_rate(self):
        from brain.backtest.metrics import compute_win_rate
        returns = [0.01, -0.02, 0.03, 0.05, -0.01]
        self.assertAlmostEqual(compute_win_rate(returns), 0.6, places=9)

    def test_metrics_compute_period_return_basic(self):
        from brain.backtest.metrics import compute_period_return
        start = {"A": 100.0, "B": 200.0}
        end = {"A": 110.0, "B": 210.0}
        weights = {"A": 0.5, "B": 0.5}
        period_ret = compute_period_return(start, end, weights)
        self.assertAlmostEqual(period_ret, 0.075, places=9)


# ─── Result Tests (2) ──────────────────────────────────────────────


class TestResult(unittest.TestCase):
    def test_backtest_result_json_serializable(self):
        from brain.backtest.result import BacktestResult
        result = BacktestResult(
            strategy_name="test",
            universe_id="synthetic_5",
            start_date="2024-01-02",
            end_date="2024-12-31",
            rebalance_freq="quarterly",
            top_n=3,
            initial_capital=100_000.0,
            final_capital=110_000.0,
            total_return_pct=0.10,
            annualized_return_pct=0.10,
            sharpe=1.5,
            sortino=2.0,
            max_drawdown_pct=0.05,
            max_drawdown_start="2024-03-01",
            max_drawdown_end="2024-04-01",
            win_rate=0.6,
            hit_rate=0.7,
            n_rebalances=4,
            n_unique_positions=5,
            returns_series=[("2024-04-01", 0.02), ("2024-07-01", 0.03)],
            equity_curve=[("2024-01-02", 100_000.0), ("2024-12-31", 110_000.0)],
            ticker_attribution={"SYN1.US": {"periods_held": 4, "total_pnl_pct": 0.05}},
            flags=[],
        )
        parsed = json.loads(result.to_json())
        self.assertEqual(parsed["strategy_name"], "test")
        self.assertEqual(parsed["sharpe"], 1.5)
        self.assertEqual(parsed["n_unique_positions"], 5)

    def test_backtest_result_to_json_handles_none_values(self):
        from brain.backtest.result import BacktestResult
        result = BacktestResult(
            strategy_name="t", universe_id="u",
            start_date="2024-01-02", end_date="2024-01-02",
            rebalance_freq="quarterly", top_n=1,
            initial_capital=100.0, final_capital=100.0,
            total_return_pct=0.0, annualized_return_pct=0.0,
            sharpe=None, sortino=None,
            max_drawdown_pct=0.0,
            max_drawdown_start=None, max_drawdown_end=None,
            win_rate=0.0, hit_rate=0.0,
            n_rebalances=1, n_unique_positions=0,
            returns_series=[], equity_curve=[],
            ticker_attribution={}, flags=["empty_universe"],
        )
        parsed = json.loads(result.to_json())
        self.assertIsNone(parsed["sharpe"])
        self.assertIsNone(parsed["max_drawdown_start"])


def run() -> tuple[int, int]:
    import sys
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return result.testsRun, len(result.failures) + len(result.errors)


if __name__ == "__main__":
    total, failed = run()
    print(f"\n=== test_backtest_metrics: {total - failed}/{total} passed ===")
    raise SystemExit(0 if failed == 0 else 1)

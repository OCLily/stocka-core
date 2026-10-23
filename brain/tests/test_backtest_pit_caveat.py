"""Offline test for the point-in-time (PIT) honesty guard on BacktestResult.

Backtest returns use today's adjusted-close basis (adjustment factors are NOT
point-in-time versioned), and no end-to-end fundamental PIT provider is wired.
To prevent any consumer from presenting a backtest as a restatement-PIT-clean
fundamental track-record, EVERY BacktestResult must carry an explicit honesty
caveat by construction (a default field, not a "remember to add it"
convention).
"""
from __future__ import annotations

import json
import unittest

from brain.backtest.result import BacktestResult, PIT_ADJUSTED_CLOSE_CAVEAT


def _minimal_result(**kw) -> BacktestResult:
    base = dict(
        strategy_name="s", universe_id="u", start_date="2024-01-01",
        end_date="2024-12-31", rebalance_freq="monthly", top_n=5,
        initial_capital=100_000.0, final_capital=110_000.0,
        total_return_pct=10.0, annualized_return_pct=10.0,
        sharpe=None, sortino=None, max_drawdown_pct=-5.0,
        max_drawdown_start=None, max_drawdown_end=None,
        win_rate=0.5, hit_rate=0.5, n_rebalances=12, n_unique_positions=20,
    )
    base.update(kw)
    return BacktestResult(**base)


class PitHonestyGuardTests(unittest.TestCase):

    def test_default_caveat_present_and_nonempty(self):
        r = _minimal_result()
        self.assertTrue(r.pit_caveat)
        self.assertEqual(r.pit_caveat, PIT_ADJUSTED_CLOSE_CAVEAT)

    def test_caveat_denies_pit_clean_track_record(self):
        # The caveat must explicitly state it is NOT a PIT-clean track-record.
        c = PIT_ADJUSTED_CLOSE_CAVEAT
        self.assertIn("KEIN", c)
        self.assertIn("PIT", c)
        self.assertIn("Track-Record", c)

    def test_to_json_includes_caveat(self):
        r = _minimal_result()
        d = json.loads(r.to_json())
        self.assertIn("pit_caveat", d)
        self.assertEqual(d["pit_caveat"], PIT_ADJUSTED_CLOSE_CAVEAT)


if __name__ == "__main__":
    unittest.main()

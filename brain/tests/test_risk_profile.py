"""risk_profile.compute_risk_profile (pure risk measures).

Covers annualized volatility, downside deviation (/n target-semideviation),
longest underwater phase. Reine Risiko-Maße, KEIN Return-Reward. Expected values
are computed INDEPENDENTLY here (not mirrored from the implementation).

Run: python3 -m brain.tests.test_risk_profile
"""
from __future__ import annotations

import math
import statistics
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from brain.risk.risk_profile import _underwater_runs, compute_risk_profile  # noqa: E402

_SQRT252 = math.sqrt(252)


def _bizdays(start: str, n: int) -> list:
    """n consecutive business days (Mon-Fri) as ISO strings from start."""
    out, d = [], date.fromisoformat(start)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def _closes_from_returns(returns: list, *, base: float = 100.0, start="2023-01-02") -> list:
    """Build [(date, adj)] on CONSECUTIVE business days from a return list (no gaps,
    so bounded_returns reproduces exactly `returns`)."""
    dates = _bizdays(start, len(returns) + 1)
    closes, lvl = [(dates[0], base)], base
    for i, r in enumerate(returns):
        lvl *= (1.0 + r)
        closes.append((dates[i + 1], lvl))
    return closes


class RiskProfileMath(unittest.TestCase):
    def test_vol_and_downside_against_independent_math(self):
        # 520 returns (>=500) → no "grobe Größenordnung" qualifier, notes stay None.
        returns = [0.012 if i % 2 == 0 else -0.008 for i in range(520)]
        series = _closes_from_returns(returns)
        out = compute_risk_profile(series)

        exp_vol = statistics.stdev(returns) * _SQRT252           # ddof=1
        n = len(returns)
        exp_dd = math.sqrt(sum(x * x for x in returns if x < 0) / n) * _SQRT252  # /n convention
        self.assertAlmostEqual(out["vol_annual"], exp_vol, places=6)
        self.assertAlmostEqual(out["downside_dev_annual"], exp_dd, places=6)
        self.assertIsNone(out["vol_note"])
        self.assertIsNone(out["downside_dev_note"])

    def test_no_downside_days_returns_none_not_zero(self):
        returns = [0.001] * 80          # strictly positive → no down days
        out = compute_risk_profile(_closes_from_returns(returns))
        self.assertIsNone(out["downside_dev_annual"])
        self.assertIn("keine Abwärtstage", out["downside_dev_note"])
        # vol still computable
        self.assertIsNotNone(out["vol_annual"])

    def test_underwater_longest_and_current_hand_counted(self):
        # running peak = 100 at idx0. idx1..7 (98,96,94,92,90,95,99) are ALL < 100
        # (recovery to 95/99 is still below the peak) → 7 consecutive underwater bars.
        # idx8 = 101 → new peak, run resets. idx9,10 = 100,100 < 101 → 2 underwater.
        levels = [100, 98, 96, 94, 92, 90, 95, 99, 101, 100, 100]
        dates = _bizdays("2023-01-02", len(levels))
        series = list(zip(dates, [float(x) for x in levels]))
        out = compute_risk_profile(series)
        self.assertEqual(out["underwater_longest_bars"], 7)
        self.assertEqual(out["underwater_current_bars"], 2)  # trailing run below peak 101

    def test_monotone_rising_no_underwater_no_crash(self):
        returns = [0.001] * 80
        out = compute_risk_profile(_closes_from_returns(returns))
        self.assertEqual(out["underwater_longest_bars"], 0)
        self.assertEqual(out["underwater_current_bars"], 0)  # max([], default=0), no ValueError

    def test_below_floor_not_computable_never_zero(self):
        returns = [0.01, -0.01, 0.01]      # n=3 << 60
        out = compute_risk_profile(_closes_from_returns(returns))
        for key in ("vol_annual", "downside_dev_annual"):
            self.assertIsNone(out[key])
        self.assertIn("nicht berechenbar", out["vol_note"])
        self.assertIn("n=3", out["vol_note"])

    def test_thin_history_qualifier_between_floor_and_500(self):
        returns = [0.012 if i % 2 == 0 else -0.008 for i in range(80)]  # 60<=80<500
        out = compute_risk_profile(_closes_from_returns(returns))
        self.assertIsNotNone(out["vol_annual"])
        self.assertIn("grobe Größenordnung", out["vol_note"])

    def test_min_obs_boundary_exactly_60_is_computable(self):
        # Grenzfall: die anderen Tests liegen nur beidseitig von _MIN_OBS=60 (n=3, n=80).
        # n=60 exakt pinnt `<` gegen `<=` in risk_profile.py (`if n < _MIN_OBS`).
        returns = [0.012 if i % 2 == 0 else -0.008 for i in range(60)]
        out = compute_risk_profile(_closes_from_returns(returns))
        self.assertEqual(out["n"], 60)
        self.assertIsNotNone(out["vol_annual"])
        self.assertNotIn("nicht berechenbar", out["vol_note"])
        self.assertIn("grobe Größenordnung", out["vol_note"])  # 60 < 500

    def test_short_history_boundary_exactly_500_has_no_qualifier(self):
        # Grenzfall: die anderen Tests liegen nur beidseitig von _SHORT_HISTORY_N=500
        # (n=80, n=520). n=500 exakt pinnt `<` gegen `<=` in risk_profile.py
        # (`if n < _SHORT_HISTORY_N`).
        returns = [0.012 if i % 2 == 0 else -0.008 for i in range(500)]
        out = compute_risk_profile(_closes_from_returns(returns))
        self.assertEqual(out["n"], 500)
        self.assertIsNone(out["vol_note"])
        self.assertIsNone(out["downside_dev_note"])

    def test_empty_and_single_no_exception(self):
        for series in ([], [("2023-01-02", 100.0)]):
            out = compute_risk_profile(series)
            self.assertIsNone(out["vol_annual"])
            self.assertEqual(out["underwater_longest_bars"], 0)

    def test_underwater_inf_bar_no_longer_poisons_rest_of_series(self):
        # Without the guard, a single inf bar would poison `peak` permanently -> (2, 2)
        # (both later bars falsely counted as underwater). Guarded: the inf bar is
        # skipped entirely, series reads as [50.0, 60.0, 70.0] -> monotone rising,
        # no underwater run at all.
        self.assertEqual(_underwater_runs([50.0, math.inf, 60.0, 70.0]), (0, 0))

    def test_underwater_zero_bar_skipped_not_counted_as_one_bar(self):
        # 0.0 is not `> 0` -> unusable, skipped like inf. Without the guard this
        # counted as a single underwater bar (0.0 < peak=50, then 60 sets a new peak).
        # Guarded: 0.0 drops out, series reads as [50.0, 60.0] -> no underwater run.
        self.assertEqual(_underwater_runs([50.0, 0.0, 60.0]), (0, 0))

    def test_underwater_normal_data_unaffected_by_guard(self):
        # Regression: no inf/0.0 bars present -> guard is a no-op, same result as
        # test_underwater_longest_and_current_hand_counted's hand-counted expectation.
        levels = [100.0, 98.0, 96.0, 94.0, 92.0, 90.0, 95.0, 99.0, 101.0, 100.0, 100.0]
        self.assertEqual(_underwater_runs(levels), (7, 2))


def suite():
    return unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])


if __name__ == "__main__":
    r = unittest.TextTestRunner(verbosity=1).run(suite())
    sys.exit(0 if r.wasSuccessful() else 1)

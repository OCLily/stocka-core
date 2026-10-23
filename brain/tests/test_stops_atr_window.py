"""atr_window pin-through regression + defensive-sector fail-closed guard.

1. Regression pin: compute_hybrid_stop / compute_hybrid_stop_from_ohlc carry
   the ATR window actually used through into the returned dict's "atr_window"
   key, so a caller's stop rationale reflects reality instead of a hardcoded
   "ATR(20)".

2. Defensive-sector fail-closed direct-call guard: compute_hybrid_stop raises
   StopCalcError when called directly with gics_sector in the defensive set
   (Utilities, Consumer Staples) AND an explicit atr_window that isn't 14 —
   instead of silently computing a non-conforming soft-stop (the defensive
   rule pairs window 14 with multiplier 2.0x and must not be split).
   compute_hybrid_stop_from_ohlc always selects the correct window itself and
   must remain unaffected (regression test below).

Offline, no DB, no network. Run:
  python3 -m brain.tests.test_stops_atr_window
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from brain.risk.stops import (
    ATR_WINDOW_DEFAULT,
    ATR_WINDOW_DEFENSIVE,
    StopCalcError,
    compute_hybrid_stop,
    compute_hybrid_stop_from_ohlc,
)


def _synthetic_ohlc(n: int = 30):
    highs = [200.0 + 2 * (i % 5) for i in range(n)]
    lows = [197.0 + 2 * (i % 5) for i in range(n)]
    closes = [199.0 + 2 * (i % 5) for i in range(n)]
    return highs, lows, closes


class TestAtrWindowPinThrough(unittest.TestCase):
    """(1) Regression pin: atr_window in the returned dict matches the window
    actually used to compute the ATR — not a hardcoded literal."""

    def test_from_ohlc_default_sector_pins_window_20(self):
        highs, lows, closes = _synthetic_ohlc()
        result = compute_hybrid_stop_from_ohlc(
            highs=highs, lows=lows, closes=closes,
            entry_price=200.0, market_cap=5e10,
            gics_sector="Information Technology",
        )
        self.assertEqual(result["atr_window"], ATR_WINDOW_DEFAULT)

    def test_from_ohlc_defensive_sector_pins_window_14(self):
        highs, lows, closes = _synthetic_ohlc()
        result = compute_hybrid_stop_from_ohlc(
            highs=highs, lows=lows, closes=closes,
            entry_price=200.0, market_cap=5e10,
            gics_sector="Utilities",
        )
        self.assertEqual(result["atr_window"], ATR_WINDOW_DEFENSIVE)

    def test_primitive_passes_through_explicit_atr_window(self):
        result = compute_hybrid_stop(
            entry_price=200.0, atr=3.0, market_cap=5e10,
            gics_sector="Information Technology",
            atr_window=20,
        )
        self.assertEqual(result["atr_window"], 20)

    def test_primitive_defaults_atr_window_to_none_when_uncalled(self):
        """Caller that never mentions atr_window gets None, not a fabricated
        number — 'visibly unknown beats wrongly documented' (docstring)."""
        result = compute_hybrid_stop(
            entry_price=200.0, atr=3.0, market_cap=5e10,
            gics_sector="Information Technology",
        )
        self.assertIsNone(result["atr_window"])


class TestDefensiveDirectCallFailsClosed(unittest.TestCase):
    """(2) A defensive-sector direct call to the primitive with a non-14
    atr_window must raise StopCalcError instead of silently computing."""

    def test_defensive_sector_wrong_window_raises(self):
        with self.assertRaises(StopCalcError):
            compute_hybrid_stop(
                entry_price=200.0, atr=3.0, market_cap=5e10,
                gics_sector="Utilities",
                atr_window=ATR_WINDOW_DEFAULT,
            )

    def test_consumer_staples_wrong_window_raises(self):
        with self.assertRaises(StopCalcError):
            compute_hybrid_stop(
                entry_price=200.0, atr=3.0, market_cap=5e10,
                gics_sector="Consumer Staples",
                atr_window=25,
            )

    def test_defensive_sector_correct_window_still_computes(self):
        result = compute_hybrid_stop(
            entry_price=200.0, atr=3.0, market_cap=5e10,
            gics_sector="Utilities",
            atr_window=ATR_WINDOW_DEFENSIVE,
        )
        self.assertEqual(result["basis"], "defensive-override")
        self.assertEqual(result["atr_window"], ATR_WINDOW_DEFENSIVE)

    def test_defensive_sector_no_window_given_does_not_raise(self):
        """atr_window is optional, not mandatory — None means the caller
        didn't say, which stays a documented gap, not a hard fail."""
        result = compute_hybrid_stop(
            entry_price=200.0, atr=3.0, market_cap=5e10,
            gics_sector="Utilities",
        )
        self.assertEqual(result["basis"], "defensive-override")
        self.assertIsNone(result["atr_window"])

    def test_non_defensive_sector_any_window_does_not_raise(self):
        """The guard is scoped to defensive sectors only."""
        result = compute_hybrid_stop(
            entry_price=200.0, atr=3.0, market_cap=5e10,
            gics_sector="Information Technology",
            atr_window=14,
        )
        self.assertEqual(result["atr_window"], 14)


class TestFromOhlcPathUnaffectedByDefensiveGuard(unittest.TestCase):
    """Regression: compute_hybrid_stop_from_ohlc always selects the
    sector-correct window itself and must never trip the guard."""

    def test_defensive_sector_from_ohlc_still_succeeds(self):
        highs, lows, closes = _synthetic_ohlc()
        result = compute_hybrid_stop_from_ohlc(
            highs=highs, lows=lows, closes=closes,
            entry_price=200.0, market_cap=5e10,
            gics_sector="Utilities",
        )
        self.assertEqual(result["atr_window"], ATR_WINDOW_DEFENSIVE)
        self.assertEqual(result["basis"], "defensive-override")

    def test_non_defensive_sector_from_ohlc_still_succeeds(self):
        highs, lows, closes = _synthetic_ohlc()
        result = compute_hybrid_stop_from_ohlc(
            highs=highs, lows=lows, closes=closes,
            entry_price=200.0, market_cap=5e10,
            gics_sector="Consumer Discretionary",
        )
        self.assertEqual(result["atr_window"], ATR_WINDOW_DEFAULT)


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(unittest.TestLoader().loadTestsFromModule(
        sys.modules[__name__]
    ))
    sys.exit(0 if result.wasSuccessful() else 1)

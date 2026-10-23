"""Boundary pins for the capital-protection core.

Covers brain/state/dd_state_machine.py and brain/risk/stops.py. Each test pins
the behaviour AT a boundary (exactly on, just below, just above), with literal
numbers instead of the module constants, so a changed constant cannot move
test and code together:

  peak_ratchet_block_reason  a measurement equal to the anchor does not lift it
  apply_tier_transition      a 0.0001 pp DD change is recorded; the cooling-off
                             window is exactly 168 h
  calc_atr                   window >= 2; needs window + 1 bars
  compute_hybrid_stop        hard_stop == 0 and atr == 0 are rejected; the
                             ordering soft < hard < entry is strict
  get_atr_multiplier         market_cap <= 0 fails closed; cap-size bands at
                             10e9 (exclusive) and 2e9 (inclusive)

Pure in-memory: no state file, no DB, no network.
Run: python3 -m brain.tests.test_capital_rules_pins
"""
from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from brain.risk.stops import (  # noqa: E402
    StopCalcError,
    calc_atr,
    compute_hybrid_stop,
    compute_hybrid_stop_from_ohlc,
    get_atr_multiplier,
)
from brain.state import dd_state_machine as sm  # noqa: E402
from brain.state.portfolio_state import PortfolioState  # noqa: E402

_T0 = datetime(2025, 6, 2, 12, 0, 0, tzinfo=timezone.utc)


def _flat_ohlc(n: int):
    return [101.0] * n, [99.0] * n, [100.0] * n


def _measurement(state, *, adjusted, **overrides):
    result = {
        "no_positions": False, "equity_eur": adjusted + state.net_flows_eur,
        "adjusted_equity_eur": adjusted, "net_flows_eur": state.net_flows_eur,
        "cash_eur": 90_000.0, "positions_value_eur": adjusted - 90_000.0,
        "currency": "EUR", "fx_usd_eur": 0.92, "peak_ytd": state.peak_ytd,
        "current_dd_fraction": 0.0, "dd_tier": 0, "persisted_tier": state.tier,
        "above_peak": adjusted > state.peak_ytd, "incomplete": False,
        "missing_prices": [], "stale_tickers": [], "per_position": [],
        "priced_at": "2025-06-02T12:00:00+00:00",
    }
    result.update(overrides)
    return result


class TestDdStateMachinePins(unittest.TestCase):

    def test_cooling_off_window_is_168_hours_literally(self):
        st = PortfolioState()
        sm.apply_tier_transition(st, -33.0, now_utc=_T0)
        self.assertEqual(st.cooling_off_until,
                         (_T0 + timedelta(hours=168)).isoformat())
        held = sm.apply_tier_transition(
            st, -5.0, now_utc=_T0 + timedelta(hours=167, minutes=59))
        self.assertEqual(held["new_tier"], 2, held)
        self.assertIs(held["blocked_by_cooling_off"], True, held)
        released = sm.apply_tier_transition(
            st, -5.0, now_utc=_T0 + timedelta(hours=168))
        self.assertEqual(released["new_tier"], 0, released)
        self.assertIs(released["blocked_by_cooling_off"], False, released)

    def test_a_dd_change_of_one_ten_thousandth_point_is_recorded(self):
        # wire_current_dd_from_fraction rounds to 4 decimals, so 0.0001 pp is
        # the finest real change; it must reach current_dd and the audit_log.
        st = PortfolioState()
        sm.apply_tier_transition(st, -10.0, now_utc=_T0)
        rows = len(st.audit_log)
        sm.apply_tier_transition(st, -10.0001, now_utc=_T0 + timedelta(hours=1))
        self.assertEqual(st.current_dd, -10.0001)
        self.assertEqual(len(st.audit_log), rows + 1, st.audit_log[-1:])

    def test_an_identical_dd_writes_no_audit_row(self):
        st = PortfolioState()
        sm.apply_tier_transition(st, -10.0001, now_utc=_T0)
        rows = len(st.audit_log)
        sm.apply_tier_transition(st, -10.0001, now_utc=_T0 + timedelta(hours=1))
        self.assertEqual(len(st.audit_log), rows)

    def test_ratchet_refuses_a_measurement_equal_to_the_anchor(self):
        # Defensive redundancy: the equity measurement sets above_peak only
        # when adjusted > peak, but the guard must hold on its own, even when
        # the flag claims otherwise.
        st = PortfolioState(peak_ytd=100_000.0)
        rows = len(st.audit_log)
        forged = _measurement(st, adjusted=100_000.0, above_peak=True)
        self.assertIsNotNone(sm.peak_ratchet_block_reason(st, forged))
        self.assertIsNone(sm.ratchet_peak_to_measured_equity(st, forged))
        self.assertEqual(st.peak_ytd, 100_000.0)
        self.assertEqual(len(st.audit_log), rows)

    def test_ratchet_lifts_one_cent_above_the_anchor(self):
        st = PortfolioState(peak_ytd=100_000.0)
        lifted = sm.ratchet_peak_to_measured_equity(
            st, _measurement(st, adjusted=100_000.01))
        self.assertEqual(lifted, {"old_peak_ytd": 100_000.0,
                                  "new_peak_ytd": 100_000.01})
        self.assertEqual(st.peak_ytd, 100_000.01)


class TestAtrWindowPins(unittest.TestCase):

    def test_window_one_is_rejected_window_two_is_the_minimum(self):
        h, l, c = _flat_ohlc(3)
        with self.assertRaises(StopCalcError):
            calc_atr(h, l, c, window=1)
        self.assertEqual(calc_atr(h, l, c, window=2), 2.0)

    def test_exactly_window_bars_is_one_bar_short(self):
        for window in (20, 14):
            with self.subTest(window=window):
                h, l, c = _flat_ohlc(window)
                with self.assertRaises(StopCalcError):
                    calc_atr(h, l, c, window=window)
                h, l, c = _flat_ohlc(window + 1)
                self.assertEqual(calc_atr(h, l, c, window=window), 2.0)

    def test_from_ohlc_needs_twenty_one_bars_default_fifteen_defensive(self):
        for sector, bars in (("Information Technology", 21), ("Utilities", 15)):
            with self.subTest(sector=sector):
                h, l, c = _flat_ohlc(bars - 1)
                with self.assertRaises(StopCalcError):
                    compute_hybrid_stop_from_ohlc(
                        highs=h, lows=l, closes=c, entry_price=100.0,
                        market_cap=5e10, gics_sector=sector)
                h, l, c = _flat_ohlc(bars)
                res = compute_hybrid_stop_from_ohlc(
                    highs=h, lows=l, closes=c, entry_price=100.0,
                    market_cap=5e10, gics_sector=sector)
                self.assertEqual(res["hard_stop"], 97.0, res)


class TestHybridStopPins(unittest.TestCase):

    def test_hard_stop_exactly_zero_is_rejected_as_hard_stop(self):
        # entry == 1.5 x ATR puts the hard stop on 0.0 exactly.
        with self.assertRaises(StopCalcError) as ctx:
            compute_hybrid_stop(entry_price=15.0, atr=10.0, market_cap=5e10,
                                gics_sector="Information Technology")
        self.assertIn("hard_stop <= 0", str(ctx.exception))

    def test_hard_stop_one_cent_above_zero_is_still_judged_on_the_soft_stop(self):
        with self.assertRaises(StopCalcError) as ctx:
            compute_hybrid_stop(entry_price=15.01, atr=10.0, market_cap=5e10,
                                gics_sector="Information Technology")
        self.assertIn("soft_stop <= 0", str(ctx.exception))

    def test_soft_and_hard_stop_collapsing_to_one_float_fails_closed(self):
        # Every soft multiplier exceeds 1.5, so only float rounding can make
        # soft == hard: at 1e16 the ulp is 2, and both 1e16 - 1.5 and
        # 1e16 - 2.25 round to 9999999999999998.0. The strict ordering check
        # must refuse that instead of returning two identical stops.
        self.assertEqual(1e16 - 1.5, 1e16 - 2.25)
        with self.assertRaises(StopCalcError) as ctx:
            compute_hybrid_stop(entry_price=1e16, atr=1.0, market_cap=5e10,
                                gics_sector="Information Technology")
        self.assertIn("stop-ordering invariant violated", str(ctx.exception))

    def test_zero_atr_is_rejected_as_atr(self):
        for atr in (0.0, 0, -0.5):
            with self.subTest(atr=atr), self.assertRaises(StopCalcError) as ctx:
                compute_hybrid_stop(entry_price=100.0, atr=atr, market_cap=5e10)
            self.assertIn("atr must be > 0", str(ctx.exception))

    def test_zero_market_cap_fails_closed(self):
        for cap in (0.0, 0, -1.0, None):
            with self.subTest(cap=cap):
                with self.assertRaises(StopCalcError):
                    get_atr_multiplier(cap, "Industrials")
                with self.assertRaises(StopCalcError):
                    compute_hybrid_stop(entry_price=100.0, atr=2.0,
                                        market_cap=cap, gics_sector="Industrials")

    def test_cap_size_bands_at_literal_ten_and_two_billion(self):
        cases = (
            (10_000_000_001.0, (2.25, "large-cap")),
            (10_000_000_000.0, (2.75, "mid-cap")),
            (10_500_000_000.0, (2.25, "large-cap")),
            (2_000_000_000.0, (2.75, "mid-cap")),
            (1_999_999_999.0, (3.25, "small-cap")),
            (1_500_000_000.0, (3.25, "small-cap")),
        )
        for cap, want in cases:
            with self.subTest(cap=cap):
                self.assertEqual(get_atr_multiplier(cap, "Industrials"), want)


if __name__ == "__main__":
    r = unittest.TextTestRunner(verbosity=1).run(
        unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    sys.exit(0 if r.wasSuccessful() else 1)

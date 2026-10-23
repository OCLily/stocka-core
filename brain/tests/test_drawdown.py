"""Offline tests for the pure level-based drawdown helper.

NO network/IO. Pure function over a [(date, adj_close)] series.
Deskriptives Level-Maß, gap-robust (kein Return über Lücken), fail-closed
(Ausfall sichtbar, nie 0), Lücken sichtbar, Adjustment/Crash-Flag.
"""
from __future__ import annotations

import unittest

from brain.risk.drawdown import max_drawdown_from_closes


def _series(*pairs):
    return list(pairs)


class TestMaxDrawdownFromCloses(unittest.TestCase):
    # core math
    def test_monotone_up_has_zero_drawdown(self):
        s = _series(("2026-01-05", 100.0), ("2026-01-06", 110.0), ("2026-01-07", 120.0))
        r = max_drawdown_from_closes(s)
        self.assertIsNone(r["error"])
        self.assertEqual(r["max_dd_pct"], 0.0)
        self.assertEqual(r["current_dd_pct"], 0.0)

    def test_known_peak_to_trough_exact(self):
        # peak 200 on 01-06, trough 150 on 01-08 → -25%
        s = _series(
            ("2026-01-05", 100.0), ("2026-01-06", 200.0),
            ("2026-01-07", 180.0), ("2026-01-08", 150.0), ("2026-01-09", 160.0),
        )
        r = max_drawdown_from_closes(s)
        self.assertAlmostEqual(r["max_dd_pct"], -25.0, places=6)
        self.assertEqual(r["peak_date"], "2026-01-06")
        self.assertEqual(r["trough_date"], "2026-01-08")

    def test_current_dd_at_last_bar(self):
        # peak 200, last 160 → current dd = -20%
        s = _series(("2026-01-05", 100.0), ("2026-01-06", 200.0), ("2026-01-09", 160.0))
        r = max_drawdown_from_closes(s)
        self.assertAlmostEqual(r["current_dd_pct"], -20.0, places=6)

    # gap NOT at trough → identical max_dd to gapless same-remaining-bars
    def test_gap_not_at_trough_does_not_change_maxdd(self):
        full = _series(
            ("2026-01-05", 100.0), ("2026-01-06", 200.0),
            ("2026-01-07", 180.0), ("2026-01-08", 150.0),
        )
        # 01-07 bar absent (NULL skipped) — trough 150 still present
        gapped = _series(
            ("2026-01-05", 100.0), ("2026-01-06", 200.0), ("2026-01-08", 150.0),
        )
        self.assertAlmostEqual(
            max_drawdown_from_closes(full)["max_dd_pct"],
            max_drawdown_from_closes(gapped)["max_dd_pct"],
            places=6,
        )

    # fail-closed: too few bars
    def test_single_bar_not_computable(self):
        r = max_drawdown_from_closes(_series(("2026-01-05", 100.0)))
        self.assertIsNotNone(r["error"])
        self.assertIn("n=1", r["error"])
        self.assertIsNone(r["max_dd_pct"])

    def test_empty_not_computable(self):
        r = max_drawdown_from_closes(_series())
        self.assertIsNotNone(r["error"])

    # fail-closed: bad data, never silently 0
    def test_nan_close_is_data_error(self):
        s = _series(("2026-01-05", 100.0), ("2026-01-06", float("nan")))
        r = max_drawdown_from_closes(s)
        self.assertIsNotNone(r["error"])
        self.assertIn("Datenfehler", r["error"])

    def test_nonpositive_close_is_data_error(self):
        s = _series(("2026-01-05", 100.0), ("2026-01-06", 0.0))
        r = max_drawdown_from_closes(s)
        self.assertIsNotNone(r["error"])
        self.assertIn("Datenfehler", r["error"])

    # >±50% single-day move flagged (could be split OR real crash)
    def test_large_single_day_move_flagged(self):
        # -60% in one day: 100 -> 40
        s = _series(("2026-01-05", 100.0), ("2026-01-06", 40.0), ("2026-01-07", 42.0))
        r = max_drawdown_from_closes(s)
        self.assertIn("2026-01-06", r["artifact_dates"])

    def test_normal_moves_not_flagged(self):
        s = _series(("2026-01-05", 100.0), ("2026-01-06", 90.0), ("2026-01-07", 95.0))
        r = max_drawdown_from_closes(s)
        self.assertEqual(r["artifact_dates"], [])

    # gap visibility: missing trading days surfaced, same span
    def test_gap_count_from_missing_business_days(self):
        # span Mon 01-05 .. Fri 01-09 = 5 business days; only 3 bars present → 2 gaps
        s = _series(("2026-01-05", 100.0), ("2026-01-07", 101.0), ("2026-01-09", 102.0))
        r = max_drawdown_from_closes(s)
        self.assertEqual(r["n_bars"], 3)
        self.assertEqual(r["expected_business_days"], 5)
        self.assertEqual(r["gap_count"], 2)

    def test_contiguous_business_days_no_gaps(self):
        # Mon..Wed contiguous = 3 business days, 3 bars, weekend not counted
        s = _series(("2026-01-05", 100.0), ("2026-01-06", 101.0), ("2026-01-07", 102.0))
        r = max_drawdown_from_closes(s)
        self.assertEqual(r["gap_count"], 0)

    def test_weekend_not_counted_as_gap(self):
        # Fri 01-09 -> Mon 01-12 is contiguous trading; no gap
        s = _series(("2026-01-09", 100.0), ("2026-01-12", 101.0))
        r = max_drawdown_from_closes(s)
        self.assertEqual(r["gap_count"], 0)

    # real trading calendar instead of a Mon-Fri approximation: without subtracting
    # holidays, gap_count would be structurally >0 for a fixed 365d window even on
    # COMPLETE data. Deliberately a real calendar, not a tolerance threshold.
    def test_holiday_week_no_gap_with_complete_bars(self):
        # Thanksgiving week 2025: Mon 11-24 .. Fri 11-28; Thu 11-27 is a holiday
        # (no bar) — with complete real trading data gap_count must be 0, not 1.
        s = _series(
            ("2025-11-24", 100.0), ("2025-11-25", 101.0),
            ("2025-11-26", 102.0), ("2025-11-28", 103.0),
        )
        r = max_drawdown_from_closes(s)
        self.assertEqual(r["n_bars"], 4)
        self.assertEqual(r["expected_business_days"], 4)
        self.assertEqual(r["gap_count"], 0)

    def test_real_hole_still_counted_alongside_holiday(self):
        # Same Thanksgiving week, but Wed 11-26 is ALSO missing (real data hole,
        # not a holiday) — must surface as gap_count=1, not be absorbed silently.
        s = _series(
            ("2025-11-24", 100.0), ("2025-11-25", 101.0), ("2025-11-28", 103.0),
        )
        r = max_drawdown_from_closes(s)
        self.assertEqual(r["n_bars"], 3)
        self.assertEqual(r["expected_business_days"], 4)
        self.assertEqual(r["gap_count"], 1)

    def test_year_boundary_new_year_saturday_has_no_observed_holiday(self):
        # 2022-01-01 fell on a Saturday. NYSE Rule 7.2 New Year's Day exception:
        # unlike every other holiday, a Saturday New Year's Day is NOT observed
        # on the preceding Friday -- the exchange trades normally on 2021-12-31
        # (confirmed real NYSE trading day, like 2010-12-31/2004-12-31/1999-12-31
        # in all other Sat-Jan-1 years). No substitute holiday falls in this
        # window at all.
        from brain.risk.drawdown import _trading_days_inclusive
        # Wed 2021-12-29 .. Tue 2022-01-04: real trading days are Wed/Thu/Fri
        # (12-29, 12-30, 12-31) and Mon/Tue (01-03, 01-04); only the weekend
        # (01-01/01-02) is excluded -> 5 trading days, not 4.
        self.assertEqual(_trading_days_inclusive("2021-12-29", "2022-01-04"), 5)

    def test_year_boundary_new_year_sunday_observed_following_monday(self):
        # 2023-01-01 fell on a Sunday -> NYSE observed New Year's Day on the
        # following Monday, 2023-01-02 (standard So->Mo rule, unaffected by the
        # Saturday-only exception). Verified real NYSE closure date.
        from brain.risk.drawdown import _trading_days_inclusive
        # Fri 2022-12-30 .. Wed 2023-01-04: real trading days are Fri (12-30),
        # Tue/Wed (01-03, 01-04); the weekend (12-31/01-01) and the observed
        # Monday holiday (01-02) are excluded -> 3 trading days.
        self.assertEqual(_trading_days_inclusive("2022-12-30", "2023-01-04"), 3)

    def test_known_nyse_holidays_2024(self):
        from datetime import date

        from brain.risk.drawdown import _nyse_holidays_for_year
        self.assertEqual(_nyse_holidays_for_year(2024), {
            date(2024, 1, 1), date(2024, 1, 15), date(2024, 2, 19), date(2024, 3, 29),
            date(2024, 5, 27), date(2024, 6, 19), date(2024, 7, 4), date(2024, 9, 2),
            date(2024, 11, 28), date(2024, 12, 25),
        })

    def test_juneteenth_not_observed_before_2022(self):
        from datetime import date

        from brain.risk.drawdown import _nyse_holidays_for_year
        self.assertNotIn(date(2021, 6, 19), _nyse_holidays_for_year(2021))
        # 2022-06-19 fell on a Sunday -> observed Monday 2022-06-20.
        self.assertIn(date(2022, 6, 20), _nyse_holidays_for_year(2022))


def _suite() -> unittest.TestSuite:
    return unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))


if __name__ == "__main__":
    unittest.main()

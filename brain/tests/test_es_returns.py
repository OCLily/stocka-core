"""Offline tests for the pure bounded-returns helper.

NO network/IO. Builds a daily-return series from [(date, adj_close)] with two guards:
  - Bad-Bar (NaN/non-positive adj) BREAKS the chain → no bridge-phantom.
  - Return only when <=1 missing business day between trade_dates (Fri→Mo incl,
    holiday incl, multi-day gap excluded + counted).
"""
from __future__ import annotations

import unittest

from brain.risk.es_returns import bounded_returns


def _s(*pairs):
    return list(pairs)


class TestBoundedReturns(unittest.TestCase):
    # 2026-01-05 Mon, 06 Tue, 07 Wed, 08 Thu, 09 Fri, 12 Mon

    def test_contiguous_all_returns(self):
        r = bounded_returns(_s(("2026-01-05", 100.0), ("2026-01-06", 110.0), ("2026-01-07", 121.0)))
        self.assertEqual(len(r["returns"]), 2)
        self.assertAlmostEqual(r["returns"][0], 0.10, places=6)
        self.assertEqual(r["n_excluded"], 0)

    # the most important guard: Fri→Mon (miss=0) MUST be included, the weekend
    # days are not business days
    def test_weekend_friday_to_monday_included(self):
        r = bounded_returns(_s(("2026-01-09", 100.0), ("2026-01-12", 102.0)))
        self.assertEqual(len(r["returns"]), 1)
        self.assertEqual(r["n_excluded"], 0)

    # holiday (1 missing business day, Wed→Fri over Thu) included
    def test_single_holiday_gap_included(self):
        r = bounded_returns(_s(("2026-01-07", 100.0), ("2026-01-09", 101.0)))
        self.assertEqual(len(r["returns"]), 1)
        self.assertEqual(r["n_excluded"], 0)

    # multi-day gap (>=2 missing business days, Mon→Thu) excluded + counted
    def test_multiday_gap_excluded(self):
        r = bounded_returns(_s(("2026-01-05", 100.0), ("2026-01-08", 90.0)))
        self.assertEqual(r["returns"], [])
        self.assertEqual(r["n_excluded"], 1)

    # The two chain-break tests below pin n_excluded == 0 deliberately (gaps are
    # counted, bad bars only break the chain): n_excluded means "Returns über
    # Mehrtages-Lücken ausgeschlossen", not "bad bars seen".
    # Known consequence: risk_profile.py feeds its "lückige Serie"-Qualifier solely
    # from n_excluded, so a NaN-riddled series is annualised without that
    # qualifier — bad-bar holes stay visible only through the lowered n. Widening
    # this (e.g. a separate n_bad_bars in the qualifier) changes the readout and
    # is deliberately left open; do not "fix" it by relaxing these assertions.

    # NaN bar breaks chain, no Mon->Wed bridge-phantom
    def test_nan_bar_breaks_chain(self):
        r = bounded_returns(_s(("2026-01-05", 100.0), ("2026-01-06", float("nan")), ("2026-01-07", 110.0)))
        self.assertEqual(r["returns"], [])
        self.assertEqual(r["n_excluded"], 0)

    # non-positive bar breaks chain like NaN
    def test_nonpositive_bar_breaks_chain(self):
        r = bounded_returns(_s(("2026-01-05", 100.0), ("2026-01-06", 0.0), ("2026-01-07", 110.0)))
        self.assertEqual(r["returns"], [])
        self.assertEqual(r["n_excluded"], 0)

    def test_chain_resumes_after_bad_bar(self):
        # bad bar in the middle, two good bars after → one return at the end
        r = bounded_returns(_s(
            ("2026-01-05", 100.0), ("2026-01-06", float("nan")),
            ("2026-01-07", 110.0), ("2026-01-08", 121.0)))
        self.assertEqual(len(r["returns"]), 1)
        self.assertAlmostEqual(r["returns"][0], 0.10, places=6)


def _suite() -> unittest.TestSuite:
    return unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))


if __name__ == "__main__":
    unittest.main()

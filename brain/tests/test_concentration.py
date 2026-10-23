"""Tests for the descriptive universe-concentration metric (brain/analytics/concentration.py).

The metric is DESCRIPTIVE ONLY — a measured property of the universe, never a
signal, forecast, or timing trigger. Coverage (priced vs un-priced members) must
stay visible so the number never pretends completeness it lacks.

The pure `concentration_metrics` unit is DB-free and fully covered here; the thin
read-layer (`load_active_market_caps`, `universe_concentration`) is exercised against
an in-memory sqlite mirror of universe_membership — no real DB, offline, like CI.

Run: python3 -m brain.tests.test_concentration
"""
from __future__ import annotations

import sqlite3
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from brain.analytics.concentration import (  # noqa: E402
    concentration_metrics,
    load_active_market_caps,
    universe_concentration,
    format_concentration_lines,
)


class PureMetrics(unittest.TestCase):
    """concentration_metrics — pure, DB-free HHI / effective-stocks / top-N weight."""

    def test_equal_weights_effective_equals_n(self):
        m = concentration_metrics([100.0, 100.0, 100.0, 100.0])
        self.assertEqual(m["n_priced"], 4)
        self.assertAlmostEqual(m["hhi"], 0.25)
        self.assertAlmostEqual(m["effective_stocks"], 4.0)
        self.assertAlmostEqual(m["top_n_weight"], 1.0)  # top_n=10 caps at 4 priced

    def test_single_stock_is_maximally_concentrated(self):
        m = concentration_metrics([500.0])
        self.assertEqual(m["n_priced"], 1)
        self.assertAlmostEqual(m["hhi"], 1.0)
        self.assertAlmostEqual(m["effective_stocks"], 1.0)
        self.assertAlmostEqual(m["top_n_weight"], 1.0)
        self.assertEqual(m["top_n"], 1)

    def test_effective_stocks_is_inverse_hhi(self):
        m = concentration_metrics([900.0, 50.0, 30.0, 20.0])
        self.assertAlmostEqual(m["effective_stocks"], 1.0 / m["hhi"])

    def test_top_n_weight_captures_dominance(self):
        caps = [1000.0] + [1.0] * 20  # one giant, twenty tiny
        m = concentration_metrics(caps, top_n=10)
        # top-10 = 1000 + 9*1 = 1009 out of total 1020
        self.assertAlmostEqual(m["top_n_weight"], 1009.0 / 1020.0)
        self.assertEqual(m["top_n"], 10)

    def test_coverage_missing_is_filtered_and_counted(self):
        m = concentration_metrics([100.0, None, 200.0, 0.0, -5.0])
        self.assertEqual(m["n_input"], 5)
        self.assertEqual(m["n_priced"], 2)          # only 100 and 200 are priced
        self.assertEqual(m["n_missing"], 3)         # None, 0.0, -5.0 excluded
        self.assertAlmostEqual(m["hhi"], 5.0 / 9.0)  # (1/3)^2 + (2/3)^2
        self.assertAlmostEqual(m["total_market_cap_usd"], 300.0)

    def test_empty_input_yields_no_metric(self):
        m = concentration_metrics([])
        self.assertEqual(m["n_input"], 0)
        self.assertEqual(m["n_priced"], 0)
        self.assertIsNone(m["hhi"])
        self.assertIsNone(m["effective_stocks"])
        self.assertIsNone(m["top_n_weight"])
        self.assertIsNone(m["total_market_cap_usd"])

    def test_all_unpriced_yields_no_metric_no_zerodiv(self):
        m = concentration_metrics([None, 0.0, -1.0])
        self.assertEqual(m["n_input"], 3)
        self.assertEqual(m["n_priced"], 0)
        self.assertIsNone(m["hhi"])
        self.assertIsNone(m["effective_stocks"])

    def test_top_n_caps_at_priced_count(self):
        m = concentration_metrics([10.0, 20.0, 30.0], top_n=10)
        self.assertEqual(m["top_n"], 3)
        self.assertAlmostEqual(m["top_n_weight"], 1.0)

    def test_nonpositive_top_n_yields_zero_weight(self):
        # a non-positive top_n means "sum the top zero" — never a negative slice
        # that would silently sum all-but-the-smallest weight.
        for tn in (0, -1, -5):
            m = concentration_metrics([10.0, 20.0, 30.0], top_n=tn)
            self.assertEqual(m["top_n"], 0)
            self.assertAlmostEqual(m["top_n_weight"], 0.0)

    def test_hhi_and_effective_stay_in_bounds(self):
        m = concentration_metrics([5.0, 4.0, 3.0, 2.0, 1.0])
        self.assertGreater(m["hhi"], 0.0)
        self.assertLessEqual(m["hhi"], 1.0)
        self.assertGreaterEqual(m["effective_stocks"], 1.0)
        self.assertLessEqual(m["effective_stocks"], 5.0)


class ReadLayer(unittest.TestCase):
    """load_active_market_caps / universe_concentration against an in-memory mirror.

    conn is caller-provided — the module never opens its own connection."""

    def _conn(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            "CREATE TABLE universe_membership "
            "(universe_id TEXT, ticker TEXT, removed_at TEXT, market_cap_usd REAL)"
        )
        conn.executemany(
            "INSERT INTO universe_membership VALUES (?,?,?,?)",
            [
                ("u1", "AAA", None, 100.0),
                ("u1", "BBB", None, None),          # active but un-priced
                ("u1", "CCC", None, 300.0),
                ("u1", "DDD", "2026-01-01", 999.0),  # removed -> excluded
                ("u2", "EEE", None, 50.0),           # other universe
            ],
        )
        return conn

    def test_load_returns_only_active_members(self):
        caps = load_active_market_caps(self._conn(), "u1")
        self.assertEqual(len(caps), 3)          # AAA, BBB, CCC (DDD removed)
        self.assertIn(None, caps)               # un-priced member preserved
        self.assertNotIn(999.0, caps)           # removed member absent

    def test_universe_concentration_reports_coverage(self):
        r = universe_concentration(self._conn(), "u1")
        self.assertEqual(r["universe_id"], "u1")
        self.assertEqual(r["n_input"], 3)
        self.assertEqual(r["n_priced"], 2)
        self.assertEqual(r["n_missing"], 1)
        # priced u1 = {100, 300}: weights 0.25/0.75 -> HHI = 0.0625 + 0.5625
        self.assertAlmostEqual(r["hhi"], 0.625)
        self.assertAlmostEqual(r["effective_stocks"], 1.6)

    def test_isolates_universes(self):
        r = universe_concentration(self._conn(), "u2")
        self.assertEqual(r["n_input"], 1)
        self.assertEqual(r["n_priced"], 1)
        self.assertAlmostEqual(r["effective_stocks"], 1.0)


class Formatter(unittest.TestCase):
    """format_concentration_lines — descriptive text, coverage-honest, never a signal."""

    def _result(self):
        return {
            "universe_id": "s&p_500", "n_input": 503, "n_priced": 498,
            "n_missing": 5, "hhi": 0.0217, "effective_stocks": 46.1,
            "top_n": 10, "top_n_weight": 0.401, "total_market_cap_usd": 5.5e13,
        }

    def test_lines_carry_descriptive_disclaimer(self):
        text = "\n".join(format_concentration_lines(self._result()))
        # the surface must frame this as descriptive, never a signal.
        self.assertIn("deskriptiv", text.lower())
        self.assertIn("kein Signal", text)

    def test_lines_show_universe_and_coverage(self):
        text = "\n".join(format_concentration_lines(self._result()))
        self.assertIn("s&p_500", text)
        self.assertIn("498", text)   # priced count visible
        self.assertIn("503", text)   # total members visible

    def test_numbers_render_with_german_comma(self):
        # pins the user-facing number formatting (German decimal comma).
        text = "\n".join(format_concentration_lines(self._result()))
        self.assertIn("46,1", text)     # effective_stocks: comma + one decimal
        self.assertIn("0,0217", text)   # HHI: four decimals, comma
        self.assertIn("40,1%", text)    # top_n_weight 0.401 -> 40,1%
        self.assertNotIn("46.1", text)  # fails if the decimal-comma replace is dropped

    def test_no_metric_renders_cleanly(self):
        empty = {
            "universe_id": "x", "n_input": 0, "n_priced": 0, "n_missing": 0,
            "hhi": None, "effective_stocks": None, "top_n": 0,
            "top_n_weight": None, "total_market_cap_usd": None,
        }
        text = "\n".join(format_concentration_lines(empty)).lower()
        self.assertIn("keine", text)  # "keine gepreisten Titel" — no crash on None


def suite():
    return unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])


if __name__ == "__main__":
    r = unittest.TextTestRunner(verbosity=2).run(suite())
    sys.exit(0 if r.wasSuccessful() else 1)

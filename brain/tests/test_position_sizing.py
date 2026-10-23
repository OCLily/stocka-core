"""Sizing limits S1-S3 (brain/risk/position_sizing.py) — pure-function pins.

The arithmetic core only, no wiring. What these tests hold in place:

  * the three limits and their exact boundaries (<= is PASS),
  * a counter-example (60.00 / 57.00 -> S1 alone lets 40 % of the book
    through, S2 must stop it),
  * fail-closed on missing or invalid inputs — above all no FX means FAIL,
    never an implicit 1:1 rate — with `bad_input` naming the culprit so a
    caller can tell an infrastructure gap from a rule breach,
  * the S3 "empty book" setting (no OTHER open position at booking time),
  * the module stays free of I/O.

Run: python3 -m brain.tests.test_position_sizing
"""
from __future__ import annotations

import ast
import math
import sys
import unittest
from datetime import date
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from brain.risk import position_sizing as ps  # noqa: E402
from brain.risk.position_sizing import (  # noqa: E402
    FAIL, NOT_APPLICABLE, PASS, WARN, book_is_empty, check_sizing,
)

_FX_REAL = 0.8684  # realistic non-round EUR-per-USD rate; provokes float noise


def _run(**overrides):
    kwargs = dict(entry_price_usd=100.0, stop_usd=95.0, shares=25.0,
                  fx_usd_eur=1.0, capital_eur=100_000.0, book_empty=False)
    kwargs.update(overrides)
    return {r.rule_id: r for r in check_sizing(**kwargs)}


class TestLimitsAndSources(unittest.TestCase):

    def test_limit_constants(self):
        self.assertEqual(ps.RISK_PER_POSITION_MAX, 0.02)
        self.assertEqual(ps.NOTIONAL_PER_POSITION_MAX, 0.10)
        self.assertEqual(ps.FIRST_POSITION_MAX, 0.03)
        self.assertEqual(ps.FIRST_POSITION_MIN_WARN, 0.02)


    def test_always_three_results_in_fixed_order(self):
        for overrides in ({}, {"fx_usd_eur": None}, {"stop_usd": None},
                          {"book_empty": True}, {"book_empty": None}):
            with self.subTest(overrides=overrides):
                results = check_sizing(**{**dict(
                    entry_price_usd=100.0, stop_usd=95.0, shares=25.0,
                    fx_usd_eur=1.0, capital_eur=100_000.0, book_empty=False),
                    **overrides})
                self.assertEqual([r.rule_id for r in results], ["S1", "S2", "S3"])

    def test_only_fail_blocks(self):
        for status, blocks in ((PASS, False), (WARN, False),
                               (NOT_APPLICABLE, False), (FAIL, True)):
            self.assertIs(ps.RuleResult("S1", status, "x").blocks, blocks)


class TestS1Risk(unittest.TestCase):

    def test_exactly_two_percent_passes(self):
        r = _run(entry_price_usd=100.0, stop_usd=90.0, shares=200.0)["S1"]
        self.assertEqual(r.status, PASS, r.detail)

    def test_exactly_two_percent_with_real_fx_float_noise_passes(self):
        risk_eur = (33.33 - 31.07) * 23.0 * _FX_REAL
        capital = risk_eur / 0.02
        self.assertGreater(risk_eur / capital, 0.02, "case no longer float-noisy")
        r = _run(entry_price_usd=33.33, stop_usd=31.07, shares=23.0,
                 fx_usd_eur=_FX_REAL, capital_eur=capital)["S1"]
        self.assertEqual(r.status, PASS, r.detail)

    def test_one_cent_over_two_percent_fails(self):
        r = _run(entry_price_usd=100.0, stop_usd=90.0, shares=200.0,
                 capital_eur=(2000.0 - 0.01) / 0.02)["S1"]
        self.assertEqual(r.status, FAIL, r.detail)
        self.assertTrue(r.blocks)
        self.assertEqual(r.bad_input, ())

    def test_risk_is_converted_with_fx(self):
        # 2 500 USD risk: FAIL at 1:1, PASS once converted at 0.78 EUR/USD.
        base = dict(entry_price_usd=100.0, stop_usd=90.0, shares=250.0)
        self.assertEqual(_run(**base, fx_usd_eur=1.0)["S1"].status, FAIL)
        self.assertEqual(_run(**base, fx_usd_eur=0.78)["S1"].status, PASS)

    def test_invalid_stop_fails_s1_only(self):
        for stop in (None, 0.0, -1.0, 100.0, 101.0, math.nan, "95"):
            with self.subTest(stop=stop):
                res = _run(stop_usd=stop)
                self.assertEqual(res["S1"].status, FAIL)
                self.assertEqual(res["S1"].bad_input, ("stop_usd",))
                self.assertEqual(res["S2"].status, PASS)


class TestS2Notional(unittest.TestCase):

    def test_exactly_ten_percent_passes(self):
        r = _run(entry_price_usd=100.0, stop_usd=99.0, shares=100.0)["S2"]
        self.assertEqual(r.status, PASS, r.detail)

    def test_exactly_ten_percent_with_real_fx_float_noise_passes(self):
        notional_eur = 3.0 * 100.1 * _FX_REAL
        capital = notional_eur / 0.10
        self.assertGreater(notional_eur / capital, 0.10, "case no longer float-noisy")
        r = _run(entry_price_usd=100.1, stop_usd=97.3, shares=3.0,
                 fx_usd_eur=_FX_REAL, capital_eur=capital)["S2"]
        self.assertEqual(r.status, PASS, r.detail)

    def test_one_cent_over_ten_percent_fails(self):
        r = _run(entry_price_usd=100.0, stop_usd=99.0, shares=100.0,
                 capital_eur=(10_000.0 - 0.01) / 0.10)["S2"]
        self.assertEqual(r.status, FAIL, r.detail)

    def test_counter_example_s1_alone_would_allow_40_percent(self):
        # entry 60.00, stop 57.00: the 2 % risk rule alone allows
        # 666 shares = 40 % of the book.
        res = _run(entry_price_usd=60.00, stop_usd=57.00, shares=666.0)
        self.assertEqual(res["S1"].status, PASS, res["S1"].detail)
        self.assertEqual(res["S2"].status, FAIL, res["S2"].detail)
        self.assertAlmostEqual(res["S2"].measured, 0.3996, places=4)

    def test_counter_example_holds_under_real_fx_too(self):
        per_share_risk_eur = (60.00 - 57.00) * _FX_REAL
        shares = math.floor(2000.0 / per_share_risk_eur)
        res = _run(entry_price_usd=60.00, stop_usd=57.00, shares=float(shares),
                   fx_usd_eur=_FX_REAL)
        self.assertEqual(res["S1"].status, PASS)
        self.assertEqual(res["S2"].status, FAIL)


class TestS3FirstPosition(unittest.TestCase):

    def _s3(self, pct, **overrides):
        return _run(entry_price_usd=100.0, stop_usd=99.0,
                    shares=pct * 10.0, book_empty=True, **overrides)["S3"]

    def test_band_on_an_empty_book(self):
        cases = ((3.1, FAIL), (3.0, PASS), (2.5, PASS), (2.0, PASS), (1.5, WARN))
        for pct, expected in cases:
            with self.subTest(pct=pct):
                r = self._s3(pct)
                self.assertEqual(r.status, expected, r.detail)


    def test_below_band_is_a_hint_not_a_block(self):
        r = self._s3(1.5)
        self.assertFalse(r.blocks)
        self.assertIn("kein FAIL", r.detail)

    def test_s3_is_tighter_than_s2_on_an_empty_book(self):
        res = _run(entry_price_usd=100.0, stop_usd=99.0, shares=80.0,
                   book_empty=True)
        self.assertEqual(res["S2"].status, PASS)
        self.assertEqual(res["S3"].status, FAIL)

    def test_not_applicable_when_book_holds_another_position(self):
        res = _run(entry_price_usd=100.0, stop_usd=99.0, shares=80.0,
                   book_empty=False)
        self.assertEqual(res["S3"].status, NOT_APPLICABLE)
        self.assertFalse(res["S3"].blocks)

    def test_non_bool_book_state_fails_closed(self):
        for value in (None, 0, 1, "yes"):
            with self.subTest(value=value):
                r = _run(book_empty=value)["S3"]
                self.assertEqual(r.status, FAIL)
                self.assertEqual(r.bad_input, ("book_empty",))


class TestFailClosedInputs(unittest.TestCase):

    def test_missing_fx_fails_every_rule_never_one_to_one(self):
        # These numbers pass all three rules at 1:1 — a missing rate must not.
        for fx in (None, 0.0, -0.9, math.nan, math.inf):
            with self.subTest(fx=fx):
                res = _run(entry_price_usd=100.0, stop_usd=95.0, shares=25.0,
                           book_empty=True, fx_usd_eur=fx)
                for r in res.values():
                    self.assertEqual(r.status, FAIL, r.detail)
                    self.assertEqual(r.bad_input, ("fx_usd_eur",))
        self.assertEqual({r.status for r in _run(book_empty=True).values()}, {PASS})

    def test_invalid_core_inputs_fail_every_rule(self):
        for name in ("entry_price_usd", "shares", "capital_eur"):
            for value in (None, 0, -5.0, math.nan, True, "100"):
                with self.subTest(name=name, value=value):
                    res = _run(**{name: value})
                    for r in res.values():
                        self.assertEqual(r.status, FAIL)
                        self.assertIn(name, r.bad_input)

    def test_bad_input_lists_every_missing_input(self):
        res = _run(fx_usd_eur=None, capital_eur=None)
        self.assertEqual(res["S1"].bad_input, ("fx_usd_eur", "capital_eur"))


class TestBookIsEmpty(unittest.TestCase):

    def test_no_positions(self):
        self.assertTrue(book_is_empty({}, ticker="XYZ.US", entry_date="2025-03-17"))

    def test_closed_positions_do_not_count(self):
        positions = {"ABC.US": {"status": "closed", "entry_date": "2026-01-15"}}
        self.assertTrue(book_is_empty(positions, ticker="XYZ.US",
                                      entry_date="2025-03-17"))

    def test_another_open_position_means_not_empty(self):
        positions = {"ABC.US": {"status": "open", "entry_date": "2026-01-15"}}
        self.assertFalse(book_is_empty(positions, ticker="XYZ.US",
                                       entry_date="2025-03-17"))

    def test_the_position_under_review_is_excluded(self):
        positions = {"XYZ.US": {"status": "open", "entry_date": "2025-03-17"}}
        self.assertTrue(book_is_empty(positions, ticker="XYZ.US",
                                      entry_date=date(2025, 3, 17)))

    def test_same_ticker_other_entry_date_is_another_position(self):
        positions = {"XYZ.US": {"status": "open", "entry_date": "2025-03-03"}}
        self.assertFalse(book_is_empty(positions, ticker="XYZ.US",
                                       entry_date="2025-03-17"))

    def test_candidate_identity_is_mandatory(self):
        for ticker, entry_date in (("", "2025-03-17"), ("XYZ.US", ""),
                                   ("XYZ.US", None)):
            with self.subTest(ticker=ticker, entry_date=entry_date):
                with self.assertRaises(ValueError):
                    book_is_empty({}, ticker=ticker, entry_date=entry_date)


class TestPurity(unittest.TestCase):

    def test_module_imports_stdlib_only(self):
        tree = ast.parse(Path(ps.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        self.assertLessEqual(imported, {"__future__", "math", "dataclasses", "typing"})

    def test_no_file_or_network_calls(self):
        src = Path(ps.__file__).read_text(encoding="utf-8")
        for token in ("open(", "Path(", "load_state", "requests", "urlopen"):
            self.assertNotIn(token, src)


if __name__ == "__main__":
    r = unittest.TextTestRunner(verbosity=1).run(
        unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    sys.exit(0 if r.wasSuccessful() else 1)

"""Sloan universe snapshot and the snapshot-boundary quintile lookup."""

import unittest
from datetime import date

from brain.scoring.sloan import compute_sloan_accruals
from brain.scoring.sloan_universe import compute_sloan_universe_snapshot, get_sloan_quintile_for_score

META_KEYS = {"quintiles", "snapshot_date"}


def two_years(accruals_ratio, sector="Industrials"):
    # Only current assets move: accruals = delta CA - depreciation (50) and
    # average total assets = 1000, so delta CA = 1000 * ratio + 50.
    def balance(current_assets):
        return {
            "totalCurrentAssets": str(current_assets), "totalCurrentLiabilities": "150",
            "cash": "100", "shortTermInvestments": "0", "shortTermDebt": "30", "totalAssets": "1000",
        }
    return {
        "General": {"GicSector": sector},
        "Financials": {
            "Balance_Sheet": {"yearly": {
                "2024-12-31": balance(400 + 1000 * accruals_ratio + 50),
                "2023-12-31": balance(400),
            }},
            "Income_Statement": {"yearly": {
                "2024-12-31": {"depreciationAndAmortization": "50"},
                "2023-12-31": {"depreciationAndAmortization": "50"},
            }},
        },
    }


def ratios(snapshot):
    return {k: v for k, v in snapshot.items() if k not in META_KEYS}


class SnapshotTests(unittest.TestCase):

    def test_fixture_produces_the_intended_ratio(self):
        self.assertAlmostEqual(compute_sloan_accruals(two_years(0.03))["accruals_ratio"], 0.03, places=6)

    def test_one_ratio_per_non_financial_ticker_plus_metadata(self):
        self.assertTrue(compute_sloan_accruals(two_years(0.5, sector="Financials"))["complete"])
        snapshot = compute_sloan_universe_snapshot({
            "AAA": two_years(-0.02), "BBB": two_years(0.04), "BANK": two_years(0.5, sector="Financials"),
        })
        self.assertEqual(set(snapshot), {"AAA", "BBB"} | META_KEYS)
        self.assertAlmostEqual(snapshot["AAA"], -0.02, places=6)
        self.assertAlmostEqual(snapshot["BBB"], 0.04, places=6)

    def test_tickers_with_incomplete_data_are_skipped(self):
        incomplete = two_years(0.01)
        del incomplete["Financials"]["Income_Statement"]
        self.assertFalse(compute_sloan_accruals(incomplete)["complete"])
        snapshot = compute_sloan_universe_snapshot({"OK": two_years(0.01), "GAP": incomplete, "NONE": None})
        self.assertEqual(set(ratios(snapshot)), {"OK"})

    def test_ticker_without_sector_is_included(self):
        fundamentals = two_years(0.01)
        del fundamentals["General"]
        self.assertIn("NOSECTOR", compute_sloan_universe_snapshot({"NOSECTOR": fundamentals}))

    def test_quintiles_need_at_least_five_tickers(self):
        four = compute_sloan_universe_snapshot({f"T{i}": two_years(0.01 * i) for i in range(4)})
        self.assertEqual(len(ratios(four)), 4)
        self.assertIsNone(four["quintiles"])
        five = compute_sloan_universe_snapshot({f"T{i}": two_years(0.01 * i) for i in range(5)})
        self.assertEqual(len(ratios(five)), 5)
        self.assertIsNotNone(five["quintiles"])

    def test_quintile_boundaries_are_order_statistics(self):
        # n = 10 sorted ratios 0.00 .. 0.09; boundary q sits at index
        # min(int(10 * q), 9) for q = 0.2, 0.4, 0.6, 0.8, 1.0 -> 2, 4, 6, 8, 9
        universe = {f"T{i}": two_years(0.01 * i) for i in range(10)}
        quintiles = compute_sloan_universe_snapshot(universe)["quintiles"]
        for got, expected in zip(quintiles, (0.02, 0.04, 0.06, 0.08, 0.09)):
            self.assertAlmostEqual(got, expected, places=6)

    def test_financials_do_not_count_towards_the_minimum(self):
        universe = {f"T{i}": two_years(0.01 * i) for i in range(4)}
        universe["BANK"] = two_years(0.1, sector="Financials")
        self.assertTrue(compute_sloan_accruals(universe["BANK"])["complete"])
        self.assertIsNone(compute_sloan_universe_snapshot(universe)["quintiles"])

    def test_empty_input_gives_only_metadata(self):
        for data in ({}, None):
            with self.subTest(data=data):
                snapshot = compute_sloan_universe_snapshot(data)
                self.assertEqual(set(snapshot), META_KEYS)
                self.assertIsNone(snapshot["quintiles"])

    def test_snapshot_date_is_an_iso_date(self):
        snapshot_date = compute_sloan_universe_snapshot({})["snapshot_date"]
        self.assertIsInstance(date.fromisoformat(snapshot_date), date)


class QuintileLookupTests(unittest.TestCase):

    BOUNDARIES = [0.02, 0.04, 0.06, 0.08, 0.09]

    def test_boundaries_are_inclusive(self):
        for score, expected in ((0.02, 1), (0.03, 2), (0.04, 2), (0.06, 3), (0.08, 4), (0.09, 5)):
            with self.subTest(score=score):
                self.assertEqual(get_sloan_quintile_for_score(score, self.BOUNDARIES), expected)

    def test_values_beyond_the_last_boundary_are_q5(self):
        self.assertEqual(get_sloan_quintile_for_score(0.5, self.BOUNDARIES), 5)

    def test_values_below_the_first_boundary_are_q1(self):
        self.assertEqual(get_sloan_quintile_for_score(-1.0, self.BOUNDARIES), 1)

    def test_missing_score_or_boundaries_give_none(self):
        self.assertIsNone(get_sloan_quintile_for_score(None, self.BOUNDARIES))
        self.assertIsNone(get_sloan_quintile_for_score(0.01, None))
        self.assertIsNone(get_sloan_quintile_for_score(0.01, []))

    def test_works_on_a_snapshot_it_produced(self):
        universe = {f"T{i}": two_years(0.01 * i) for i in range(10)}
        quintiles = compute_sloan_universe_snapshot(universe)["quintiles"]
        self.assertEqual(get_sloan_quintile_for_score(0.0, quintiles), 1)
        self.assertEqual(get_sloan_quintile_for_score(0.085, quintiles), 5)


if __name__ == "__main__":
    unittest.main()

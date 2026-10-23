"""compute_cvar_bundle: naive and volatility-adjusted VaR/CVaR from daily closes."""

import unittest

from brain.risk.risk_metrics import (
    HARD_CAP_CONFIDENCE,
    MIN_HISTORY_DAYS,
    PRIMARY_CONFIDENCE,
    _ewma_vol,
    _simple_returns,
    _vol_adjusted_returns,
    compute_cvar_bundle,
)

# Three losses and 37 small gains, so every tail can be computed by hand.
TAIL_CHANGES = [-0.05, -0.04, -0.03] + [0.01] * 37


def closes_from(changes, start=100.0):
    closes = [start]
    for change in changes:
        closes.append(closes[-1] * (1 + change))
    return closes


def alternating(size, n):
    return [size if i % 2 else -size for i in range(n)]


CALM_THEN_TURBULENT = alternating(0.002, 300) + alternating(0.03, 30)
TURBULENT_THEN_CALM = alternating(0.03, 150) + alternating(0.002, 300)


def scale_factors(changes):
    returns = _simple_returns(closes_from(changes))
    adjusted = _vol_adjusted_returns(returns, 252, 21)
    return [a / r for a, r in zip(adjusted, returns)]


class SimpleReturnsTests(unittest.TestCase):

    def test_close_to_close_changes(self):
        out = _simple_returns([100.0, 110.0, 99.0])
        self.assertAlmostEqual(out[0], 0.10)
        self.assertAlmostEqual(out[1], -0.10)

    def test_fewer_than_two_closes_give_nothing(self):
        self.assertEqual(_simple_returns([]), [])
        self.assertEqual(_simple_returns([100.0]), [])


class NaiveTailTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.bundle = compute_cvar_bundle(closes_from(TAIL_CHANGES))

    def test_tail_values_per_confidence(self):
        # n = 40 changes. k = floor(40 * (1 - c)) + 1:
        #   95 %:  k = 3 -> VaR 3 %, CVaR mean(5, 4, 3) = 4 %
        #   97.5 %: k = 2 -> VaR 4 %, CVaR mean(5, 4) = 4.5 %
        #   99 %:  k = 1 -> VaR 5 %, CVaR 5 %
        expected = {
            "var_naive_950": 3.0, "cvar_naive_950": 4.0,
            "var_naive_975": 4.0, "cvar_naive_975": 4.5,
            "var_naive_990": 5.0, "cvar_naive_990": 5.0,
        }
        for key, value in expected.items():
            with self.subTest(key=key):
                self.assertAlmostEqual(self.bundle[key], value, places=6)

    def test_cvar_is_never_below_var(self):
        for suffix in ("950", "975", "990"):
            for kind in ("naive", "adj"):
                with self.subTest(suffix=suffix, kind=kind):
                    self.assertGreaterEqual(self.bundle[f"cvar_{kind}_{suffix}"], self.bundle[f"var_{kind}_{suffix}"])

    def test_metadata(self):
        self.assertEqual(self.bundle["history_days"], 40)
        self.assertEqual(self.bundle["primary_confidence"], PRIMARY_CONFIDENCE)
        self.assertEqual(self.bundle["hard_cap_confidence"], HARD_CAP_CONFIDENCE)
        self.assertEqual((PRIMARY_CONFIDENCE, HARD_CAP_CONFIDENCE), (0.975, 0.99))


class HistoryLengthTests(unittest.TestCase):

    def test_fewer_than_30_closes_give_an_all_none_bundle(self):
        bundle = compute_cvar_bundle(closes_from(TAIL_CHANGES[:28]))
        self.assertEqual(bundle["history_days"], 29)
        self.assertTrue(bundle["low_confidence"])
        for key, value in bundle.items():
            if key.startswith(("var_", "cvar_")):
                self.assertIsNone(value, key)

    def test_thirty_closes_are_enough(self):
        bundle = compute_cvar_bundle(closes_from(TAIL_CHANGES[:29]))
        self.assertIsNotNone(bundle["cvar_naive_975"])

    def test_empty_or_missing_prices_give_an_all_none_bundle(self):
        for prices in (None, []):
            with self.subTest(prices=prices):
                bundle = compute_cvar_bundle(prices)
                self.assertEqual(bundle["history_days"], 0)
                self.assertIsNone(bundle["cvar_naive_975"])

    def test_low_confidence_below_two_years_of_changes(self):
        self.assertEqual(MIN_HISTORY_DAYS, 504)
        short = compute_cvar_bundle(closes_from(alternating(0.01, MIN_HISTORY_DAYS - 1)))
        full = compute_cvar_bundle(closes_from(alternating(0.01, MIN_HISTORY_DAYS)))
        self.assertEqual(short["history_days"], MIN_HISTORY_DAYS - 1)
        self.assertTrue(short["low_confidence"])
        self.assertFalse(full["low_confidence"])
        self.assertEqual(full["history_days"], MIN_HISTORY_DAYS)


class VolatilityAdjustmentTests(unittest.TestCase):

    def test_calm_history_then_turbulence_raises_the_adjusted_tail(self):
        returns = _simple_returns(closes_from(CALM_THEN_TURBULENT))
        self.assertGreater(_ewma_vol(returns, 21)[-1], 2 * _ewma_vol(returns, 252)[-1])
        bundle = compute_cvar_bundle(closes_from(CALM_THEN_TURBULENT))
        self.assertGreater(bundle["cvar_adj_975"], bundle["cvar_naive_975"])
        self.assertGreater(bundle["cvar_adj_vs_naive_rel"], 0)

    def test_scale_factors_are_capped_at_five(self):
        scales = scale_factors(CALM_THEN_TURBULENT)
        self.assertAlmostEqual(max(scales), 5.0)
        bundle = compute_cvar_bundle(closes_from(CALM_THEN_TURBULENT))
        self.assertLessEqual(bundle["cvar_adj_975"], 5 * 3.0 + 1e-9)

    def test_turbulent_history_then_calm_is_floored_at_a_fifth(self):
        # Every scale factor hits the 0.2 floor, so each -3 % day counts as -0.6 %.
        scales = scale_factors(TURBULENT_THEN_CALM)
        self.assertTrue(all(abs(f - 0.2) < 1e-9 for f in scales))
        bundle = compute_cvar_bundle(closes_from(TURBULENT_THEN_CALM))
        self.assertAlmostEqual(bundle["cvar_naive_975"], 3.0, places=6)
        self.assertAlmostEqual(bundle["cvar_adj_975"], 0.6, places=4)
        self.assertLess(bundle["cvar_adj_vs_naive_rel"], 0)

    def test_flat_prices_give_zero_tail_and_no_relative_difference(self):
        bundle = compute_cvar_bundle([100.0] * 40)
        self.assertEqual(bundle["cvar_naive_975"], 0.0)
        self.assertEqual(bundle["cvar_adj_975"], 0.0)
        self.assertIsNone(bundle["cvar_adj_vs_naive_rel"])


class ConfidenceLevelTests(unittest.TestCase):

    def test_keys_follow_the_requested_levels(self):
        bundle = compute_cvar_bundle(closes_from(TAIL_CHANGES), confidences=(0.9,))
        tail_keys = sorted(k for k in bundle if k.startswith(("var_", "cvar_")) and k != "cvar_adj_vs_naive_rel")
        self.assertEqual(tail_keys, ["cvar_adj_900", "cvar_naive_900", "var_adj_900", "var_naive_900"])

    def test_relative_difference_falls_back_to_alpha_without_the_primary_level(self):
        prices = closes_from(CALM_THEN_TURBULENT)
        without_primary = compute_cvar_bundle(prices, confidences=(0.9,), alpha=0.975)
        self.assertNotIn("cvar_naive_975", without_primary)
        with_primary = compute_cvar_bundle(prices)
        self.assertIsNotNone(with_primary["cvar_adj_vs_naive_rel"])
        self.assertEqual(without_primary["cvar_adj_vs_naive_rel"], with_primary["cvar_adj_vs_naive_rel"])


if __name__ == "__main__":
    unittest.main()

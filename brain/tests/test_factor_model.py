"""Factor model: idio-share quintile, factor count, in-memory PCA pipeline."""

import unittest

import numpy as np

from brain.scoring.factor_model import (
    EXPLAINED_VAR_TARGET,
    MAX_K,
    MIN_HISTORY_DAYS,
    _build_return_matrix,
    _choose_k,
    _ewm_covariance,
    classify_idio_share_quintile,
    compute_idio_share_inmemory,
    latest_idio_share,
)
from brain.scoring.sloan import compute_sloan_quintile

UNIVERSE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def closes(log_moves):
    return (100.0 * np.exp(np.cumsum(log_moves))).tolist()


def one_factor_moves(seed, *, n_tickers=12, days=300, idio_vol=0.002):
    rng = np.random.default_rng([seed, 0])
    factor = rng.normal(0.0, 0.01, days)
    return factor, {f"F{i:02d}": factor + rng.normal(0.0, idio_vol, days) for i in range(n_tickers)}


def one_factor_universe(seed, **kwargs):
    _, moves = one_factor_moves(seed, **kwargs)
    return {t: closes(m) for t, m in moves.items()}


def noise_universe(seed, *, n_tickers=12, days=300):
    rng = np.random.default_rng([seed, 2])
    return {f"N{i:02d}": closes(rng.normal(0.0, 0.01, days)) for i in range(n_tickers)}


def pairwise_correlations(universe):
    moves = np.array([np.diff(np.log(universe[t])) for t in sorted(universe)])
    corr = np.corrcoef(moves)
    return corr[np.triu_indices_from(corr, k=1)]


def first_factor_share_and_k(universe):
    returns, _, _ = _build_return_matrix(universe)
    eigenvalues = np.maximum(np.linalg.eigh(_ewm_covariance(returns, 252))[0][::-1], 0.0)
    k, _ = _choose_k(eigenvalues)
    return eigenvalues[0] / eigenvalues.sum(), k


class IdioQuintileTests(unittest.TestCase):

    def test_boundaries_are_inclusive(self):
        # n = 10 sorted samples; boundaries at indices 2, 4, 6, 8 -> 0.3, 0.5, 0.7, 0.9
        for value, expected in ((0.3, 1), (0.31, 2), (0.5, 2), (0.7, 3), (0.9, 4), (0.91, 5)):
            with self.subTest(value=value):
                self.assertEqual(classify_idio_share_quintile(value, UNIVERSE), expected)

    def test_needs_ten_numeric_samples(self):
        self.assertIsNone(classify_idio_share_quintile(0.5, UNIVERSE[:9]))
        self.assertIsNone(classify_idio_share_quintile(0.5, UNIVERSE[:9] + ["n/a", None]))
        self.assertEqual(classify_idio_share_quintile(0.5, UNIVERSE[:9] + [1]), 2)

    def test_missing_value_or_universe_gives_none(self):
        self.assertIsNone(classify_idio_share_quintile(None, UNIVERSE))
        self.assertIsNone(classify_idio_share_quintile(0.5, None))

    def test_matches_the_sloan_quintile_math(self):
        for value in (-1.0, 0.1, 0.25, 0.5, 0.55, 0.9, 2.0):
            with self.subTest(value=value):
                self.assertEqual(
                    classify_idio_share_quintile(value, UNIVERSE), compute_sloan_quintile(value, UNIVERSE),
                )


class ChooseKTests(unittest.TestCase):

    def test_smallest_k_reaching_the_target(self):
        # cumulative shares 0.5, 0.8, 0.9, 1.0 -> 0.9 first reached at k = 3
        k, explained = _choose_k(np.array([5.0, 3.0, 1.0, 1.0]))
        self.assertEqual(k, 3)
        self.assertAlmostEqual(explained, 0.9)

    def test_k_is_capped(self):
        k, explained = _choose_k(np.ones(20))
        self.assertEqual(k, MAX_K)
        self.assertAlmostEqual(explained, MAX_K / 20)

    def test_zero_variance_gives_one_factor(self):
        self.assertEqual(_choose_k(np.zeros(4)), (1, 0.0))


class InMemoryPipelineTests(unittest.TestCase):

    def test_empty_input(self):
        out, meta = compute_idio_share_inmemory({})
        self.assertEqual(out, {})
        self.assertEqual(meta["reason"], "empty_input")
        self.assertEqual(meta["coverage_pct"], 0.0)

    def test_fewer_than_ten_tickers_give_no_output(self):
        out, meta = compute_idio_share_inmemory(one_factor_universe(1, n_tickers=9))
        self.assertEqual(out, {})
        self.assertTrue(meta["reason"].startswith("insufficient_universe"))
        self.assertEqual(len(meta["missing_tickers"]), 9)

    def test_histories_shorter_than_the_minimum_are_filtered(self):
        universe = one_factor_universe(2, n_tickers=10)
        universe["F00"] = universe["F00"][-(MIN_HISTORY_DAYS - 1):]
        out, meta = compute_idio_share_inmemory(universe)
        self.assertEqual(meta["input_count"], 10)
        self.assertEqual(out, {})
        self.assertEqual(meta["reason"], "insufficient_after_history_filter (9 eligible)")

    def test_tuples_count_as_histories_but_arrays_do_not(self):
        universe = one_factor_universe(2, n_tickers=10)
        universe["F01"] = tuple(universe["F01"])
        out, _ = compute_idio_share_inmemory(universe)
        self.assertIn("F01", out)
        universe["F02"] = np.array(universe["F02"])
        out, meta = compute_idio_share_inmemory(universe)
        self.assertEqual(out, {})
        self.assertEqual(meta["reason"], "insufficient_after_history_filter (9 eligible)")

    def test_history_shorter_than_the_window_is_reported(self):
        out, meta = compute_idio_share_inmemory(one_factor_universe(3, days=240))
        self.assertEqual(out, {})
        self.assertEqual(meta["reason"], "history_shorter_than_window (239d < 252d)")

    def test_window_can_be_shortened(self):
        out, meta = compute_idio_share_inmemory(one_factor_universe(3, days=240), window_days=120)
        self.assertEqual(meta["output_count"], 12)
        self.assertEqual(meta["reason"], "")

    def test_full_coverage_and_shares_between_zero_and_one(self):
        out, meta = compute_idio_share_inmemory(one_factor_universe(4))
        self.assertEqual(sorted(out), [f"F{i:02d}" for i in range(12)])
        self.assertEqual(meta["coverage_pct"], 100.0)
        self.assertEqual(meta["missing_tickers"], [])
        self.assertTrue(all(0.0 <= v <= 1.0 for v in out.values()))

    def test_factor_driven_universe_has_low_idio_share(self):
        universe = one_factor_universe(5)
        self.assertGreater(min(pairwise_correlations(universe)), 0.9)
        out, _ = compute_idio_share_inmemory(universe)
        self.assertLess(max(out.values()), 0.1)

    def test_independent_noise_leaves_more_idiosyncratic_variance(self):
        factor_universe, noise = one_factor_universe(6), noise_universe(6)
        self.assertGreater(min(pairwise_correlations(factor_universe)), 0.9)
        self.assertLess(max(abs(pairwise_correlations(noise))), 0.25)
        factor_out, _ = compute_idio_share_inmemory(factor_universe)
        noise_out, _ = compute_idio_share_inmemory(noise)
        self.assertGreater(np.mean(list(noise_out.values())), 3 * np.mean(list(factor_out.values())))

    def test_lone_uncorrelated_high_variance_ticker_becomes_its_own_factor(self):
        # Eleven tickers share one factor (variance about 1e-4 each); LONE is an
        # independent draw with variance 4e-4. The first factor then explains
        # well under the 90 % target, so a second factor is kept, and it absorbs
        # LONE: its share is "idiosyncratic" relative to the statistical factors,
        # not relative to an economic market factor.
        factor, moves = one_factor_moves(7, n_tickers=11)
        lone = np.random.default_rng([7, 1]).normal(0.0, 0.02, len(factor))
        self.assertLess(abs(np.corrcoef(lone, factor)[0, 1]), 0.2)
        universe = {t: closes(m) for t, m in moves.items()}
        universe["LONE"] = closes(lone)
        first_share, k = first_factor_share_and_k(universe)
        self.assertLess(first_share, EXPLAINED_VAR_TARGET)
        self.assertEqual(k, 2)
        out, _ = compute_idio_share_inmemory(universe)
        self.assertLess(out["LONE"], 0.05)

    def test_zero_variance_ticker_is_reported_as_missing(self):
        universe = one_factor_universe(8)
        universe["FLAT"] = [100.0] * 300
        out, meta = compute_idio_share_inmemory(universe)
        self.assertNotIn("FLAT", out)
        self.assertEqual(meta["missing_tickers"], ["FLAT"])
        self.assertAlmostEqual(meta["coverage_pct"], 100.0 * 12 / 13)

    def test_same_input_gives_the_same_shares(self):
        first, _ = compute_idio_share_inmemory(one_factor_universe(9))
        second, _ = compute_idio_share_inmemory(one_factor_universe(9))
        self.assertEqual(first, second)


class LatestIdioShareTests(unittest.TestCase):

    def test_lookup(self):
        self.assertEqual(latest_idio_share("A", {"A": 0.3}), 0.3)
        self.assertIsNone(latest_idio_share("B", {"A": 0.3}))
        self.assertIsNone(latest_idio_share("A", {}))
        self.assertIsNone(latest_idio_share("A", None))


if __name__ == "__main__":
    unittest.main()

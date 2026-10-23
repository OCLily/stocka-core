"""Pearson correlation helpers: rolling series, latest-window matrix, pair edges."""

import unittest

from brain.risk.correlation import (
    CLUSTER_THRESHOLD_DEFAULT,
    CORRELATION_WINDOW_DEFAULT,
    CorrelationError,
    _pearson,
    find_correlation_clusters,
    pairwise_correlation_matrix,
    pearson_rolling_correlation,
)

BASE = [0.01, -0.02, 0.015, 0.0, -0.005, 0.02, -0.01, 0.005]


class PearsonTests(unittest.TestCase):

    def test_identical_series_correlate_at_one(self):
        self.assertAlmostEqual(_pearson(BASE, BASE), 1.0)

    def test_scaled_and_shifted_series_correlate_at_one(self):
        self.assertAlmostEqual(_pearson(BASE, [3 * x + 0.1 for x in BASE]), 1.0)

    def test_mirrored_series_correlate_at_minus_one(self):
        self.assertAlmostEqual(_pearson(BASE, [-x for x in BASE]), -1.0)

    def test_hand_computed_value(self):
        # x = 1,2,3; y = 1,3,2: deviations (-1,0,1) and (-1,1,0)
        # covariance sum = 1, variance sums = 2 and 2, rho = 1 / 2
        self.assertAlmostEqual(_pearson([1.0, 2.0, 3.0], [1.0, 3.0, 2.0]), 0.5)

    def test_flat_series_gives_zero_not_an_error(self):
        self.assertEqual(_pearson([0.01] * 5, [0.01, 0.02, 0.0, 0.03, 0.01]), 0.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(CorrelationError):
            _pearson([0.1, 0.2, 0.3], [0.1, 0.2])

    def test_fewer_than_two_observations_raise(self):
        with self.assertRaises(CorrelationError):
            _pearson([0.1], [0.2])


class RollingCorrelationTests(unittest.TestCase):

    def test_one_value_per_full_window(self):
        a = BASE * 3
        b = [x * 0.5 + 0.001 * i for i, x in enumerate(a)]
        out = pearson_rolling_correlation(a, b, window=5)
        self.assertEqual(len(out), len(a) - 5 + 1)

    def test_each_value_is_the_pearson_of_its_window(self):
        a = BASE * 2
        b = list(reversed(a))
        out = pearson_rolling_correlation(a, b, window=4)
        for i, value in enumerate(out):
            self.assertAlmostEqual(value, _pearson(a[i:i + 4], b[i:i + 4]))

    def test_series_exactly_one_window_long_gives_one_value(self):
        self.assertEqual(len(pearson_rolling_correlation(BASE, BASE, window=len(BASE))), 1)

    def test_default_window_is_sixty(self):
        self.assertEqual(CORRELATION_WINDOW_DEFAULT, 60)
        with self.assertRaises(CorrelationError):
            pearson_rolling_correlation(BASE * 7, BASE * 7)
        self.assertEqual(len(pearson_rolling_correlation(BASE * 8, BASE * 8)), len(BASE) * 8 - 59)

    def test_invalid_inputs_raise(self):
        cases = [
            (BASE, BASE[:-1], 3),   # length mismatch
            (BASE, BASE, 1),        # window below 2
            (BASE, BASE, 9),        # fewer observations than the window
        ]
        for a, b, window in cases:
            with self.subTest(window=window, n=(len(a), len(b))):
                with self.assertRaises(CorrelationError):
                    pearson_rolling_correlation(a, b, window=window)


class PairwiseMatrixTests(unittest.TestCase):

    def setUp(self):
        self.series = {
            "B": BASE,
            "A": [2 * x for x in BASE],
            "C": [-x for x in BASE],
        }

    def test_matrix_is_symmetric_with_unit_diagonal(self):
        m = pairwise_correlation_matrix(self.series, window=len(BASE))
        self.assertEqual(sorted(m), ["A", "B", "C"])
        for a in m:
            self.assertEqual(m[a][a], 1.0)
            for b in m:
                self.assertAlmostEqual(m[a][b], m[b][a])
        self.assertAlmostEqual(m["A"]["B"], 1.0)
        self.assertAlmostEqual(m["A"]["C"], -1.0)

    def test_only_the_last_window_counts(self):
        early_opposite = [-x for x in BASE]
        series = {"X": early_opposite + BASE, "Y": BASE + BASE}
        self.assertLess(_pearson(series["X"], series["Y"]), 0.5)
        m = pairwise_correlation_matrix(series, window=len(BASE))
        self.assertAlmostEqual(m["X"]["Y"], 1.0)

    def test_series_of_different_length_are_aligned_at_the_end(self):
        series = {"LONG": [0.5, -0.5, 0.3] + BASE, "SHORT": BASE}
        self.assertLess(_pearson(series["LONG"][:len(BASE)], BASE), 0.9)
        m = pairwise_correlation_matrix(series, window=len(BASE))
        self.assertAlmostEqual(m["LONG"]["SHORT"], 1.0)

    def test_ticker_shorter_than_the_window_raises(self):
        with self.assertRaisesRegex(CorrelationError, "SHORT"):
            pairwise_correlation_matrix({"OK": BASE, "SHORT": BASE[:3]}, window=5)


class PairEdgeTests(unittest.TestCase):

    MATRIX = {
        "A": {"A": 1.0, "B": 0.9, "C": -0.95, "D": 0.75},
        "B": {"A": 0.9, "B": 1.0, "C": 0.2, "D": 0.1},
        "C": {"A": -0.95, "B": 0.2, "C": 1.0, "D": 0.0},
        "D": {"A": 0.75, "B": 0.1, "C": 0.0, "D": 1.0},
    }

    def test_returns_each_pair_above_the_threshold_once(self):
        pairs = find_correlation_clusters(self.MATRIX, threshold=0.75)
        self.assertEqual([(a, b) for a, b, _ in pairs], [("A", "C"), ("A", "B")])

    def test_strong_negative_correlation_counts_as_an_edge(self):
        pairs = find_correlation_clusters(self.MATRIX, threshold=0.75)
        self.assertIn(("A", "C", -0.95), pairs)

    def test_sorted_by_absolute_correlation_descending(self):
        pairs = find_correlation_clusters(self.MATRIX, threshold=0.0)
        magnitudes = [abs(rho) for _, _, rho in pairs]
        self.assertEqual(magnitudes, sorted(magnitudes, reverse=True))

    def test_threshold_is_strict(self):
        pairs = find_correlation_clusters(self.MATRIX, threshold=0.75)
        self.assertNotIn(("A", "D", 0.75), pairs)

    def test_default_threshold_is_0_75(self):
        self.assertEqual(CLUSTER_THRESHOLD_DEFAULT, 0.75)
        self.assertEqual(find_correlation_clusters(self.MATRIX), find_correlation_clusters(self.MATRIX, threshold=0.75))

    def test_missing_entries_are_skipped(self):
        # Only matrix[a][b] with a < b is read; here A->B is missing, and the
        # B->A entry is not consulted.
        partial = {"A": {"A": 1.0}, "B": {"B": 1.0, "A": 0.99}}
        self.assertNotIn("B", partial["A"])
        self.assertEqual(find_correlation_clusters(partial), [])

    def test_empty_matrix_gives_no_pairs(self):
        self.assertEqual(find_correlation_clusters({}), [])


if __name__ == "__main__":
    unittest.main()

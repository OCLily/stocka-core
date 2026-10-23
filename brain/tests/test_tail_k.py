"""tail_k: how many worst observations enter the ES/CVaR tail mean."""

import math
import unittest

from brain.risk import es_historical
from brain.risk._tail_k import tail_k


class TailKTests(unittest.TestCase):

    def test_is_floor_of_tail_share_plus_one(self):
        cases = [
            (756, 0.975, 19),   # floor(18.9) + 1
            (756, 0.99, 8),     # floor(7.56) + 1
            (40, 0.95, 3),      # floor(2.0) + 1
            (500, 0.95, 26),    # floor(25.0) + 1
            (20, 0.99, 1),      # floor(0.2) + 1
        ]
        for n, alpha, expected in cases:
            with self.subTest(n=n, alpha=alpha):
                self.assertEqual(tail_k(n, alpha), expected)

    def test_takes_one_more_observation_than_the_plain_floor_rule(self):
        for n in (25, 100, 252, 756):
            for alpha in (0.95, 0.975, 0.99):
                with self.subTest(n=n, alpha=alpha):
                    self.assertEqual(tail_k(n, alpha), math.floor(n * (1 - alpha)) + 1)

    def test_small_samples_still_use_at_least_one_observation(self):
        for n in (1, 2, 5, 19):
            with self.subTest(n=n):
                self.assertEqual(tail_k(n, 0.99), 1)

    def test_never_exceeds_the_sample_size(self):
        self.assertEqual(tail_k(10, 0.0), 10)
        self.assertEqual(tail_k(1, 0.5), 1)

    def test_stays_within_one_and_n(self):
        for n in range(1, 60):
            for alpha in (0.0, 0.5, 0.9, 0.95, 0.975, 0.99, 1.0):
                k = tail_k(n, alpha)
                self.assertTrue(1 <= k <= n, (n, alpha, k))

    def test_does_not_shrink_as_the_sample_grows(self):
        ks = [tail_k(n, 0.975) for n in range(1, 800)]
        self.assertEqual(ks, sorted(ks))

    def test_es_historical_uses_the_same_rule(self):
        for n, alpha in ((756, 0.975), (756, 0.99), (20, 0.99), (40, 0.95)):
            with self.subTest(n=n, alpha=alpha):
                self.assertEqual(es_historical._worst_k(alpha, n), tail_k(n, alpha))


if __name__ == "__main__":
    unittest.main()

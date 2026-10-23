"""Pure calibration functions — Brier score + ECE.

All expected values are hand-computed in comments so every pin is auditable
without running the code.
"""

import unittest

from brain.forecast.calibration import brier_score, ece


class BrierScoreTests(unittest.TestCase):

    def test_perfect_confident_predictions(self):
        # brier([1.0, 0.0], [1, 0])
        # pair 1: (1.0 - 1)^2 = 0.0
        # pair 2: (0.0 - 0)^2 = 0.0
        # mean = 0.0
        self.assertEqual(brier_score([1.0, 0.0], [1, 0]), 0.0)

    def test_uniformly_uncertain(self):
        # brier([0.5, 0.5], [1, 0])
        # pair 1: (0.5 - 1)^2 = 0.25
        # pair 2: (0.5 - 0)^2 = 0.25
        # mean = 0.25
        self.assertAlmostEqual(brier_score([0.5, 0.5], [1, 0]), 0.25)

    def test_worst_single_prediction(self):
        # brier([0.0], [1])
        # (0.0 - 1)^2 = 1.0
        self.assertEqual(brier_score([0.0], [1]), 1.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            brier_score([0.5, 0.5], [1])

    def test_prob_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            brier_score([1.5, 0.5], [1, 0])

    def test_bad_outcome_raises(self):
        with self.assertRaises(ValueError):
            brier_score([0.5, 0.5], [1, 2])

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            brier_score([], [])


class ECETests(unittest.TestCase):

    def test_perfectly_calibrated_returns_zero(self):
        # p=1.0 → outcome=1: bin = min(int(1.0*10), 9) = 9, conf=1.0, acc=1.0 → diff=0
        # p=0.0 → outcome=0: bin = min(int(0.0*10), 9) = 0, conf=0.0, acc=0.0 → diff=0
        # ece = sum(weight * 0) = 0.0
        self.assertEqual(ece([1.0, 0.0], [1, 0]), 0.0)

    def test_multi_bin_hand_computed(self):
        # 4 predictions across 2 bins (n_bins=2):
        # probs =    [0.2, 0.3, 0.7, 0.8]
        # outcomes = [0,   1,   1,   1  ]
        #
        # bin assignment: bin = min(int(p * 2), 1)
        #   p=0.2 → int(0.4)=0 → bin 0
        #   p=0.3 → int(0.6)=0 → bin 0
        #   p=0.7 → int(1.4)=1 → bin 1
        #   p=0.8 → int(1.6)=1 → bin 1
        #
        # bin 0: probs=[0.2,0.3], outcomes=[0,1]
        #   confidence = (0.2+0.3)/2 = 0.25
        #   accuracy   = (0+1)/2     = 0.5
        #   weight     = 2/4         = 0.5
        #   contribution = 0.5 * |0.5 - 0.25| = 0.5 * 0.25 = 0.125
        #
        # bin 1: probs=[0.7,0.8], outcomes=[1,1]
        #   confidence = (0.7+0.8)/2 = 0.75
        #   accuracy   = (1+1)/2     = 1.0
        #   weight     = 2/4         = 0.5
        #   contribution = 0.5 * |1.0 - 0.75| = 0.5 * 0.25 = 0.125
        #
        # ece = 0.125 + 0.125 = 0.25
        result = ece([0.2, 0.3, 0.7, 0.8], [0, 1, 1, 1], n_bins=2)
        self.assertAlmostEqual(result, 0.25)

    def test_prob_one_lands_in_last_bin(self):
        # p=1.0 must NOT raise — bin = min(int(1.0 * 10), 9) = min(10, 9) = 9
        # conf=1.0, acc=1.0, |diff|=0 → ece=0.0
        self.assertEqual(ece([1.0], [1]), 0.0)

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            ece([], [])

    def test_n_bins_less_than_one_raises(self):
        with self.assertRaises(ValueError):
            ece([0.5], [1], n_bins=0)

    def test_out_of_range_prob_raises(self):
        with self.assertRaises(ValueError):
            ece([-0.1, 0.5], [0, 1])

    def test_bad_outcome_raises(self):
        with self.assertRaises(ValueError):
            ece([0.5], [2])

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            ece([0.5, 0.5], [1])


if __name__ == "__main__":
    unittest.main()

"""get_dd_multiplier: sizing multiplier per drawdown stage."""

import unittest

from brain.state.dd_multiplier import DD_MULTIPLIER_TABLE, DDMultiplierError, get_dd_multiplier
from brain.state.dd_state_machine import TIER_ACTIONS, TIER_THRESHOLDS_PCT


class DDMultiplierTests(unittest.TestCase):

    def test_documented_multiplier_per_stage(self):
        for tier, expected in ((0, 1.0), (1, 0.5), (2, 0.0), (3, 0.0)):
            with self.subTest(tier=tier):
                self.assertEqual(get_dd_multiplier(tier), expected)

    def test_multiplier_never_rises_with_a_deeper_stage(self):
        values = [get_dd_multiplier(t) for t in sorted(DD_MULTIPLIER_TABLE)]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_multipliers_stay_between_zero_and_one(self):
        self.assertTrue(all(0.0 <= v <= 1.0 for v in DD_MULTIPLIER_TABLE.values()))

    def test_every_stage_of_the_state_machine_has_a_multiplier(self):
        self.assertEqual(set(DD_MULTIPLIER_TABLE), set(TIER_ACTIONS))
        self.assertEqual(set(DD_MULTIPLIER_TABLE) - {0}, set(TIER_THRESHOLDS_PCT))

    def test_unknown_stage_raises(self):
        for tier in (-1, 4, 99, None, "1"):
            with self.subTest(tier=tier):
                with self.assertRaises(DDMultiplierError):
                    get_dd_multiplier(tier)

    def test_error_message_names_the_bad_stage(self):
        with self.assertRaisesRegex(DDMultiplierError, "7"):
            get_dd_multiplier(7)


if __name__ == "__main__":
    unittest.main()

"""Example screen: each threshold, each engine signal, and the reasons it reports."""

import unittest
from dataclasses import replace

from stocka.config import ScreenSettings
from stocka.screen import EngineResult, run_screen, screen_ticker

DEFAULTS = ScreenSettings()


def make_result(*, score=90, vetos=(), flags=(), **overrides):
    result = EngineResult(
        ticker="TST", sector="Industrials",
        dq={"composite_score": score, "vetos": list(vetos), "flags": list(flags)},
        dq_error=None, beneish_m=-2.6, sloan_ratio=-0.03, sloan_quintile=2,
    )
    return replace(result, **overrides)


class ScreenTests(unittest.TestCase):

    def test_clean_result_passes_without_reasons(self):
        decision = screen_ticker(make_result(), DEFAULTS)
        self.assertTrue(decision.passed)
        self.assertEqual(decision.reasons, ())

    def test_contract_error_fails_with_the_error_as_only_reason(self):
        decision = screen_ticker(make_result(dq=None, dq_error="missing restatements_36m"), DEFAULTS)
        self.assertFalse(decision.passed)
        self.assertEqual(decision.reasons, ("audit input rejected: missing restatements_36m",))

    def test_every_engine_veto_is_a_reason(self):
        decision = screen_ticker(make_result(vetos=["going_concern", "restatement_36m"]), DEFAULTS)
        self.assertFalse(decision.passed)
        self.assertIn("engine veto: going_concern", decision.reasons)
        self.assertIn("engine veto: restatement_36m", decision.reasons)

    def test_minimum_dq_score_is_inclusive(self):
        self.assertTrue(screen_ticker(make_result(score=70), DEFAULTS).passed)
        decision = screen_ticker(make_result(score=69), DEFAULTS)
        self.assertEqual(decision.reasons, ("DQ score 69 < 70",))

    def test_maximum_beneish_m_is_inclusive(self):
        self.assertTrue(screen_ticker(make_result(beneish_m=-2.22), DEFAULTS).passed)
        decision = screen_ticker(make_result(beneish_m=-2.1), DEFAULTS)
        self.assertEqual(decision.reasons, ("Beneish M -2.10 > -2.22",))

    def test_beneish_red_veto_is_not_repeated_as_a_threshold_reason(self):
        decision = screen_ticker(make_result(beneish_m=-1.0, vetos=["beneish_red_flag"]), DEFAULTS)
        self.assertEqual(decision.reasons, ("engine veto: beneish_red_flag",))

    def test_sector_bypass_skips_the_beneish_threshold_with_a_note(self):
        self.assertFalse(screen_ticker(make_result(beneish_m=-1.0), DEFAULTS).passed)
        for flag in ("beneish_red_not_applicable_utility", "beneish_not_applicable_financial"):
            with self.subTest(flag=flag):
                decision = screen_ticker(make_result(beneish_m=-1.0, flags=[flag]), DEFAULTS)
                self.assertTrue(decision.passed)
                self.assertIn("Beneish not applied (sector bypass)", decision.notes)

    def test_sloan_q5_fails_only_when_excluded(self):
        result = make_result(sloan_quintile=5, flags=["sloan_q5_boundary"])
        self.assertEqual(screen_ticker(result, DEFAULTS).reasons, ("Sloan accruals in Q5",))
        self.assertTrue(screen_ticker(result, replace(DEFAULTS, exclude_sloan_q5=False)).passed)

    def test_sloan_anomaly_veto_is_not_repeated_as_a_q5_reason(self):
        decision = screen_ticker(make_result(sloan_quintile=5, vetos=["sloan_q5_anomaly"]), DEFAULTS)
        self.assertEqual(decision.reasons, ("engine veto: sloan_q5_anomaly",))

    def test_missing_sloan_quintile_is_a_note_not_a_failure(self):
        decision = screen_ticker(make_result(sloan_quintile=None, flags=["sloan_quintile_unavailable"]), DEFAULTS)
        self.assertTrue(decision.passed)
        self.assertIn("Sloan quintile unavailable", decision.notes)

    def test_sloan_sector_bypass_is_a_note(self):
        self.assertIn("Sloan quintile unavailable", screen_ticker(make_result(sloan_quintile=None), DEFAULTS).notes)
        decision = screen_ticker(make_result(sloan_quintile=None, flags=["sloan_not_applicable_financial"]), DEFAULTS)
        self.assertTrue(decision.passed)
        self.assertIn("Sloan not applied (sector bypass)", decision.notes)

    def test_unverified_audit_inputs_fail_only_when_required(self):
        result = make_result(flags=["going_concern_unverified", "restatements_unverified"])
        decision = screen_ticker(result, DEFAULTS)
        self.assertEqual(
            decision.reasons,
            ("unverified input: going_concern_unverified", "unverified input: restatements_unverified"),
        )
        self.assertTrue(screen_ticker(result, replace(DEFAULTS, require_verified_audit=False)).passed)

    def test_run_screen_keeps_input_order(self):
        results = [make_result(ticker="B"), make_result(ticker="A", score=10)]
        decisions = run_screen(results, DEFAULTS)
        self.assertEqual([d.ticker for d in decisions], ["B", "A"])
        self.assertEqual([d.passed for d in decisions], [True, False])


if __name__ == "__main__":
    unittest.main()

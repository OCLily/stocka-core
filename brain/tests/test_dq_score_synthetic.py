"""Behaviour of compute_dq_score and get_dq_vetos on synthetic EODHD-shaped input.

All figures are made up. Where a test needs an exact Beneish M or Sloan
accruals ratio, the two calculators are replaced on the dq_score module with
unittest.mock.patch so that the boundaries are hit exactly.
"""

import unittest
from datetime import date, timedelta
from unittest import mock

from brain.scoring import dq_score
from brain.scoring.dq_score import (
    SLOAN_Q5_ABSOLUTE_VETO_THRESHOLD,
    DQContractError,
    compute_dq_score,
    get_dq_vetos,
)

AS_OF = date(2025, 6, 30)
FRESH_FILING = (AS_OF - timedelta(days=60)).isoformat()
UNIVERSE = [-0.09, -0.08, -0.07, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0.0]


def make_fundamentals(sector="Industrials", filing_date=FRESH_FILING):
    return {
        "General": {"Code": "TST", "GicSector": sector},
        "Financials": {"Income_Statement": {"yearly": {
            "2024-12-31": {"filing_date": filing_date},
            "2023-12-31": {"filing_date": "2024-03-01"},
        }}},
    }


def make_audit(**overrides):
    audit = {
        "auditor_tier": "A",
        "going_concern": False,
        "restatements_36m": 0,
        "revenue_recognition_restatement": False,
    }
    audit.update(overrides)
    return audit


def beneish_result(m_score):
    return {"m_score": m_score, "components": {}, "complete": True, "missing_inputs": [], "flag": "stub"}


BENEISH_INCOMPLETE = {
    "m_score": None, "components": {}, "complete": False,
    "missing_inputs": ["Income.grossProfit[t]"], "flag": "incomplete",
}


def sloan_result(ratio, complete=True):
    return {
        "accruals": None, "avg_total_assets": None, "accruals_ratio": ratio,
        "complete": complete, "missing_inputs": [] if complete else ["Balance.totalAssets[t]"],
    }


def score(*, m_score=-3.0, ratio=-0.05, sector="Industrials", audit=None, universe=UNIVERSE,
          fundamentals=None, beneish=None, sloan=None, **kwargs):
    with mock.patch.object(dq_score, "compute_beneish_m_score", return_value=beneish or beneish_result(m_score)), \
         mock.patch.object(dq_score, "compute_sloan_accruals", return_value=sloan or sloan_result(ratio)):
        return compute_dq_score(
            ticker="TST",
            fundamentals=fundamentals if fundamentals is not None else make_fundamentals(sector),
            audit_signals=make_audit() if audit is None else audit,
            sloan_universe=universe,
            as_of=AS_OF,
            **kwargs,
        )


class CompositeTests(unittest.TestCase):

    def test_clean_input_scores_100_without_vetos_or_flags(self):
        # 30 report age + 15 auditor A + 0 restatements + 0 revenue recognition
        # + 5 Sloan Q2 + 30 Beneish clear + 20 no going concern = 100
        result = score()
        self.assertEqual(result["composite_score"], 100)
        self.assertEqual(result["vetos"], [])
        self.assertEqual(result["flags"], [])
        self.assertIsNone(result["composite_cap_applied"])

    def test_composite_equals_raw_points_when_nothing_caps_it(self):
        result = score(audit=make_audit(auditor_tier="B"))
        self.assertEqual(result["raw_points"], 95)
        self.assertEqual(result["composite_score"], 95)

    def test_composite_is_clamped_at_zero(self):
        result = score(audit=make_audit(going_concern=True, auditor_tier="X"))
        self.assertLess(result["raw_points"], 0)
        self.assertEqual(result["composite_score"], 0)

    def test_result_echoes_ticker_and_as_of(self):
        result = score()
        self.assertEqual(result["ticker"], "TST")
        self.assertEqual(result["as_of"], AS_OF.isoformat())


class BeneishThresholdTests(unittest.TestCase):

    def test_m_just_below_gray_floor_earns_full_points(self):
        result = score(m_score=-2.2201)
        self.assertEqual(result["components"]["beneish"]["m_score"], -2.2201)
        self.assertEqual(result["components"]["beneish"]["points"], 30)
        self.assertEqual(result["vetos"], [])

    def test_m_exactly_at_gray_floor_is_gray(self):
        result = score(m_score=-2.22)
        self.assertEqual(result["components"]["beneish"]["points"], 5)
        self.assertEqual(result["vetos"], [])

    def test_m_exactly_at_red_line_is_still_gray_without_veto(self):
        result = score(m_score=-1.78)
        self.assertEqual(result["components"]["beneish"]["points"], 5)
        self.assertEqual(result["vetos"], [])
        self.assertIsNone(result["composite_cap_applied"])

    def test_m_above_red_line_vetoes_and_caps_composite_at_30(self):
        result = score(m_score=-1.7799)
        self.assertEqual(result["components"]["beneish"]["points"], 0)
        self.assertIn("beneish_red_flag", result["vetos"])
        self.assertEqual(result["composite_cap_applied"], 30)
        self.assertLessEqual(result["composite_score"], 30)

    def test_red_m_in_utilities_is_flagged_not_vetoed(self):
        self.assertIn("beneish_red_flag", score(m_score=-1.0, sector="Industrials")["vetos"])
        result = score(m_score=-1.0, sector="Utilities")
        self.assertEqual(result["vetos"], [])
        self.assertIn("beneish_red_not_applicable_utility", result["flags"])
        self.assertEqual(result["components"]["beneish"]["points"], 5)

    def test_red_m_in_financials_is_flagged_not_vetoed(self):
        self.assertIn("beneish_red_flag", score(m_score=-1.0, sector="Industrials")["vetos"])
        result = score(m_score=-1.0, sector="Financials")
        self.assertEqual(result["vetos"], [])
        self.assertIn("beneish_red_not_applicable_financial", result["flags"])

    def test_red_m_in_real_estate_gets_no_bypass(self):
        result = score(m_score=-1.0, sector="Real Estate")
        self.assertIn("beneish_red_flag", result["vetos"])

    def test_incomplete_beneish_vetoes_a_non_financial_ticker(self):
        result = score(beneish=BENEISH_INCOMPLETE)
        self.assertIn("beneish_data_incomplete_phase1", result["vetos"])
        self.assertIn("beneish_incomplete", result["flags"])
        self.assertEqual(result["composite_cap_applied"], 30)
        self.assertEqual(result["components"]["beneish"]["missing_inputs"], ["Income.grossProfit[t]"])

    def test_incomplete_beneish_is_not_applicable_for_financials(self):
        self.assertIn("beneish_data_incomplete_phase1", score(beneish=BENEISH_INCOMPLETE)["vetos"])
        result = score(beneish=BENEISH_INCOMPLETE, sector="Financials")
        self.assertNotIn("beneish_data_incomplete_phase1", result["vetos"])
        self.assertIn("beneish_not_applicable_financial", result["flags"])


class SloanTests(unittest.TestCase):

    def test_absolute_veto_threshold_is_five_percent(self):
        self.assertEqual(SLOAN_Q5_ABSOLUTE_VETO_THRESHOLD, 0.05)

    def test_q5_ratio_exactly_at_threshold_is_a_boundary_flag_not_a_veto(self):
        result = score(ratio=0.05)
        component = result["components"]["sloan_quintile"]
        self.assertEqual(component["quintile"], 5)
        self.assertEqual(component["classification"], "boundary")
        self.assertEqual(component["points"], -10)
        self.assertIn("sloan_q5_boundary", result["flags"])
        self.assertEqual(result["vetos"], [])

    def test_q5_ratio_above_threshold_is_an_anomaly_veto(self):
        result = score(ratio=0.0501)
        component = result["components"]["sloan_quintile"]
        self.assertEqual(component["quintile"], 5)
        self.assertEqual(component["classification"], "anomaly")
        self.assertEqual(component["points"], -30)
        self.assertIn("sloan_q5_anomaly", result["vetos"])

    def test_quintile_below_five_earns_five_points(self):
        result = score(ratio=-0.085)
        self.assertEqual(result["components"]["sloan_quintile"]["quintile"], 1)
        self.assertEqual(result["components"]["sloan_quintile"]["points"], 5)

    def test_financials_skip_quintile_classification(self):
        self.assertIn("sloan_q5_anomaly", score(ratio=0.2)["vetos"])
        result = score(ratio=0.2, sector="Financials")
        self.assertIsNone(result["components"]["sloan_quintile"]["quintile"])
        self.assertIn("sloan_not_applicable_financial", result["flags"])
        self.assertNotIn("sloan_q5_anomaly", result["vetos"])

    def test_missing_universe_gives_a_soft_flag_without_veto(self):
        self.assertIn("sloan_q5_anomaly", score(ratio=0.2)["vetos"])
        result = score(ratio=0.2, universe=None)
        self.assertIn("sloan_quintile_unavailable", result["flags"])
        self.assertEqual(result["vetos"], [])

    def test_universe_below_ten_samples_is_treated_as_incomplete_and_vetoed(self):
        self.assertEqual(score(universe=UNIVERSE)["vetos"], [])
        result = score(universe=UNIVERSE[:9])
        self.assertIn("sloan_data_incomplete_phase1", result["vetos"])
        self.assertIn("sloan_incomplete", result["flags"])
        self.assertEqual(result["composite_cap_applied"], 30)

    def test_incomplete_sloan_inputs_with_a_universe_are_vetoed(self):
        result = score(sloan=sloan_result(None, complete=False))
        self.assertIn("sloan_data_incomplete_phase1", result["vetos"])


class RestatementTests(unittest.TestCase):

    def test_no_restatement_costs_nothing(self):
        result = score(audit=make_audit(restatements_36m=0))
        self.assertEqual(result["components"]["restatements_36m"], {"count": 0, "points": 0})
        self.assertEqual(result["vetos"], [])

    def test_one_restatement_costs_20_points_and_vetoes(self):
        result = score(audit=make_audit(restatements_36m=1))
        self.assertEqual(result["components"]["restatements_36m"]["points"], -20)
        self.assertIn("restatement_36m", result["vetos"])
        self.assertIsNone(result["composite_cap_applied"])

    def test_two_restatements_cost_100_points_cap_at_40_and_veto(self):
        result = score(audit=make_audit(restatements_36m=2))
        self.assertEqual(result["components"]["restatements_36m"]["points"], -100)
        self.assertEqual(result["composite_cap_applied"], 40)
        self.assertIn("restatement_36m", result["vetos"])

    def test_unknown_restatement_count_is_flagged_not_vetoed(self):
        result = score(audit=make_audit(restatements_36m=None))
        self.assertEqual(result["components"]["restatements_36m"], {"count": None, "points": 0})
        self.assertIn("restatements_unverified", result["flags"])
        self.assertEqual(result["vetos"], [])

    def test_revenue_recognition_restatement_costs_10_points(self):
        result = score(audit=make_audit(revenue_recognition_restatement=True))
        self.assertEqual(result["components"]["revenue_recognition_restatement"]["points"], -10)
        self.assertEqual(result["composite_score"], 90)


class ReportAgeTests(unittest.TestCase):

    def test_age_bands(self):
        for days, points in ((0, 30), (60, 30), (61, 20), (120, 20), (121, 10), (180, 10), (181, 0), (400, 0)):
            with self.subTest(days=days):
                filing = (AS_OF - timedelta(days=days)).isoformat()
                result = score(fundamentals=make_fundamentals(filing_date=filing))
                self.assertEqual(result["components"]["report_age"]["days"], days)
                self.assertEqual(result["components"]["report_age"]["points"], points)

    def test_filing_date_in_the_future_counts_as_fresh(self):
        filing = (AS_OF + timedelta(days=5)).isoformat()
        result = score(fundamentals=make_fundamentals(filing_date=filing))
        self.assertEqual(result["components"]["report_age"]["days"], 0)
        self.assertEqual(result["components"]["report_age"]["points"], 30)

    def test_audit_filing_date_takes_precedence_over_income_statement(self):
        audit = make_audit(filing_date=(AS_OF - timedelta(days=10)).isoformat())
        result = score(audit=audit, fundamentals=make_fundamentals(filing_date="2020-01-01"))
        self.assertEqual(result["components"]["report_age"]["source"], "audit_signals.filing_date")
        self.assertEqual(result["components"]["report_age"]["points"], 30)

    def test_most_recent_quarter_is_the_last_fallback(self):
        fundamentals = {
            "General": {"GicSector": "Industrials"},
            "Highlights": {"MostRecentQuarter": (AS_OF - timedelta(days=100)).isoformat()},
        }
        result = score(fundamentals=fundamentals)
        self.assertEqual(result["components"]["report_age"]["source"], "Highlights.MostRecentQuarter")
        self.assertEqual(result["components"]["report_age"]["points"], 20)

    def test_unresolvable_filing_date_scores_zero_and_is_flagged(self):
        fundamentals = make_fundamentals(filing_date="0000-00-00")
        result = score(fundamentals=fundamentals)
        self.assertEqual(result["components"]["report_age"]["source"], "none")
        self.assertEqual(result["components"]["report_age"]["points"], 0)
        self.assertIn("report_age_unresolvable", result["flags"])


class AuditorAndGoingConcernTests(unittest.TestCase):

    def test_auditor_tier_points(self):
        for tier, points in (("A", 15), ("B", 10), ("C", 0), ("X", -20)):
            with self.subTest(tier=tier):
                result = score(audit=make_audit(auditor_tier=tier))
                self.assertEqual(result["components"]["auditor_tier"]["points"], points)

    def test_sanctioned_auditor_is_flagged(self):
        self.assertIn("auditor_tier_X_sanctioned", score(audit=make_audit(auditor_tier="X"))["flags"])

    def test_missing_auditor_tier_scores_zero_and_is_flagged(self):
        audit = make_audit()
        del audit["auditor_tier"]
        result = score(audit=audit)
        self.assertEqual(result["components"]["auditor_tier"]["points"], 0)
        self.assertIn("auditor_tier_unverified", result["flags"])

    def test_going_concern_opinion_vetoes_and_costs_100_points(self):
        result = score(audit=make_audit(going_concern=True))
        self.assertIn("going_concern", result["vetos"])
        self.assertEqual(result["components"]["going_concern"]["points"], -100)

    def test_unknown_going_concern_is_flagged_not_vetoed(self):
        result = score(audit=make_audit(going_concern=None))
        self.assertIn("going_concern_unverified", result["flags"])
        self.assertNotIn("going_concern", result["vetos"])
        self.assertEqual(result["components"]["going_concern"]["points"], 0)


class ContractTests(unittest.TestCase):

    def test_contract_error_is_a_value_error(self):
        self.assertTrue(issubclass(DQContractError, ValueError))

    def test_missing_audit_signals_raise(self):
        with self.assertRaises(DQContractError):
            compute_dq_score(ticker="TST", fundamentals=make_fundamentals(), audit_signals=None, as_of=AS_OF)

    def test_audit_signals_must_be_a_dict(self):
        with self.assertRaises(DQContractError):
            score(audit=[("going_concern", False)])

    def test_each_required_key_is_enforced(self):
        for key in ("going_concern", "restatements_36m"):
            with self.subTest(key=key):
                audit = make_audit()
                del audit[key]
                with self.assertRaises(DQContractError):
                    score(audit=audit)

    def test_required_keys_must_have_the_documented_types(self):
        for overrides in ({"going_concern": "no"}, {"going_concern": 0},
                          {"restatements_36m": "1"}, {"restatements_36m": 1.0}):
            with self.subTest(**overrides):
                with self.assertRaises(DQContractError):
                    score(audit=make_audit(**overrides))

    def test_ticker_must_be_a_non_empty_string(self):
        for ticker in ("", None, 7):
            with self.subTest(ticker=ticker):
                with self.assertRaises(DQContractError):
                    compute_dq_score(ticker=ticker, fundamentals=make_fundamentals(),
                                     audit_signals=make_audit(), as_of=AS_OF)

    def test_mock_audit_data_is_rejected_unless_allowed(self):
        audit = make_audit(_mock_warning="synthetic")
        with self.assertRaises(DQContractError):
            score(audit=audit)
        self.assertIn("mock_audit_data", score(audit=audit, allow_mock=True)["flags"])

    def test_placeholder_audit_source_is_flagged(self):
        result = score(audit=make_audit(source="phase1_placeholder"))
        self.assertIn("audit_signals_unverified", result["flags"])


class GetDqVetosTests(unittest.TestCase):

    def test_returns_a_copy_of_the_veto_list(self):
        result = score(audit=make_audit(going_concern=True))
        vetos = get_dq_vetos(result)
        vetos.append("extra")
        self.assertEqual(result["vetos"], ["going_concern"])

    def test_missing_veto_key_gives_an_empty_list(self):
        self.assertEqual(get_dq_vetos({}), [])

    def test_non_dict_input_raises(self):
        with self.assertRaises(DQContractError):
            get_dq_vetos(["going_concern"])


def _year(values):
    return {k: f"{v:.2f}" for k, v in values.items()}


def _eodhd_two_years():
    prior = {
        "totalRevenue": 1000, "grossProfit": 400, "sellingGeneralAdministrative": 150,
        "depreciationAndAmortization": 50, "netIncome": 100,
        "netReceivables": 120, "cash": 150, "shortTermInvestments": 50, "totalCurrentAssets": 420,
        "propertyPlantAndEquipmentNet": 500, "longTermInvestments": 100, "totalAssets": 1220,
        "totalCurrentLiabilities": 150, "shortTermDebt": 30, "longTermDebt": 300,
        "totalCashFromOperatingActivities": 150,
    }
    current = {k: v * 1.05 for k, v in prior.items()}
    current["totalCashFromOperatingActivities"] = 155
    years = {"2024-12-31": current, "2023-12-31": prior}
    return {
        "General": {"Code": "TST", "GicSector": "Industrials"},
        "Financials": {
            branch: {"yearly": {fy: {**_year(v), "filing_date": FRESH_FILING} for fy, v in years.items()}}
            for branch in ("Income_Statement", "Balance_Sheet", "Cash_Flow")
        },
    }


class UnpatchedIntegrationTests(unittest.TestCase):

    def test_two_year_eodhd_dict_scores_through_the_real_calculators(self):
        # Every figure grows 5 % except operating cash flow, so all Beneish
        # indices are 1 except SGI = 1.05 and TATA = (105 - 155) / 1281:
        # M = -2.618 (clear). Sloan: accruals = 21 - 7.5 - (10 - 1.5) - 52.5
        # = -47.5 over average assets 1250.5 = -0.037985, which is Q3 here.
        result = compute_dq_score(
            ticker="TST", fundamentals=_eodhd_two_years(), audit_signals=make_audit(),
            sloan_universe=UNIVERSE, as_of=AS_OF,
        )
        self.assertAlmostEqual(result["components"]["beneish"]["m_score"], -2.618, places=3)
        self.assertAlmostEqual(result["components"]["sloan_quintile"]["ratio"], -0.037985, places=6)
        self.assertEqual(result["components"]["sloan_quintile"]["quintile"], 3)
        self.assertEqual(result["composite_score"], 100)
        self.assertEqual(result["vetos"], [])


if __name__ == "__main__":
    unittest.main()

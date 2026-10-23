"""DD-State-Machine: cooling-off and cushion edge cases.

Driven through the real state machine (no mocks):
  - the Stufe-2 cooling-off must NOT be bypassable via a Stufe-3 detour;
  - a step-down must not skip intermediate cushions on multi-tier recovery.
"""

import unittest
from datetime import datetime, timedelta, timezone

from brain.state.portfolio_state import PortfolioState
from brain.state.dd_state_machine import (
    apply_tier_transition,
    TIER_2_COOLING_OFF_HOURS,
)

_NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)


class CoolingOffBypassTests(unittest.TestCase):
    """A portfolio that enters Stufe 2 (-32er-Zone, 7d cooling-off armed), steps
    UP to Stufe 3, then recovers to a Stufe-1 drawdown BEFORE the window expires
    must be held at Stufe 2 — the Ulysses-pact survives the Stufe-3 detour.
    Gating cooling-off on the entry-tier, or clearing `cooling_off_until` on the
    step-up, would let the recovery bypass the pact."""

    def test_stage3_detour_does_not_bypass_cooling_off(self):
        t0 = datetime(2026, 6, 1, tzinfo=timezone.utc)
        state = PortfolioState(tier=0)

        # 1) enter Stufe 2 (-32er-Zone) -> cooling window armed for +7d
        apply_tier_transition(state, -33.0, now_utc=t0)
        self.assertEqual(state.tier, 2)
        self.assertIsNotNone(state.cooling_off_until)

        # 2) drawdown worsens -> step up to Stufe 3 (still inside cooling window)
        apply_tier_transition(state, -45.0, now_utc=t0 + timedelta(days=1))
        self.assertEqual(state.tier, 3)

        # 3) recovery to an unambiguous Stufe-1 drawdown (-25%), still BEFORE
        # t0+7d. -25% maps to Stufe 1 with or without the cushion walk,
        # isolating the cooling-off mechanism from the cushion-skip case.
        result = apply_tier_transition(
            state, -25.0, now_utc=t0 + timedelta(days=2)
        )

        # The 7d pact (entered at t0) is still active -> must hold at Stufe 2.
        self.assertEqual(state.tier, 2)
        self.assertTrue(result["blocked_by_cooling_off"])

    def test_recovery_after_window_expiry_steps_down(self):
        # Counter-test: once the original window has expired, the same recovery
        # IS allowed to step down (cooling-off is a delay, not a permanent lock).
        t0 = datetime(2026, 6, 1, tzinfo=timezone.utc)
        state = PortfolioState(tier=0)
        apply_tier_transition(state, -33.0, now_utc=t0)
        apply_tier_transition(state, -45.0, now_utc=t0 + timedelta(days=1))
        after = t0 + timedelta(hours=TIER_2_COOLING_OFF_HOURS + 1)
        apply_tier_transition(state, -25.0, now_utc=after)
        self.assertEqual(state.tier, 1)

    def test_cooling_off_expires_exactly_at_168h_not_one_moment_later(self):
        # Boundary the two tests above only bracket (167h held / 169h expired),
        # never hit: _cooling_off_active uses `now_utc < until` (strict) — at
        # now_utc == until EXACTLY the window is already OPEN (blocked=False),
        # the recovery is allowed. A `<` -> `<=` change (one extra hour of
        # lock) would pass every other test in this file and only fail here.
        t0 = datetime(2026, 6, 1, tzinfo=timezone.utc)
        state = PortfolioState(tier=0)
        apply_tier_transition(state, -33.0, now_utc=t0)
        apply_tier_transition(state, -45.0, now_utc=t0 + timedelta(days=1))
        exactly_at_expiry = t0 + timedelta(hours=TIER_2_COOLING_OFF_HOURS)
        result = apply_tier_transition(state, -25.0, now_utc=exactly_at_expiry)
        self.assertEqual(state.tier, 1)
        self.assertFalse(result["blocked_by_cooling_off"])


class StepDownCushionTests(unittest.TestCase):
    """A multi-tier recovery must walk down one tier at a time, each gated by
    THAT tier's 3pp cushion (lines -19 / -29 / -39)."""

    def test_stage2_to_minus20_holds_at_stage1_not_zero(self):
        # -20% sits inside Stufe-1's downward cushion line (-22 + 3 = -19): a
        # recovery above -19 is required to leave Stufe 1. So Stufe 2 -> -20%
        # must hold at Stufe 1, NOT jump to Stufe 0.
        state = PortfolioState(tier=2)
        apply_tier_transition(state, -20.0, now_utc=_NOW)
        self.assertEqual(state.tier, 1)

    def test_stage3_to_minus30_holds_at_stage2_not_stage1(self):
        # -30% is inside Stufe-2's cushion (-32 + 3 = -29): cannot leave Stufe 2.
        state = PortfolioState(tier=3, cooling_off_until=None)
        apply_tier_transition(state, -30.0, now_utc=_NOW)
        self.assertEqual(state.tier, 2)

    def test_full_recovery_clears_all_cushions_to_tier0(self):
        # Counter-test: at -5% every cushion is cleared (-5 > -19 > -29 > -39),
        # so a Stufe-3 portfolio may reach Stufe 0 in one cycle. The cushion
        # walk must NOT over-correct into a one-tier-per-cycle lock.
        state = PortfolioState(tier=3, cooling_off_until=None)
        apply_tier_transition(state, -5.0, now_utc=_NOW)
        self.assertEqual(state.tier, 0)

    def test_cushion_lines_hold_at_the_exact_boundary(self):
        # The three tests above only exercise DD values well inside/outside a
        # cushion (-20 vs -19, -30 vs -29). `cushion_line = floor + CUSHION_PP`
        # and the guard is `if dd_pct <= cushion_line: break` (non-strict) — at
        # dd_pct EXACTLY ON the line, the tier must still HOLD (not step down);
        # a `<=` -> `<` change would silently let it through here and nowhere
        # else in the suite. All three tier floors' cushion lines pinned
        # together: -22+3=-19 (tier 1), -32+3=-29 (tier 2), -42+3=-39 (tier 3).
        for start_tier, cushion_exact in ((1, -19.0), (2, -29.0), (3, -39.0)):
            state = PortfolioState(tier=start_tier, cooling_off_until=None)
            apply_tier_transition(state, cushion_exact, now_utc=_NOW)
            self.assertEqual(
                state.tier, start_tier,
                f"tier {start_tier} at its exact cushion line {cushion_exact} "
                f"must hold, got tier {state.tier}")


if __name__ == "__main__":
    unittest.main()

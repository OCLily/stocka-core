"""Direct unit-pins for the Episode-Freeze lag-gate (brain/forecast/episode_freeze.py).

An off-by-one in `window_elapsed` would shift the inclusion boundary of EVERY
calibration-relevant number by one day; tests that use wide margins (long horizons,
far-future as_of) would stay green.

Pinned here: the exact boundary (created_at + horizon_days == as_of -> True, one day
earlier -> False), horizon_days=0, and that masking clears ALL four _SEALED_COLS — a
leaked `outcome` on a masked row would be a direct anti-peeking leak.

Pure, no DB, no clock: fixed literals are safe because these call the gate functions
directly with an explicit as_of.
"""
from __future__ import annotations

import unittest

from brain.forecast.episode_freeze import _SEALED_COLS, lag_gate_rows, window_elapsed


class WindowElapsedBoundary(unittest.TestCase):
    def test_boundary_day_is_elapsed(self):
        # created 2026-05-01 + 5d == 2026-05-06 -> inclusive boundary is elapsed
        self.assertTrue(window_elapsed("2026-05-01T00:00:00Z", 5, "2026-05-06"))

    def test_one_day_before_boundary_is_not_elapsed(self):
        self.assertFalse(window_elapsed("2026-05-01T00:00:00Z", 5, "2026-05-05"))

    def test_zero_horizon_elapses_on_creation_day(self):
        self.assertTrue(window_elapsed("2026-05-01T00:00:00Z", 0, "2026-05-01"))
        self.assertFalse(window_elapsed("2026-05-01T00:00:00Z", 0, "2026-04-30"))

    def test_intraday_timestamp_does_not_move_the_boundary(self):
        # date-only truncation ([:10]): a late-in-day created_at must not shift the gate
        self.assertTrue(window_elapsed("2026-05-01T23:59:59.999999Z", 5, "2026-05-06"))
        self.assertFalse(window_elapsed("2026-05-01T23:59:59.999999Z", 5, "2026-05-05"))


def _resolved_row(created_at: str, horizon_days: int) -> dict:
    return {
        "forecast_id": 1,
        "created_at": created_at,
        "horizon_days": horizon_days,
        "status": "resolved",
        "outcome": 1,
        "resolved_at": "2026-05-02T00:00:00Z",
        "resolution_source": "manual",
        "resolution_note": "early-seal-intra-episode",
    }


class LagGateMasking(unittest.TestCase):
    def test_not_elapsed_resolved_row_is_masked_to_open_with_all_sealed_cols_cleared(self):
        row = _resolved_row("2026-05-01T00:00:00Z", 5)
        (gated,) = lag_gate_rows([row], as_of="2026-05-05")
        self.assertEqual(gated["status"], "open")
        for col in _SEALED_COLS:
            self.assertIsNone(gated[col], f"{col} leaked through the lag-gate")
        # the input row must not be mutated in place
        self.assertEqual(row["status"], "resolved")
        self.assertEqual(row["outcome"], 1)

    def test_elapsed_resolved_row_passes_through_unchanged(self):
        row = _resolved_row("2026-05-01T00:00:00Z", 5)
        (gated,) = lag_gate_rows([row], as_of="2026-05-06")
        self.assertEqual(gated, row)

    def test_open_row_is_never_touched(self):
        row = {"created_at": "2026-05-01T00:00:00Z", "horizon_days": 5, "status": "open"}
        (gated,) = lag_gate_rows([row], as_of="2026-05-05")
        self.assertEqual(gated, row)


if __name__ == "__main__":
    unittest.main()

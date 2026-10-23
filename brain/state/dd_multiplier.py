"""DD-Multiplier lookup — dormant.

Maps the DD-Stufe (0..3, see dd_state_machine) to a position-sizing multiplier:
Stufe 0 -> 1.0, Stufe 1 (-22%) -> 0.5, Stufe 2 (-32%) -> 0.0, Stufe 3 (-42%) -> 0.0.

Not wired into any sizing path, neither in this excerpt nor in the private
system: no production code calls get_dd_multiplier. The drawdown stages therefore
only escalate review depth and never stop new purchases (see dd_state_machine).
"""

from __future__ import annotations


DD_MULTIPLIER_TABLE = {
    0: 1.0,
    1: 0.5,
    2: 0.0,
    3: 0.0,
}


class DDMultiplierError(Exception):
    """Raised on an unknown tier passed to get_dd_multiplier."""


def get_dd_multiplier(tier: int) -> float:
    """Return the sizing multiplier for the given DD tier (0..3)."""
    if tier not in DD_MULTIPLIER_TABLE:
        raise DDMultiplierError(
            f"unknown DD-tier {tier!r} (expected 0..3)"
        )
    return DD_MULTIPLIER_TABLE[tier]

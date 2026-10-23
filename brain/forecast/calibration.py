"""Pure calibration metrics for forecast evaluation.

- brier_score: mean squared difference between probabilities and 0/1 outcomes.
- ece: expected calibration error over n_bins equal-width bins, the
  count-weighted mean of |accuracy - confidence|; p == 1.0 lands in the last bin.

Both validate their inputs (non-empty, equal length, p in [0, 1], outcomes in
{0, 1}) and raise ValueError otherwise. No I/O, no DB, no network. stdlib only.
"""

from __future__ import annotations

from typing import Sequence


def brier_score(probs: Sequence[float], outcomes: Sequence[int]) -> float:
    probs = list(probs)
    outcomes = list(outcomes)
    if len(probs) == 0 or len(outcomes) == 0:
        raise ValueError("probs and outcomes must not be empty")
    if len(probs) != len(outcomes):
        raise ValueError(
            f"length mismatch: probs has {len(probs)} elements, outcomes has {len(outcomes)}"
        )
    for p in probs:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"prob {p!r} is outside [0, 1]")
    for o in outcomes:
        if o not in (0, 1):
            raise ValueError(f"outcome {o!r} is not in {{0, 1}}")
    return sum((p - o) ** 2 for p, o in zip(probs, outcomes)) / len(probs)


def ece(
    probs: Sequence[float],
    outcomes: Sequence[int],
    *,
    n_bins: int = 10,
) -> float:
    probs = list(probs)
    outcomes = list(outcomes)
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if len(probs) == 0 or len(outcomes) == 0:
        raise ValueError("probs and outcomes must not be empty")
    if len(probs) != len(outcomes):
        raise ValueError(
            f"length mismatch: probs has {len(probs)} elements, outcomes has {len(outcomes)}"
        )
    for p in probs:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"prob {p!r} is outside [0, 1]")
    for o in outcomes:
        if o not in (0, 1):
            raise ValueError(f"outcome {o!r} is not in {{0, 1}}")

    bin_probs: list[list[float]] = [[] for _ in range(n_bins)]
    bin_outcomes: list[list[int]] = [[] for _ in range(n_bins)]
    for p, o in zip(probs, outcomes):
        idx = min(int(p * n_bins), n_bins - 1)
        bin_probs[idx].append(p)
        bin_outcomes[idx].append(o)

    total = len(probs)
    result = 0.0
    for b in range(n_bins):
        count = len(bin_probs[b])
        if count == 0:
            continue
        confidence = sum(bin_probs[b]) / count
        accuracy = sum(bin_outcomes[b]) / count
        weight = count / total
        result += weight * abs(accuracy - confidence)
    return result

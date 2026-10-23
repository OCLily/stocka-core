"""Shared ES/CVaR tail-size helper.

Single source of truth for "how many worst observations go into the tail
mean at confidence alpha, for a sample of size n", so that
brain/risk/risk_metrics.py and brain/risk/es_historical.py cannot compute
this number two different ways (floor+1 vs. floor) for the same metric.

Convention: floor+1. Acerbi/Tasche (2002), Prop. 4.1, use plain floor
instead. The reason for floor+1 is consistency, not a statistical claim:
both modules must share one convention, and floor+1 is the formula
risk_metrics.py's VaR/CVaR already used, so adopting it left that output
unchanged. Effect versus the textbook rule: exactly one more observation in
the tail (worst_k 19 instead of 18 at n=756/alpha=0.975, 8 instead of 7 at
alpha=0.99), so k >= 1 holds for small samples too (plain floor gives k = 0
whenever n * (1 - alpha) < 1), and the tail mean is never taken over fewer
points than the textbook rule uses.
The resulting ES/VaR values are pinned in brain/tests/test_numeric_regression.py.
"""
import math


def tail_k(n: int, alpha: float) -> int:
    """Number of worst observations for the ES/CVaR tail mean.

    k = floor(n * (1 - alpha)) + 1, clamped to [1, n].

    Equivalent to the index form `idx = floor((1-alpha)*n)`, clamped to
    `[0, n-1]`, tail = sorted[:idx+1], i.e. k = idx+1.
    """
    idx = int(math.floor((1 - alpha) * n))
    idx = max(0, min(idx, n - 1))
    return idx + 1

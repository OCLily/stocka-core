"""Stationary Bootstrap (Politis-Romano 1994) for a pre-trade drawdown simulation.

Resamples historical daily portfolio returns in random-length blocks (geometric-
distributed lengths, mean = block_len) to preserve serial dependence. Each sim
path is composed by concatenating blocks until target length is reached, then
truncated. Path's cumulative return is converted to a drawdown curve; the min
across the path is the max-drawdown for that sim. Tail metrics (P95/P99/CDaR)
over 10k sims.

The arch lib provides the primary implementation. A numpy fallback is included
for environments where arch is unavailable or convergence is suspect.

Intended use: the P95 drawdown informs drawdown-tier thresholds and validates
an aggregate risk cap. The module has no production caller in the private
system; in this excerpt the demo calls it.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


# Pre-trade pass bar: the P95 simulated max-drawdown must stay within the
# personal maximum tolerable drawdown (MTD).
P95_DD_HARD_FLOOR = -0.25


def _build_portfolio_returns(
    asset_returns: np.ndarray,
    weights: Optional[np.ndarray],
) -> np.ndarray:
    """Given (T, N) asset-returns matrix and (N,) weights, return (T,) portfolio
    returns. If asset_returns is 1-D (single-asset history), weights is ignored
    and returns are passed through.
    """
    if asset_returns.ndim == 1:
        return asset_returns.astype(np.float64, copy=False)
    if weights is None:
        n = asset_returns.shape[1]
        weights = np.ones(n) / n
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape[0] != asset_returns.shape[1]:
        raise ValueError(
            f"weights len {weights.shape[0]} != asset cols {asset_returns.shape[1]}"
        )
    return asset_returns @ weights


def _max_drawdown_from_returns(returns_path: np.ndarray) -> float:
    """Max-DD over a single simulated path. Returns NEGATIVE number (or 0 if
    monotonically up). Uses compound-equity-curve, not naive cumsum.
    """
    equity = np.cumprod(1.0 + returns_path)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(dd.min()) if dd.size else 0.0


def _block_bootstrap_fallback(
    returns: np.ndarray,
    n_sims: int,
    block_len: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pure numpy stationary-bootstrap fallback. Generates an array of max-DDs.

    For each sim: build a path of len(returns) days by concatenating geometric-
    length blocks starting at uniform-random indices (wrap-around).
    """
    T = len(returns)
    if T < 2:
        raise ValueError("returns must have at least 2 observations")
    if block_len < 1:
        raise ValueError(f"block_len must be >= 1, got {block_len}")
    p = 1.0 / block_len
    max_dds = np.empty(n_sims, dtype=np.float64)
    # KNOWN PERFORMANCE LIMITATION: this is a per-sim Python loop with numpy
    # calls in the innermost body. For the defaults (n_sims=10_000,
    # block_len=5) over a 756-day window it draws ~T/block_len = ~151 blocks
    # per path, i.e. ~1.5M block iterations across all paths -- each
    # iteration costs four calls (rng.integers, rng.geometric, np.arange,
    # fancy-index), so ~6M calls total -- plus 10_000 path allocations here
    # and four more temporary arrays per sim inside
    # _max_drawdown_from_returns (np.cumprod, np.maximum.accumulate, the
    # equity-minus-peak subtraction, the division). The path is not a corner
    # case: it also runs directly whenever use_arch=False, which is what the
    # test suite exercises.
    # The known remedy (draw all start indices and block lengths vectorized up
    # front, compute drawdowns over the full (n_sims, T) matrix) changes the
    # RNG draw ORDER and therefore the exact p95_dd/p99_dd/cdar_95plus values
    # -- and with them the passes_mtd_p95 verdict -- for any given rng_seed.
    # That is a capital-relevant output change, not a pure refactor, so it is
    # deliberately not done as a drive-by optimization. No live impact today
    # (no production caller in the private system; in this excerpt the demo
    # calls it); the loop only becomes hot once the module is wired into a
    # production path, so the fix belongs in the same pre-activation design
    # as the block_len limitation below.
    for i in range(n_sims):
        path = np.empty(T, dtype=np.float64)
        filled = 0
        while filled < T:
            start = int(rng.integers(0, T))
            blen = int(rng.geometric(p))
            blen = min(blen, T - filled)
            idx = (start + np.arange(blen)) % T
            path[filled:filled + blen] = returns[idx]
            filled += blen
        max_dds[i] = _max_drawdown_from_returns(path)
    return max_dds


def _arch_bootstrap_maxdds(
    returns: np.ndarray,
    n_sims: int,
    block_len: int,
    rng_seed: Optional[int],
) -> np.ndarray:
    """Use arch.bootstrap.StationaryBootstrap to generate max-DDs. Falls back
    on the numpy implementation if arch raises or returns unusable shapes.
    """
    try:
        from arch.bootstrap import StationaryBootstrap
    except ImportError:
        return _block_bootstrap_fallback(
            returns, n_sims, block_len, np.random.default_rng(rng_seed)
        )
    if rng_seed is not None:
        bs = StationaryBootstrap(block_len, returns, seed=rng_seed)
    else:
        bs = StationaryBootstrap(block_len, returns)
    max_dds = np.empty(n_sims, dtype=np.float64)
    i = 0
    for data, _kwargs in bs.bootstrap(n_sims):
        if i >= n_sims:
            break
        sim_returns = np.asarray(data[0]).ravel()
        max_dds[i] = _max_drawdown_from_returns(sim_returns)
        i += 1
    if i != n_sims:
        # An arch count mismatch must NOT crash the pre-trade check.
        # Specific exception so bootstrap_dd_sim can tag the fallback explicitly
        # ("numpy_fallback_after_arch_count_mismatch") instead of generic
        # "numpy_fallback" — gives ops visibility into provider-lib drift.
        raise _ArchCountMismatch(
            f"arch produced {i}/{n_sims} sims; falling back to numpy"
        )
    return max_dds


class _ArchCountMismatch(RuntimeError):
    """arch.bootstrap.StationaryBootstrap returned fewer paths than requested.

    Caught by `bootstrap_dd_sim` and routed to numpy fallback with explicit
    method-tag (capital protection: no silent crash in the pre-trade check).
    """


def bootstrap_dd_sim(
    asset_returns: Sequence[Sequence[float]] | Sequence[float],
    weights: Optional[Sequence[float]] = None,
    *,
    n_sims: int = 10_000,
    # KNOWN METHODOLOGY LIMITATION: mean block_len=5 (~1 trading
    # week) captures only short-range dependence, but max-DD tails over a
    # multi-year window are dominated by vol-regime persistence spanning
    # months (Cogneau/Zakamouline 2013; optimal block length grows ~T^(1/3),
    # Politis/White 2004 -> roughly 10-20+ for a 756-day window, not 5).
    # Resampling is also bounded by the input history's own realized
    # combinations, so a calm sample window yields a structurally optimistic
    # DD distribution. Do NOT resolve by silently changing this default --
    # the module has no production caller in the private system (in this
    # excerpt only the demo calls it); a block_len-sensitivity run
    # plus a crash-conditioned second run belongs in the pre-activation
    # design.
    block_len: int = 5,
    rng_seed: Optional[int] = None,
    use_arch: bool = True,
) -> dict:
    """Run Stationary Bootstrap on historical asset-returns; return tail metrics.

    Parameters
    ----------
    asset_returns : (T, N) matrix or (T,) vector of daily simple returns.
        For multi-asset, T = trading-days, N = assets.
    weights : (N,) portfolio weights summing to 1.0. Ignored if asset_returns
        is 1-D. Defaults to equal-weight when None.
    n_sims : number of bootstrap paths (default 10_000).
    block_len : mean block length for stationary bootstrap (default 5).
        Known limitation: may understate long-horizon DD-tail risk when the
        module is activated -- see the comment above the parameter default
        before wiring this in.
    rng_seed : optional seed for reproducibility.
    use_arch : if True, use arch.bootstrap.StationaryBootstrap (primary).
        Set False to force the numpy fallback.

    Returns
    -------
    dict with keys:
        p95_dd       : 5th-percentile of max-DDs (negative float)
        p99_dd       : 1st-percentile of max-DDs (negative float, more extreme)
        cdar_95plus  : mean of all max-DDs at or below the p95-DD threshold
        n_sims       : echo of n_sims param
        block_len    : echo of block_len param
        method       : 'arch' or 'numpy_fallback'
        passes_mtd_p95 : True iff p95_dd >= P95_DD_HARD_FLOOR (-0.25)
    """
    arr = np.asarray(asset_returns, dtype=np.float64)
    portfolio_returns = _build_portfolio_returns(arr, weights)
    if len(portfolio_returns) < 2:
        raise ValueError("need >= 2 daily portfolio returns")
    if n_sims < 100:
        raise ValueError(f"n_sims must be >= 100, got {n_sims}")
    if block_len < 1:
        raise ValueError(f"block_len must be >= 1, got {block_len}")

    method = "arch"
    if use_arch:
        try:
            max_dds = _arch_bootstrap_maxdds(
                portfolio_returns, n_sims, block_len, rng_seed
            )
        except _ArchCountMismatch:
            method = "numpy_fallback_after_arch_count_mismatch"
            rng = np.random.default_rng(rng_seed)
            max_dds = _block_bootstrap_fallback(
                portfolio_returns, n_sims, block_len, rng
            )
        except Exception:
            method = "numpy_fallback"
            rng = np.random.default_rng(rng_seed)
            max_dds = _block_bootstrap_fallback(
                portfolio_returns, n_sims, block_len, rng
            )
    else:
        method = "numpy_fallback"
        rng = np.random.default_rng(rng_seed)
        max_dds = _block_bootstrap_fallback(
            portfolio_returns, n_sims, block_len, rng
        )

    p95 = float(np.percentile(max_dds, 5))
    p99 = float(np.percentile(max_dds, 1))
    tail_mask = max_dds <= p95
    cdar = float(max_dds[tail_mask].mean()) if tail_mask.any() else p95

    return {
        "p95_dd": p95,
        "p99_dd": p99,
        "cdar_95plus": cdar,
        "n_sims": n_sims,
        "block_len": block_len,
        "method": method,
        "passes_mtd_p95": p95 >= P95_DD_HARD_FLOOR,
    }


def check_p95_pass(sim_result: dict) -> dict:
    """Pre-trade-check helper. Returns a verdict-dict for downstream
    consumers.

    Output:
        passes : bool
        reason : str ("OK" or human-readable failure message)
        p95_dd : echo
        floor  : echo of P95_DD_HARD_FLOOR
    """
    p95 = sim_result["p95_dd"]
    passes = p95 >= P95_DD_HARD_FLOOR
    reason = (
        "OK"
        if passes
        else f"P95-DD={p95:.2%} breaches MTD-floor {P95_DD_HARD_FLOOR:.0%} (personal MTD)"
    )
    return {
        "passes": passes,
        "reason": reason,
        "p95_dd": p95,
        "floor": P95_DD_HARD_FLOOR,
    }

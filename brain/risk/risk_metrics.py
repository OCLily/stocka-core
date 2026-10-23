"""
Risk Metrics Engine — CVaR/VaR computation with vol-adjustment.

Computes Value-at-Risk and Conditional VaR (Expected Shortfall) from
daily close-to-close simple returns, both naive (historical) and
volatility-adjusted using EWMA.

Konfidenz-Outputs:
- 95%   — Legacy-Backward-Compat (existing callers)
- 97.5% — Primary, FRTB-aligned (ES-97.5% als Portfolio-Constraint, Basel-Standard)
- 99%   — Hard-Cap-Level

No external dependencies beyond Python stdlib + list of prices.
"""
import math

from brain.risk._tail_k import tail_k

# ─── Constants ───────────────────────────────────────────────────

MIN_HISTORY_DAYS = 504  # ~2 years; below this => low_confidence


def _simple_returns(prices):
    """Compute daily simple returns from a list of close prices."""
    if len(prices) < 2:
        return []
    return [(prices[i] / prices[i - 1]) - 1.0 for i in range(1, len(prices))]


def _ewma_vol(returns, half_life):
    """
    Compute EWMA volatility series.
    Returns a list of same length as returns, where each entry is the
    EWMA standard deviation estimate at that point.
    """
    if not returns:
        return []

    decay = math.log(2) / half_life
    alpha = 1 - math.exp(-decay)

    # Initialize with sample variance of first `half_life` observations
    init_n = min(half_life, len(returns))
    init_mean = sum(returns[:init_n]) / init_n
    init_var = sum((r - init_mean) ** 2 for r in returns[:init_n]) / init_n
    if init_var < 1e-18:
        init_var = 1e-8

    var_est = init_var
    vols = []

    for r in returns:
        var_est = alpha * (r ** 2) + (1 - alpha) * var_est
        vols.append(math.sqrt(max(var_est, 1e-18)))

    return vols


def _vol_adjusted_returns(returns, hl_long, hl_short):
    """
    Rescale historical returns by (current_vol / historical_vol) to
    make them reflect current volatility regime.

    Uses short-term EWMA vol as "current" and long-term EWMA vol as
    each observation's "historical" vol.
    """
    if len(returns) < 2:
        return returns[:]

    long_vols = _ewma_vol(returns, hl_long)
    short_vols = _ewma_vol(returns, hl_short)

    # Current vol = last short-term EWMA estimate
    current_vol = short_vols[-1] if short_vols else 1.0

    adjusted = []
    for i, r in enumerate(returns):
        hist_vol = long_vols[i] if long_vols[i] > 1e-12 else 1e-8
        scale = current_vol / hist_vol
        # Clamp scale to avoid extreme blowup
        scale = max(0.2, min(5.0, scale))
        adjusted.append(r * scale)

    return adjusted


def _var_cvar(returns, alpha):
    """
    Compute VaR and CVaR (Expected Shortfall) at given confidence level.

    Parameters
    ----------
    returns : list of float
        Daily returns (can be raw or vol-adjusted).
    alpha : float
        Confidence level (e.g. 0.95 for 95%).

    Returns
    -------
    (var, cvar) : tuple of float
        Both expressed as positive percentages (e.g. 2.5 means 2.5% daily loss).
    """
    if not returns:
        return None, None

    sorted_rets = sorted(returns)
    n = len(sorted_rets)

    # k-convention (floor+1) comes from the shared brain/risk/_tail_k.py
    # helper, which es_historical.py's `_worst_k` uses as well, so both
    # modules size the ES/CVaR tail identically.
    k = tail_k(n, alpha)

    var_value = -sorted_rets[k - 1]  # Negate because loss is negative return

    # CVaR = mean of all returns below VaR threshold
    tail = sorted_rets[:k]
    if tail:
        cvar_value = -(sum(tail) / len(tail))
    else:
        cvar_value = var_value

    # Convert to percentage
    return round(var_value * 100, 4), round(cvar_value * 100, 4)


# ─── Public API ──────────────────────────────────────────────────

PRIMARY_CONFIDENCE = 0.975  # FRTB-aligned primary level
HARD_CAP_CONFIDENCE = 0.99  # Hard-Cap level
LEGACY_CONFIDENCE = 0.95    # backward-compat für existing callers


def _suffix(alpha):
    """Map confidence level to 3-digit output-key suffix. 0.95 -> '950', 0.975 -> '975', 0.99 -> '990'.

    Assumes alpha*1000 is integral (clean confidence levels like 0.95, 0.975, 0.99).
    """
    return str(int(round(alpha * 1000)))


def _key(stem, alpha):
    """Build output-key like 'var_naive_975' from stem='var_naive' and alpha=0.975."""
    return f"{stem}_{_suffix(alpha)}"


def compute_cvar_bundle(prices, alpha=LEGACY_CONFIDENCE, hl_long=252, hl_short=21,
                        confidences=(0.95, 0.975, 0.99)):
    """
    Compute CVaR/VaR bundle from a list of daily close prices.

    Parameters
    ----------
    prices : list of float
        Daily close prices, oldest first.
    alpha : float
        Legacy confidence level for backward-compat output keys
        (default 0.95 — keeps var_naive_950/cvar_naive_950 keys stable).
    hl_long : int
        EWMA half-life for long-term vol (default 252 = ~1 year).
    hl_short : int
        EWMA half-life for short-term vol (default 21 = ~1 month).
    confidences : tuple of float
        All confidence levels to compute (default 95% + 97.5% + 99%).
        Outputs are emitted per level with suffixed keys, e.g.
        var_naive_950, var_naive_975, var_naive_990.

    Returns
    -------
    dict
        Per confidence level (in `confidences`):
          var_naive_{lvl}, cvar_naive_{lvl},
          var_adj_{lvl},   cvar_adj_{lvl}.
        Plus:
          cvar_adj_vs_naive_rel  — relative diff at PRIMARY_CONFIDENCE
          history_days, low_confidence
          primary_confidence     — FRTB-aligned reference level
          hard_cap_confidence    — Hard-Cap level
        Returns None-filled bundle if insufficient data.
    """
    unavailable = {
        _key('var_naive', c): None for c in confidences
    }
    unavailable.update({_key('cvar_naive', c): None for c in confidences})
    unavailable.update({_key('var_adj', c): None for c in confidences})
    unavailable.update({_key('cvar_adj', c): None for c in confidences})
    unavailable.update({
        'cvar_adj_vs_naive_rel': None,
        'history_days': len(prices) if prices else 0,
        'low_confidence': True,
        'primary_confidence': PRIMARY_CONFIDENCE,
        'hard_cap_confidence': HARD_CAP_CONFIDENCE,
    })

    if not prices or len(prices) < 30:
        return unavailable

    returns = _simple_returns(prices)
    history_days = len(returns)

    adj_returns = _vol_adjusted_returns(returns, hl_long, hl_short)

    result = {
        'history_days': history_days,
        'low_confidence': history_days < MIN_HISTORY_DAYS,
        'primary_confidence': PRIMARY_CONFIDENCE,
        'hard_cap_confidence': HARD_CAP_CONFIDENCE,
    }

    # Compute VaR/CVaR for each confidence level (naive + vol-adjusted)
    cvar_naive_at_primary = None
    cvar_adj_at_primary = None

    for c in confidences:
        var_naive, cvar_naive = _var_cvar(returns, c)
        var_adj, cvar_adj = _var_cvar(adj_returns, c)
        result[_key('var_naive', c)] = var_naive
        result[_key('cvar_naive', c)] = cvar_naive
        result[_key('var_adj', c)] = var_adj
        result[_key('cvar_adj', c)] = cvar_adj
        if abs(c - PRIMARY_CONFIDENCE) < 1e-6:
            cvar_naive_at_primary = cvar_naive
            cvar_adj_at_primary = cvar_adj

    # Relative diff at PRIMARY_CONFIDENCE (FRTB-aligned reference)
    # Fall back to legacy `alpha` only if PRIMARY_CONFIDENCE not in confidences.
    if cvar_naive_at_primary is None:
        _, cvar_naive_at_primary = _var_cvar(returns, alpha)
        _, cvar_adj_at_primary = _var_cvar(adj_returns, alpha)

    if cvar_naive_at_primary and cvar_naive_at_primary > 1e-6:
        result['cvar_adj_vs_naive_rel'] = round(
            ((cvar_adj_at_primary - cvar_naive_at_primary) / cvar_naive_at_primary) * 100, 1
        )
    else:
        result['cvar_adj_vs_naive_rel'] = None

    return result

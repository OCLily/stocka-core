"""
Factor Model — PCA-based statistical factor decomposition per universe.

Entry point: `compute_idio_share_inmemory` — pure-numpy, no persistence.
Pipeline: log-return matrix → PCA on an EWM covariance → keep the first k
factors (>= 90% explained variance, capped at MAX_K) → split returns into
factor-explained and idiosyncratic parts → latest rolling idio_share
(Var(idio) / Var(total)) per ticker.

Does NOT implement: loadings persistence, varimax, procrustes, multi-year.
"""
import numpy as np

# ─── Constants ────────────────────────────────────────────────────
MIN_HISTORY_DAYS = 200       # Tickers with fewer trading days are dropped
EXPLAINED_VAR_TARGET = 0.90  # Choose k to explain >= 90%
MAX_K = 8                    # Cap number of factors

# Audit quintile: smallest sample count required to compute a meaningful
# relative idio_share quintile. Matches sloan.py convention. Below this size
# the quintile boundaries are too unstable.
_IDIO_QUINTILE_MIN_SAMPLES = 10


def classify_idio_share_quintile(idio_share, universe_samples):
    """Audit-only relative quintile for idiosyncratic-share.

    Q1 = lowest idio_share (stock follows the common factors closely).
    Q5 = highest idio_share (stock returns dominated by idiosyncratic risk).

    Returns 1..5 or None when value is None / universe too small. Boundary
    is inclusive: value == boundary[i] → quintile i (matches sloan.py).

    Math intentionally mirrors brain.scoring.sloan.compute_sloan_quintile
    (both are relative-quintile classifiers). Kept as a separate function
    because the semantic direction is opposite (Sloan-Q1=good quality,
    Idio-Q1=low idiosyncratic risk → factor-dominated returns).

    AUDIT-ONLY: no programmatic effect on composite_score or recommendation.
    A downstream orchestrator (outside this excerpt) records it alongside
    candidate entries so a later backtest can correlate the quintile with
    realized drawdowns.
    """
    if idio_share is None:
        return None
    clean = [s for s in (universe_samples or []) if isinstance(s, (int, float))]
    if len(clean) < _IDIO_QUINTILE_MIN_SAMPLES:
        return None
    s = sorted(clean)
    n = len(s)
    boundaries = [s[int(n * q)] for q in (0.2, 0.4, 0.6, 0.8)]
    for i, b in enumerate(boundaries, start=1):
        if idio_share <= b:
            return i
    return 5


def _build_return_matrix(price_dict):
    """
    Build aligned return matrix from price dict.
    Returns (returns_array [T x N], ticker_list, num_days).

    We use simple log returns. All tickers are right-aligned
    (we take the last T days where T = min common length).
    """
    if not price_dict:
        return None, [], 0

    # Find minimum length across all tickers
    min_len = min(len(v) for v in price_dict.values())
    if min_len < MIN_HISTORY_DAYS:
        return None, [], 0

    tickers = sorted(price_dict.keys())

    # Build price matrix [T x N], right-aligned to latest date
    price_matrix = np.zeros((min_len, len(tickers)))
    for j, ticker in enumerate(tickers):
        prices = price_dict[ticker]
        # Take the last min_len prices
        price_matrix[:, j] = prices[-min_len:]

    # Log returns
    with np.errstate(divide='ignore', invalid='ignore'):
        returns = np.diff(np.log(price_matrix), axis=0)

    # Replace NaN/Inf with 0
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    return returns, tickers, returns.shape[0]


def _ewm_covariance(returns, halflife):
    """
    Compute exponentially weighted covariance matrix.
    returns: [T x N] array
    halflife: decay half-life in days
    """
    T, N = returns.shape
    alpha = 1 - np.exp(-np.log(2) / halflife)

    # Compute EWM weights
    weights = np.array([(1 - alpha) ** i for i in range(T - 1, -1, -1)])
    weights /= weights.sum()

    # Weighted mean
    weighted_mean = (weights[:, None] * returns).sum(axis=0)

    # Centered returns
    centered = returns - weighted_mean[None, :]

    # Weighted covariance
    cov = (centered * weights[:, None]).T @ centered

    return cov


def _choose_k(eigenvalues, target=EXPLAINED_VAR_TARGET, max_k=MAX_K):
    """
    Choose number of factors k such that cumulative explained variance >= target.
    Cap at max_k.
    """
    total_var = eigenvalues.sum()
    if total_var <= 0:
        return 1, 0.0

    cumulative = np.cumsum(eigenvalues) / total_var
    # Find first k where cumulative >= target
    candidates = np.where(cumulative >= target)[0]
    if len(candidates) > 0:
        k = int(candidates[0]) + 1
    else:
        k = len(eigenvalues)

    k = min(k, max_k)
    explained = float(cumulative[min(k - 1, len(cumulative) - 1)])
    return k, explained


def _decompose_returns(returns, eigenvectors, k):
    """
    Decompose returns into explained + idiosyncratic components.

    returns: [T x N]
    eigenvectors: [N x N] (columns = eigenvectors, sorted by eigenvalue desc)
    k: number of factors

    Returns (explained [T x N], idio [T x N]).
    """
    # Factor loadings: first k eigenvectors
    W = eigenvectors[:, :k]  # [N x k]

    # Factor scores for each day: F = R @ W  -> [T x k]
    F = returns @ W

    # Explained returns: F @ W.T  -> [T x N]
    explained = F @ W.T

    # Idiosyncratic = total - explained
    idio = returns - explained

    return explained, idio


def _rolling_idio_share(explained, idio, window=252):
    """
    Compute rolling idio_share = Var(idio) / Var(total) over `window` days.
    Returns [T x N] array (NaN for first window-1 days).
    """
    T, N = explained.shape
    total = explained + idio  # = original returns
    result = np.full((T, N), np.nan)

    for t in range(window - 1, T):
        start = t - window + 1
        total_window = total[start:t + 1, :]
        idio_window = idio[start:t + 1, :]

        total_var = np.var(total_window, axis=0, ddof=1)
        idio_var = np.var(idio_window, axis=0, ddof=1)

        with np.errstate(divide='ignore', invalid='ignore'):
            share = np.where(total_var > 0, idio_var / total_var, np.nan)

        result[t, :] = share

    return result


# ─── In-Memory Idio-Share ─────────────────────────────────────────
#
# No persistence needed for scoring — idio-share is recomputed in-memory on
# each run from a peer-universe price-bundle.
#
# Cost: O(N²·T) for cov + O(N³) for eigh; ~1-3s for 12-50 tickers × 300d.

_FACTOR_MIN_TICKERS_INMEM = 10  # below this no factor signal — universe too thin


def compute_idio_share_inmemory(
    price_histories: dict,
    *,
    window_days: int = 252,
    cov_halflife: int = 252,
) -> tuple:
    """Pure-numpy in-memory factor pipeline. NO DB persistence.

    Args:
        price_histories: {ticker: list[float]} closes oldest-first per ticker.
        window_days: Rolling window for idio_share computation.
        cov_halflife: EWM-covariance half-life.

    Returns:
        (idio_share_map, meta) where
          idio_share_map: {ticker: float in [0,1]} — LATEST idio_share per ticker
                          (NaN-trailing-row tickers excluded)
          meta: {
            coverage_pct: float,           # % of input tickers in output
            input_count: int,
            output_count: int,
            missing_tickers: list[str],    # input tickers absent from output
            computation_seconds: float,
            reason: str,                    # populated when output empty
          }
    """
    import time as _time
    t0 = _time.monotonic()
    input_tickers = list(price_histories.keys())
    input_count = len(input_tickers)

    def _empty_meta(reason):
        return {
            "coverage_pct": 0.0,
            "input_count": input_count,
            "output_count": 0,
            "missing_tickers": list(input_tickers),
            "computation_seconds": _time.monotonic() - t0,
            "reason": reason,
        }

    if not price_histories:
        return {}, _empty_meta("empty_input")

    if input_count < _FACTOR_MIN_TICKERS_INMEM:
        return {}, _empty_meta(
            f"insufficient_universe ({input_count} < {_FACTOR_MIN_TICKERS_INMEM})"
        )

    eligible = {
        t: prices for t, prices in price_histories.items()
        if isinstance(prices, (list, tuple)) and len(prices) >= MIN_HISTORY_DAYS
    }
    if len(eligible) < _FACTOR_MIN_TICKERS_INMEM:
        return {}, _empty_meta(
            f"insufficient_after_history_filter ({len(eligible)} eligible)"
        )

    returns, valid_tickers, num_days = _build_return_matrix(eligible)
    if returns is None or num_days < MIN_HISTORY_DAYS:
        return {}, _empty_meta("return_matrix_too_short")
    # If window > num_days, _rolling_idio_share returns all-NaN → silent
    # empty output. Surface explicitly.
    if num_days < window_days:
        return {}, _empty_meta(
            f"history_shorter_than_window ({num_days}d < {window_days}d)"
        )

    cov = _ewm_covariance(returns, cov_halflife)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    eigenvalues = np.maximum(eigenvalues, 0)
    k, _explained_var = _choose_k(eigenvalues)

    explained, idio = _decompose_returns(returns, eigenvectors, k)
    idio_share = _rolling_idio_share(explained, idio, window=window_days)

    last_row = idio_share[-1]
    out: dict = {}
    for j, ticker in enumerate(valid_tickers):
        val = float(last_row[j])
        if not np.isnan(val):
            out[ticker] = val

    missing = [t for t in input_tickers if t not in out]
    coverage_pct = (100.0 * len(out) / input_count) if input_count else 0.0
    meta = {
        "coverage_pct": coverage_pct,
        "input_count": input_count,
        "output_count": len(out),
        "missing_tickers": missing,
        "computation_seconds": _time.monotonic() - t0,
        "reason": "",
    }
    return out, meta


def latest_idio_share(ticker: str, idio_share_map: dict):
    """Lookup helper. Returns float in [0,1] or None if ticker not mapped."""
    if not idio_share_map:
        return None
    return idio_share_map.get(ticker)

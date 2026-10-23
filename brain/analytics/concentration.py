"""Descriptive universe-concentration metric.

WHAT it is: how concentrated a universe's market cap is, measured on the data as-is.
  - HHI (Herfindahl-Hirschman-Index) = Σ wᵢ²  over cap-weights wᵢ, range (0, 1].
  - effective_stocks = 1 / HHI — the "effective number of stocks" (inverse Simpson).
    An equal-weighted universe of n names has effective_stocks == n; the more the
    cap piles into a few names, the lower it falls (toward 1).
  - top_n_weight = combined cap-weight of the n largest names (default top-10).

WHAT it is NOT: a signal, a forecast, a timing trigger, or a scoring input. It
never touches the screening / scoring pipeline. It is read-only and descriptive —
a sentence you could say about the universe today, nothing you would act on.

Honesty: market_cap_usd is not populated for every member. The metric is computed
over PRICED members only, and n_input / n_priced / n_missing are always reported so
the number never pretends completeness it does not have.

This module opens no sqlite connection of its own. The read-layer takes a
caller-provided `conn` whose rows are addressable by column name (sqlite3.Row).
"""
from __future__ import annotations

import sqlite3
from typing import Optional, Sequence


def concentration_metrics(
    market_caps: Sequence[Optional[float]],
    *,
    top_n: int = 10,
) -> dict:
    """Descriptive concentration of a set of market caps. Pure, DB-free.

    Args:
        market_caps: raw caps as stored (may contain None / 0 / negative — these
            are treated as un-priced and excluded, but counted as missing).
        top_n: how many of the largest names the top-weight sums over (default 10).

    Returns dict with:
        n_input, n_priced, n_missing: coverage.
        hhi: Σ wᵢ² over priced caps, or None when nothing is priced.
        effective_stocks: 1 / hhi, or None.
        top_n: the count actually summed (min(top_n, n_priced)).
        top_n_weight: combined weight of the top_n largest names (0..1), or None.
        total_market_cap_usd: Σ of priced caps (context), or None.
    """
    n_input = len(market_caps)
    priced = [c for c in market_caps if c is not None and c > 0]
    n_priced = len(priced)
    if n_priced == 0:
        return {
            "n_input": n_input,
            "n_priced": 0,
            "n_missing": n_input,
            "hhi": None,
            "effective_stocks": None,
            "top_n": 0,
            "top_n_weight": None,
            "total_market_cap_usd": None,
        }

    total = sum(priced)
    weights = [c / total for c in priced]
    hhi = sum(w * w for w in weights)
    # clamp to [0, n_priced]: a non-positive top_n means "sum the top zero", not a
    # negative slice — [:-1] would silently sum all-but-the-smallest and mislabel it.
    used_n = max(0, min(top_n, n_priced))
    top_n_weight = sum(sorted(weights, reverse=True)[:used_n])

    return {
        "n_input": n_input,
        "n_priced": n_priced,
        "n_missing": n_input - n_priced,
        "hhi": hhi,
        "effective_stocks": 1.0 / hhi,
        "top_n": used_n,
        "top_n_weight": top_n_weight,
        "total_market_cap_usd": total,
    }


def load_active_market_caps(conn: sqlite3.Connection, universe_id: str) -> list:
    """Active members' market_cap_usd for a universe (removed_at IS NULL).

    Preserves None entries — coverage is decided downstream in concentration_metrics,
    not silently dropped here. conn is caller-provided.
    """
    cur = conn.execute(
        "SELECT market_cap_usd FROM universe_membership "
        "WHERE universe_id = ? AND removed_at IS NULL",
        (universe_id,),
    )
    return [row["market_cap_usd"] for row in cur.fetchall()]


def universe_concentration(
    conn: sqlite3.Connection,
    universe_id: str,
    *,
    top_n: int = 10,
) -> dict:
    """Concentration metrics for one universe, read from universe_membership."""
    caps = load_active_market_caps(conn, universe_id)
    return {"universe_id": universe_id, **concentration_metrics(caps, top_n=top_n)}


def _de_num(value: float, decimals: int) -> str:
    """German decimal-comma formatting for a user-facing number."""
    return f"{value:.{decimals}f}".replace(".", ",")


def _pct(fraction: Optional[float]) -> str:
    """German-formatted percent from a 0..1 fraction, em-dash when absent."""
    if fraction is None:
        return "—"
    return _de_num(fraction * 100, 1) + "%"


def format_concentration_lines(result: dict) -> list[str]:
    """Pure formatter: a universe_concentration result -> descriptive text lines.

    The "descriptive, not a signal" disclaimer is carried on the surface itself, so
    the number cannot be read out of context as a signal. Coverage is always shown."""
    uid = result["universe_id"]
    lines = [
        f"Konzentration — {uid}  (deskriptiv, kein Signal/Timing)",
    ]
    if result["n_priced"] == 0:
        lines.append(
            f"  keine gepreisten Titel ({result['n_input']} Mitglieder, "
            f"0 mit gültiger Marktkapitalisierung) — keine Kennzahl berechenbar"
        )
        return lines

    lines.append(
        f"  effektive Aktienzahl (1/HHI): {_de_num(result['effective_stocks'], 1)}   "
        f"HHI: {_de_num(result['hhi'], 4)}"
    )
    lines.append(
        f"  Top-{result['top_n']}-Gewicht: {_pct(result['top_n_weight'])}"
    )
    lines.append(
        f"  Abdeckung: {result['n_priced']} von {result['n_input']} Mitgliedern "
        f"gepreist ({result['n_missing']} ohne gültige Marktkapitalisierung)"
    )
    return lines

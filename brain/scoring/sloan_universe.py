"""Sloan Universe-Wide Quintile Snapshot.

Universe-pass Sloan accruals computation so the hard-veto in DQ-Score can
reference a cross-sectional Q5 threshold instead of a fixed scalar.

Computed in-memory from fundamentals dicts the caller has already loaded
(no extra provider calls). Financials-sector tickers excluded per Sloan
spec (accruals not meaningful for banks/insurance).

Output dict schema:
  {
    "<TICKER.EXCHANGE>": <accruals_ratio: float>,   # one key per included ticker
    "quintiles": [q20, q40, q60, q80, q100],        # None if < 5 computable tickers
    "snapshot_date": "<ISO date>",
  }

Quintile convention (Sloan 1996 anomaly — HIGH accruals = Q5 = worst):
  Q1 (1) = lowest accruals = best earnings quality
  Q5 (5) = highest accruals = highest anomaly risk
"""

from __future__ import annotations

from datetime import date
from typing import Optional


_EXCLUDED_SECTORS = frozenset({"Financials"})
_MIN_SAMPLE_SIZE = 5


def compute_sloan_universe_snapshot(
    fundamentals_by_ticker: dict,
) -> dict:
    """Single-pass Sloan accruals for all non-Financial tickers.

    Returns dict with per-ticker accruals_ratio floats plus 'quintiles' and
    'snapshot_date' metadata keys.

    Tickers with missing data or Financials sector are silently skipped
    (they remain excluded from quintile computation).
    """
    from brain.scoring.sloan import compute_sloan_accruals

    snapshot_date = date.today().isoformat()
    scores: dict[str, float] = {}

    for ticker, fundamentals in (fundamentals_by_ticker or {}).items():
        sector = (fundamentals or {}).get("General", {}).get("GicSector") or ""
        if sector in _EXCLUDED_SECTORS:
            continue
        result = compute_sloan_accruals(fundamentals)
        if result["complete"] and result["accruals_ratio"] is not None:
            scores[ticker] = result["accruals_ratio"]

    # Bekannte Inkonsistenz: 'quintiles' wird berechnet, aber vom
    # Produktions-Caller (ein Orchestrator, nicht Teil dieses Auszugs) aktiv
    # herausgefiltert — die DQ-Hard-Veto-Kette klassifiziert stattdessen ueber
    # sloan.compute_sloan_quintile auf den Roh-Ratios. Zudem abweichende
    # Formel-Variante: 5 Grenzen inkl. q=1.0 mit min(int(n*q), n-1)-Clamp,
    # waehrend sloan.compute_sloan_quintile und
    # factor_model.classify_idio_share_quintile je 4 Grenzen ohne Clamp
    # rechnen. Konsolidieren oder Entfernen aendert das dokumentierte
    # Output-Schema (Modul-Docstring) — hier bewusst NICHT angefasst.
    quintiles: Optional[list] = None
    if len(scores) >= _MIN_SAMPLE_SIZE:
        sorted_scores = sorted(scores.values())
        n = len(sorted_scores)
        quintiles = [
            sorted_scores[min(int(n * q), n - 1)]
            for q in (0.2, 0.4, 0.6, 0.8, 1.0)
        ]

    return {
        **scores,
        "quintiles": quintiles,
        "snapshot_date": snapshot_date,
    }


def get_sloan_quintile_for_score(
    score: Optional[float],
    quintiles: list,
) -> Optional[int]:
    """Map a single accruals_ratio to its quintile bucket (1=best, 5=worst).

    Returns None when score is None or quintiles is None/empty.
    Q1 = accruals_ratio <= q20 boundary (lowest accruals = best quality).
    Q5 = accruals_ratio > q80 boundary (highest accruals = anomaly risk).

    Derzeit ohne Produktions-Aufrufer; bewusst als Scoring-API behalten.
    """
    if score is None or not quintiles:
        return None
    for i, boundary in enumerate(quintiles, start=1):
        if score <= boundary:
            return i
    return 5

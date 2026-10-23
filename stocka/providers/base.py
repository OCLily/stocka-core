"""The data contract between a data source and the engine.

`DataProvider` is a feeding point, not a full data abstraction. The engine
functions in `brain.scoring` (`beneish`, `sloan`, `sloan_universe`,
`dq_score`) read a fundamentals dict with the shape of the EODHD
fundamentals API response, so `fundamentals()` must return that shape:

    {
      "General":    {"Code": str, "GicSector": str, ...},
      "Highlights": {"MarketCapitalization": float, "MostRecentQuarter": "YYYY-MM-DD", ...},
      "Financials": {
        "Income_Statement": {"yearly": {"YYYY-MM-DD": {...}, ...}},
        "Balance_Sheet":    {"yearly": {"YYYY-MM-DD": {...}, ...}},
        "Cash_Flow":        {"yearly": {"YYYY-MM-DD": {...}, ...}},
      },
    }

At least two fiscal years are needed per statement; numeric values may be
strings, as in the vendor response. A source with a different schema needs
an adapter that produces this shape.

`audit_signals()` feeds `compute_dq_score`. Required keys are
`going_concern` (bool or None) and `restatements_36m` (int or None);
optional keys are `auditor_tier` ("A", "B", "C" or "X"),
`revenue_recognition_restatement` (bool) and `filing_date` ("YYYY-MM-DD").
Returning None or omitting a required key is allowed: the engine then
rejects the ticker with a contract error instead of scoring it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Ticker:
    symbol: str
    sector: str


@runtime_checkable
class DataProvider(Protocol):
    def universe(self) -> list[Ticker]:
        """Tickers with their GICS sector name, in a stable order."""
        ...

    def prices(self, symbol: str) -> list[tuple[str, float]]:
        """Daily closes as [(ISO date, close)], oldest first."""
        ...

    def fundamentals(self, symbol: str) -> dict:
        """EODHD-shaped fundamentals dict (see module docstring)."""
        ...

    def audit_signals(self, symbol: str) -> dict | None:
        """Audit-signal dict for `compute_dq_score`, or None if unknown."""
        ...

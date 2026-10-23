"""Pure gap-bounded daily-return series for single-asset ES.

Two guards make the series honest before it reaches es_historical.compute_es_historical:
  - A bad bar (NaN / non-positive adj) BREAKS the chain (prev→None) → no bridge-return
    spans it, so no NaN or phantom multi-day return can reach the tail estimate.
  - A return is only formed when <=`max_missing_bizdays` business days are missing between
    the two trade_dates: weekend Fri→Mon (0) and a single holiday (1) are included; a real
    multi-day gap (>=2 missing business days) is excluded and counted in n_excluded.

Known limit: the function only sees the bars it is given. If the upstream price store
drops NULL-close bars instead of keeping them (the author's private data layer, outside
this excerpt, does), a true single-day mid-week data hole is indistinguishable from a
holiday (1 missing business day) and is included. In the author's data a liquid US
universe showed no such holes; for illiquid/young tickers the residual distortion is
bounded to ~1 day's move. Pure function, no I/O.
"""
from __future__ import annotations

import math
from datetime import date, timedelta


def _missing_bizdays_strict(d0: str, d1: str) -> int:
    """Werktage (Mo–Fr) STRIKT zwischen zwei ISO-Daten (Endpunkte exklusiv).

    Fr→Mo = 0 (Sa/So keine Werktage), Mi→Fr über Feiertags-Do = 1, Mo→Do = 2.
    """
    a = date.fromisoformat(d0)
    b = date.fromisoformat(d1)
    return sum(1 for i in range(1, (b - a).days) if (a + timedelta(days=i)).weekday() < 5)


def bounded_returns(dated_closes, *, max_missing_bizdays: int = 1) -> dict:
    """Return {returns: list[float], n_excluded: int} from [(date, adj)] oldest-first."""
    prev = None  # (date, adj) — last USABLE bar
    returns = []
    n_excluded = 0
    for d, adj in dated_closes:
        usable = adj is not None and math.isfinite(adj) and adj > 0
        if not usable:
            prev = None  # chain breaks — never bridge a return over a bad bar
            continue
        if prev is not None:
            if _missing_bizdays_strict(prev[0], d) <= max_missing_bizdays:
                returns.append(adj / prev[1] - 1.0)
            else:
                n_excluded += 1
        prev = (d, adj)
    return {"returns": returns, "n_excluded": n_excluded}

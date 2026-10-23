"""Episode-Freeze — Lag-Gate an der Quelle. Rein, as_of-deterministisch, status-blind.

Eine log-abgeleitete Zahl reflektiert nur Forecasts, deren ursprüngliches Fenster voll
abgelaufen ist: created_at + horizon_days <= as_of. Crash-agnostisch, unabhängig vom
Early-TRUE-/Manual-Frühsiegeln (heilt die Ankunfts-Asymmetrie selbst). NUR für gerenderte +
weitergereichte Zahlen; interner forecast_log behält Early-TRUE.

Lag-basiert, nicht zustands-getriggert: ein zustands-getriggerter Freeze wäre selbst
ein 1-Bit-Leck. Die Status-Zuordnung wird gegatet — ein früh gesiegelter Forecast rendert
als *offen* bis sein Fenster abläuft (NICHT: aus der Zählung entfernt).
"""
from __future__ import annotations

from datetime import date, timedelta

# Resolution columns cleared when a not-yet-elapsed row is masked back to open, so the
# early seal (outcome/resolved_at/...) cannot leak downstream via the masked row.
_SEALED_COLS = ("outcome", "resolved_at", "resolution_source", "resolution_note")


def window_elapsed(created_at: str, horizon_days: int, as_of: str) -> bool:
    """True iff the forecast's original window has fully elapsed by as_of."""
    c = date.fromisoformat(created_at[:10])
    return c + timedelta(days=int(horizon_days)) <= date.fromisoformat(as_of[:10])


def _as_open_until_elapsed(row: dict, *, as_of: str) -> dict:
    """Row unchanged if its window has elapsed; otherwise the resolved status is masked
    back to open (resolution fields cleared) so it counts as OPEN, not resolved."""
    if row.get("status") != "resolved":
        return row
    if window_elapsed(row["created_at"], row["horizon_days"], as_of):
        return row
    masked = dict(row)
    masked["status"] = "open"
    for col in _SEALED_COLS:
        if col in masked:
            masked[col] = None
    return masked


def lag_gate_rows(rows, *, as_of: str):
    """Status-Zuordnung lag-gaten: nicht voll abgelaufene Fenster zählen als OPEN,
    unabhängig vom internen resolved-Status."""
    return [_as_open_until_elapsed(r, as_of=as_of) for r in rows]

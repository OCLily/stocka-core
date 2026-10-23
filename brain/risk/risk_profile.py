"""Pure risk-profile measures for a single asset.

REINE Risiko-Maße — annualisierte Volatilität, Downside-Deviation, längste Underwater-Phase.
KEIN Return-Reward-Maß (bewusst: Rendite im Zähler trägt Chance-Valenz, die eine reine
Risiko-Beschreibung nicht haben soll). Reused `bounded_returns` (gap-bounded). Pure
function, kein I/O, kein Verdict, kein Cross-Ticker-Vergleich, keine Schwelle/Aktion.
"""
from __future__ import annotations

import math
import statistics

from brain.risk.es_returns import bounded_returns

_SQRT252 = math.sqrt(252)
_MIN_OBS = 60              # hard floor: std*√252 on <60 returns fakes precision (~±16% SE)
_SHORT_HISTORY_N = 500     # below → "grobe Größenordnung" qualifier
_ARTIFACT_MOVE = 0.50


def _underwater_runs(closes) -> tuple:
    """(längste, aktuelle) zusammenhängende Underwater-Strecke in Bars.

    Underwater = close < laufendem Allzeithoch. Ein neues Hoch (close >= peak) beendet
    den Run. `longest` = längste solche Sequenz; `current` = die am Ende laufende Sequenz
    (0 wenn der letzte Bar ein neues Hoch ist). Leere/monoton steigende Serie → (0, 0).
    """
    # Usable-Wert-Guard analog bounded_returns (es_returns.py: `adj is not None and
    # math.isfinite(adj) and adj > 0`). Im privaten Datenlayer des Autors (SQLite,
    # nicht Teil dieses Auszugs) ist das ueber adj == 0.0 UND adj == inf erreichbar:
    # der dortige Lesepfad filtert nur `adj is None`, und SQLite speichert inf als
    # REAL. Ein unbrauchbarer Bar (nicht `isfinite(c) and c > 0`) wird
    # uebersprungen: er aktualisiert `peak` nicht und zaehlt weder als neuer Peak
    # noch als Underwater-Bar — dieselbe "Kette bricht, nie ueberbruecken"-Haltung
    # wie bounded_returns, nur dass hier die gesamte Position aus dem Fenster faellt
    # statt nur den Return. Ohne Guard wuerde ein einzelner inf-Bar `peak` dauerhaft
    # vergiften und den GESAMTEN Rest der Serie als underwater markieren
    # (`[50.0, inf, 60.0, 70.0]` -> `(2, 2)`); mit Guard wird die Serie als
    # [50.0, 60.0, 70.0] gelesen -> `(0, 0)`, monoton steigend, kein Underwater-Run.
    # NaN wird ebenso uebersprungen; None faengt der Guard nicht ab (math.isfinite
    # wirft TypeError) -- im privaten Datenlayer kommt None hier nicht an, weil
    # SQLite NaN als NULL speichert und NULL beim Lesen uebersprungen wird.
    peak = None
    longest = 0
    run = 0
    for c in closes:
        if not (math.isfinite(c) and c > 0):
            continue
        if peak is None or c >= peak:
            peak = c
            run = 0
        else:
            run += 1
            if run > longest:
                longest = run
    return longest, run


def compute_risk_profile(dated_closes) -> dict:
    """[(date_iso, adj)] oldest-first → reine Risiko-Kennzahlen.

    Jede nicht-berechenbare Zahl ist None + ehrlicher `*_note` (nie still 0).
    underwater_* bleibt bewusst 0 statt None auf leerer/monotoner Serie (kein
    Non-Computable-Fall, siehe `_underwater_runs`-Docstring); inf/0.0-Bars werden
    dort per Usable-Wert-Guard uebersprungen.
    """
    pairs = list(dated_closes)
    closes = [c for _, c in pairs]
    uw_longest, uw_current = _underwater_runs(closes)

    br = bounded_returns(pairs)
    r = br["returns"]
    n = len(r)
    n_excluded = br["n_excluded"]

    out = {
        "vol_annual": None, "vol_note": None,
        "downside_dev_annual": None, "downside_dev_note": None,
        "underwater_longest_bars": uw_longest,
        "underwater_current_bars": uw_current,
        "n": n, "n_excluded": n_excluded,
        "artifact": any(abs(x) > _ARTIFACT_MOVE for x in r),
    }

    if n < _MIN_OBS:
        note = f"nicht berechenbar (n={n})"
        out["vol_note"] = note
        out["downside_dev_note"] = note
        return out

    out["vol_annual"] = statistics.stdev(r) * _SQRT252            # ddof=1

    downs = [x for x in r if x < 0]
    if not downs:
        out["downside_dev_note"] = "keine Abwärtstage im Fenster"
    else:
        out["downside_dev_annual"] = math.sqrt(sum(x * x for x in downs) / n) * _SQRT252

    quals = []
    if n < _SHORT_HISTORY_N:
        quals.append(f"grobe Größenordnung (n={n})")
    if n_excluded:
        quals.append(f"annualisiert über lückige Serie ({n_excluded} Returns ausgeschlossen)")
    if quals:
        q = "; ".join(quals)
        out["vol_note"] = q
        if out["downside_dev_annual"] is not None:
            out["downside_dev_note"] = q

    return out

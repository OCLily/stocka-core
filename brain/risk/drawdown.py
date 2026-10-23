"""Pure level-based drawdown over a [(date, adj_close)] series.

Level-based (Peak-to-Trough auf der Kursreihe),
NICHT returns-basiert: ein fehlender Bar verschiebt weder running-max noch Tief, ein
NULL-geskippter Gap kann also keinen Phantom-Mehrtages-Return einschleusen (der
Gap-Sprung-Failure-Modus returns-basierter Rechnung existiert hier nicht). Pure
function, kein I/O. Rein erklaerend: das Ergebnis loest keine Aktion aus.

gap_count nutzt einen echten Handelskalender: eine reine Mo-Fr-Zaehlung OHNE Feiertage
laege bei einem festen 365-Tage-Fenster strukturell ~9-13 Tage zu hoch, gap_count waere
dann auch bei LUECKENLOSEN Daten > 0 (bewusst Handelskalender statt Toleranzschwelle).
_nyse_holidays_for_year berechnet die zehn NYSE-Standardfeiertage inkl. Wochenend-
Verschiebung (Sa->vorheriger Fr, So->folgender Mo) selbst — keine externe
Kalender-Bibliothek. Statisch, kein Netzwerk-Call, Funktion bleibt pur.
"""
from __future__ import annotations

import math
from datetime import date, timedelta

# |Einzeltages-Move| > 50% → sichtbar flaggen: Split/Adjustment ODER echter Crash-Tag
# (nie still schlucken, nie pauschal als "Artefakt" abtun).
_ARTIFACT_MOVE = 0.50

_MON, _TUE, _WED, _THU, _FRI, _SAT, _SUN = range(7)

# Juneteenth wurde erst 2022 zum Bundesfeiertag; NYSE schliesst seither dafuer.
_JUNETEENTH_FIRST_YEAR = 2022


def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    """n-ter Wochentag (weekday: Mo=0..So=6) eines Monats, n>=1."""
    d0 = date(year, month, 1)
    offset = (weekday - d0.weekday()) % 7
    return d0 + timedelta(days=offset + 7 * (n - 1))


def _last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    """Letzter Wochentag (weekday: Mo=0..So=6) eines Monats."""
    next_month_first = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    last_day = next_month_first - timedelta(days=1)
    offset = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=offset)


def _easter_sunday(year: int) -> date:
    """Oster-Sonntag (Gregorianisch), Anonymous-Gregorian-Algorithmus (Meeus/Jones/Butcher)."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 22 * m) // 451
    month = (h + m - 7 * n + 114) // 31
    day = ((h + m - 7 * n + 114) % 31) + 1
    return date(year, month, day)


def _observed(d: date) -> date:
    """NYSE-Wochenend-Verschiebung: faellt der Feiertag auf Sa -> vorheriger Fr,
    auf So -> folgender Mo. Werktags-Feiertage bleiben unveraendert."""
    if d.weekday() == _SAT:
        return d - timedelta(days=1)
    if d.weekday() == _SUN:
        return d + timedelta(days=1)
    return d


def _observed_new_year(d: date) -> date | None:
    """NYSE Rule 7.2 Sonderregel NUR fuer New Year's Day: faellt der 1. Januar auf
    einen Samstag, gibt es in diesem Jahr KEINEN beobachteten Feiertag -- die
    Boerse handelt am 31.12. regulaer (reale Handelstage: 2021-12-31, 2010-12-31,
    2004-12-31, 1999-12-31, jeweils Freitag vor einem Samstag-1.1.). Faellt der
    1. Januar auf einen Sonntag, gilt weiterhin die uebliche Verschiebung auf den
    folgenden Montag. Kein anderer Feiertag hat diese Ausnahme."""
    if d.weekday() == _SAT:
        return None
    return _observed(d)


def _nyse_holidays_for_year(year: int) -> set:
    """NYSE-Ganztags-Schliesstage (Standardkalender) fuer ein Kalenderjahr, inkl.
    Wochenend-Verschiebung. Keine Sonderschliessungen (z.B. Trauertage) — die sind
    nicht per fester Regel herleitbar und ausserhalb des Scopes dieser Funktion."""
    days = {
        _nth_weekday_of_month(year, 1, _MON, 3),           # Martin Luther King Jr. Day
        _nth_weekday_of_month(year, 2, _MON, 3),           # Washington's Birthday
        _easter_sunday(year) - timedelta(days=2),          # Good Friday (immer ein Freitag)
        _last_weekday_of_month(year, 5, _MON),             # Memorial Day
        _nth_weekday_of_month(year, 9, _MON, 1),           # Labor Day
        _nth_weekday_of_month(year, 11, _THU, 4),          # Thanksgiving Day
        _observed(date(year, 7, 4)),                       # Independence Day
        _observed(date(year, 12, 25)),                     # Christmas Day
    }
    new_year = _observed_new_year(date(year, 1, 1))        # New Year's Day (Sa-Ausnahme!)
    if new_year is not None:
        days.add(new_year)
    if year >= _JUNETEENTH_FIRST_YEAR:
        days.add(_observed(date(year, 6, 19)))             # Juneteenth
    return days


def _us_market_holidays(d0: date, d1: date) -> set:
    """NYSE-Feiertage (observed-Daten) im geschlossenen Intervall [d0, d1].

    year-1/year+2-Puffer als Sicherheitsmarge um Jahresgrenzen: kein Standardfeiertag
    faellt nach der Sa/So-Verschiebungsregel tatsaechlich auf ein Nachbarjahr (New
    Year's Day speziell NICHT mehr seit der Sa-Ausnahme in _observed_new_year --
    Sa-1.1. hat schlicht keinen beobachteten Tag, faellt also nie auf den 31.12. des
    Vorjahres). Puffer bleibt trotzdem als defensive Marge bestehen.
    """
    holidays = set()
    for year in range(d0.year - 1, d1.year + 2):
        holidays.update(_nyse_holidays_for_year(year))
    return {d for d in holidays if d0 <= d <= d1}


def _trading_days_inclusive(d0: str, d1: str) -> int:
    """Handelstage (Mo-Fr, ohne NYSE-Feiertage) zwischen zwei ISO-Daten, inklusive
    Endpunkte. Echter Handelskalender statt Werktags-Naeherung (Mo-Fr OHNE
    Feiertagsabzug)."""
    a = date.fromisoformat(d0)
    b = date.fromisoformat(d1)
    holidays = _us_market_holidays(a, b)
    span = (b - a).days
    return sum(
        1 for i in range(span + 1)
        if (a + timedelta(days=i)).weekday() < 5 and (a + timedelta(days=i)) not in holidays
    )


# Hinweis: Max-DD wird im Paket mehrfach berechnet, mit unterschiedlichen Konventionen:
#   (1) diese Funktion                           — level-basiert, NEGATIVES Prozent, mit Datenguard
#   (2) bootstrap_sim._max_drawdown_from_returns — returns-basiert, negativer Bruch
#   (3) backtest/metrics.compute_max_drawdown    — level-basiert wie (1), aber
#       POSITIVER Bruch; von single_asset_metrics.py genutzt
# BEWUSST NICHT vereinheitlicht: (1) vs (3) unterscheiden sich in Vorzeichen UND Massstab,
# und beide Werte erscheinen als max_dd_pct bzw. max_drawdown nebeneinander im selben
# Report — ein Angleichen aendert bereits ausgegebene Werte und ist kein Drive-by-Refactor.
# Offene Restluecke: der Datenguard unten (None/NaN/<=0 -> error, fail-closed) fehlt in
# (2) und (3).
def max_drawdown_from_closes(dated_closes) -> dict:
    """Größter Rückgang im Fenster + aktueller Rückgang, level-basiert.

    dated_closes: list[(date_iso, adj_close)] oldest-first.
    Returns dict; bei Ausfall ist `error` gesetzt und die Zahlen sind None
    (nie still 0).
    """
    pairs = list(dated_closes)
    n = len(pairs)
    out = {
        "error": None, "max_dd_pct": None, "peak_date": None, "trough_date": None,
        "current_dd_pct": None, "n_bars": n, "expected_business_days": None,
        "gap_count": None, "artifact_dates": [],
    }
    if n < 2:
        out["error"] = f"nicht berechenbar (n={n})"
        return out

    closes = [c for _, c in pairs]
    for c in closes:
        if c is None or (isinstance(c, float) and math.isnan(c)) or c <= 0:
            out["error"] = "nicht berechenbar (Datenfehler)"
            return out

    peak = closes[0]
    cur_peak_date = pairs[0][0]
    max_dd = 0.0
    mdd_peak = pairs[0][0]
    mdd_trough = pairs[0][0]
    for d, c in pairs:
        if c > peak:
            peak = c
            cur_peak_date = d
        dd = (c - peak) / peak
        if dd < max_dd:
            max_dd = dd
            mdd_peak = cur_peak_date
            mdd_trough = d
    current_dd = (closes[-1] - peak) / peak

    artifacts = [
        pairs[i][0] for i in range(1, n)
        if abs(closes[i] / closes[i - 1] - 1.0) > _ARTIFACT_MOVE
    ]

    expected = _trading_days_inclusive(pairs[0][0], pairs[-1][0])
    out.update({
        "max_dd_pct": round(max_dd * 100.0, 4),
        "peak_date": mdd_peak,
        "trough_date": mdd_trough,
        "current_dd_pct": round(current_dd * 100.0, 4),
        "expected_business_days": expected,
        "gap_count": max(0, expected - n),
        "artifact_dates": artifacts,
    })
    return out

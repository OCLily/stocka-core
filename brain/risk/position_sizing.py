"""Sizing-Grenzen S1-S3 fuer eine neue Position — reiner Rechenkern, kein I/O.

Dieses Modul rechnet nur; es liest weder State noch Kurse noch FX. Aufrufer
(nicht Teil dieses Auszugs) wenden S1-S3 vor dem Buchen bzw. am Order-Entwurf
an und reichen alle Eingaben herein. Die Grenzen sind persoenliche
Kapitalregeln des Autors, keine allgemeine Empfehlung.

DIE GRENZEN (alle drei gelten, die engste gewinnt)
  S1  RISK_PER_POSITION_MAX = 0.02
      Risiko ((Einstieg − Stop) × Stueckzahl) ≤ 2 % des Kapitals
  S2  NOTIONAL_PER_POSITION_MAX = 0.10
      Notional ≤ 10 % des Kapitals pro Position
  S3  FIRST_POSITION_MAX = 0.03, FIRST_POSITION_MIN_WARN = 0.02
      Erste Position in einem leeren Buch: 2–3 % des Kapitals — klein genug,
      dass ein Totalverlust den Hochwasserstand (HWM) um < 3 % bewegt und
      keine Drawdown-Stufe beruehrt.

SETZUNGEN (Umsetzungs-Entscheide mit Politik-Gehalt)
  1. Nenner `capital_eur` = Live-Equity des Buchs VOR dem Einstieg,
     mark-to-market, FX-konvertiert. Dieses Modul rechnet den Nenner nicht,
     der Aufrufer reicht ihn herein.
  2. S3 "leeres Buch" = zum Buchungszeitpunkt keine ANDERE offene Position im
     Buch (`book_is_empty`). NICHT gemeint: "erste Position in der
     Geschichte dieses Buchs" — diese Lesart waere nach der ersten Position
     verbraucht, S3 koennte nie wieder greifen. Folge der Setzung: S3 gilt
     jedes Mal neu, wenn das Buch flach ist.
  3. S3-Untergrenze 2 % ist WARN, kein FAIL. Kapitalschutz verlangt die
     Obergrenze; die Untergrenze soll nur sicherstellen, dass die Position
     gross genug ist, um sich echt anzufuehlen, sie schuetzt nichts.
  4. "≤" zaehlt als PASS. Genau auf der Grenze entscheidet eine relative
     Toleranz von 1e-9 gegen Gleitkomma-Rauschen — weit unter einem Cent.
  5. Fehlt eine Eingabe oder ist sie ungueltig, sind die betroffenen Regeln
     FAIL (fail-closed) — nie ein 1:1-FX, nie eine Schaetzung.
     `RuleResult.bad_input` nennt die Eingabe, damit der Aufrufer
     Infrastruktur (etwa: kein FX-Kurs) von einer Regelverletzung
     unterscheiden kann.

PFLICHTEN DES AUFRUFERS
  - `shares`, `entry_price_usd`, `stop_usd` beim Buchen aus der Position
    selbst, nicht aus deren EUR-Groessenangabe. Ein Order-Entwurf, der noch
    keine Stueckzahl kennt, leitet sie aus dem geplanten EUR-Notional ab.
  - `fx_usd_eur` = EUR je USD aus einer Referenzkurs-Quelle (z.B. FRED),
    nie ein von Hand gesetzter Kurs.
  - `book_empty` aus der aktuellen Buchsicht: ein Close im selben Lauf
    zaehlt als zu, eine vorher zugelassene Neueroeffnung im selben Lauf als
    offen.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

RISK_PER_POSITION_MAX = 0.02    # S1, Quelle: Modul-Docstring
NOTIONAL_PER_POSITION_MAX = 0.10  # S2
FIRST_POSITION_MAX = 0.03       # S3 Obergrenze (FAIL)
FIRST_POSITION_MIN_WARN = 0.02  # S3 Untergrenze (nur WARN, Setzung 3)

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
NOT_APPLICABLE = "N/A"

RULE_IDS = ("S1", "S2", "S3")

_BOUNDARY_REL_TOL = 1e-9


@dataclass(frozen=True)
class RuleResult:
    rule_id: str
    status: str
    detail: str
    measured: float | None = None
    limit: float | None = None
    bad_input: tuple[str, ...] = ()

    @property
    def blocks(self) -> bool:
        return self.status == FAIL


def check_sizing(*, entry_price_usd, stop_usd, shares, fx_usd_eur,
                 capital_eur, book_empty) -> list[RuleResult]:
    """S1, S2, S3 — immer alle drei, in dieser Reihenfolge; nur FAIL blockt."""
    bad = tuple(name for name, value in (
        ("entry_price_usd", entry_price_usd), ("shares", shares),
        ("fx_usd_eur", fx_usd_eur), ("capital_eur", capital_eur))
        if not _positive_number(value))
    if bad:
        detail = (f"nicht auswertbar — Eingabe fehlt/ungueltig: {', '.join(bad)} "
                  f"(fail-closed: kein 1:1-FX, keine Schaetzung)")
        return [RuleResult(rid, FAIL, detail, bad_input=bad) for rid in RULE_IDS]

    notional_eur = shares * entry_price_usd * fx_usd_eur
    return [
        _check_risk(entry_price_usd, stop_usd, shares, fx_usd_eur, capital_eur),
        _check_notional(notional_eur, capital_eur),
        _check_first_position(notional_eur, capital_eur, book_empty),
    ]


def book_is_empty(positions: Mapping[str, Mapping], *, ticker: str,
                  entry_date: str) -> bool:
    """Setzung 2: keine offene Position ausser der zu pruefenden (ticker, entry_date)."""
    if not ticker or not entry_date:
        raise ValueError("ticker und entry_date identifizieren die zu pruefende "
                         "Position — beide Pflicht")
    return not any(
        pos.get("status") == "open"
        and not (held == ticker and str(pos.get("entry_date")) == str(entry_date))
        for held, pos in positions.items())


def _check_risk(entry, stop, shares, fx, capital) -> RuleResult:
    if not _positive_number(stop) or stop >= entry:
        return RuleResult(
            "S1", FAIL,
            f"Risiko nicht berechenbar: stop_usd={stop!r} bei entry_price_usd="
            f"{entry!r} (long-only: 0 < stop < entry)",
            limit=RISK_PER_POSITION_MAX, bad_input=("stop_usd",))
    risk_eur = (entry - stop) * shares * fx
    ratio = risk_eur / capital
    status = FAIL if _exceeds(ratio, RISK_PER_POSITION_MAX) else PASS
    return RuleResult(
        "S1", status,
        f"Risiko {risk_eur:.2f} EUR = {ratio:.2%} von {capital:.2f} EUR "
        f"(Grenze {RISK_PER_POSITION_MAX:.0%})",
        measured=ratio, limit=RISK_PER_POSITION_MAX)


def _check_notional(notional_eur, capital) -> RuleResult:
    ratio = notional_eur / capital
    status = FAIL if _exceeds(ratio, NOTIONAL_PER_POSITION_MAX) else PASS
    return RuleResult(
        "S2", status,
        f"Notional {notional_eur:.2f} EUR = {ratio:.2%} von {capital:.2f} EUR "
        f"(Grenze {NOTIONAL_PER_POSITION_MAX:.0%})",
        measured=ratio, limit=NOTIONAL_PER_POSITION_MAX)


def _check_first_position(notional_eur, capital, book_empty) -> RuleResult:
    if not isinstance(book_empty, bool):
        return RuleResult(
            "S3", FAIL,
            f"nicht auswertbar — book_empty={book_empty!r} ist kein bool "
            f"(fail-closed)",
            limit=FIRST_POSITION_MAX, bad_input=("book_empty",))
    if not book_empty:
        return RuleResult(
            "S3", NOT_APPLICABLE,
            "Buch nicht leer — S3 gilt nur fuer die erste Position in einem "
            "leeren Buch (Setzung 2)",
            limit=FIRST_POSITION_MAX)
    ratio = notional_eur / capital
    band = f"{FIRST_POSITION_MIN_WARN:.0%}-{FIRST_POSITION_MAX:.0%}"
    measured = f"Erste Position {notional_eur:.2f} EUR = {ratio:.2%}"
    if _exceeds(ratio, FIRST_POSITION_MAX):
        return RuleResult("S3", FAIL, f"{measured} ueber {band}",
                          measured=ratio, limit=FIRST_POSITION_MAX)
    if _falls_short(ratio, FIRST_POSITION_MIN_WARN):
        return RuleResult(
            "S3", WARN, f"{measured} unter {band} — Hinweis, kein FAIL (Setzung 3)",
            measured=ratio, limit=FIRST_POSITION_MIN_WARN)
    return RuleResult("S3", PASS, f"{measured} in {band}",
                      measured=ratio, limit=FIRST_POSITION_MAX)


def _positive_number(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def _exceeds(value: float, limit: float) -> bool:
    return value > limit and not math.isclose(value, limit, rel_tol=_BOUNDARY_REL_TOL)


def _falls_short(value: float, bound: float) -> bool:
    return value < bound and not math.isclose(value, bound, rel_tol=_BOUNDARY_REL_TOL)

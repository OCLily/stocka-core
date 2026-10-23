"""ATR(20) + Hybrid-Stop (Hard 1.5× / Soft 2.0–3.75× by size, sector and biotech) + Trailing-Update.

Pure-functions on OHLC arrays + scalars. No data-provider imports.
"""

from __future__ import annotations

from typing import Iterable


ATR_WINDOW_DEFAULT = 20
# Defensive sectors — Utilities + Consumer Staples — get a tighter ATR window
# (14) AND a tighter multiplier (2.0×); the two always go together.
# Real Estate / Health Care non-Biotech intentionally NOT included — a
# conservative first cut; cap-size defaults apply.
ATR_WINDOW_DEFENSIVE = 14
SOFT_MULT_DEFENSIVE = 2.0
DEFENSIVE_GICS_SECTORS = frozenset({"Utilities", "Consumer Staples"})

LARGE_CAP_THRESHOLD = 10_000_000_000.0  # > $10B
MID_CAP_THRESHOLD = 2_000_000_000.0     # $2B-$10B

HARD_MULT = 1.5

SOFT_MULT_LARGE = 2.25
SOFT_MULT_MID = 2.75
SOFT_MULT_SMALL = 3.25
SOFT_MULT_BIOTECH = 3.75

BIOTECH_INDUSTRY_TOKENS = ("Biotech", "Pharmaceutic", "Life Sciences")
BIOTECH_GICS_SECTOR = "Health Care"

TRAILING_TRIGGER_GAIN = 0.15
TRAILING_PROFIT_LOCK_RATIO = 0.50


class StopCalcError(ValueError):
    """Stop calculation contract violation."""


def calc_atr(
    highs: Iterable[float],
    lows: Iterable[float],
    closes: Iterable[float],
    window: int = ATR_WINDOW_DEFAULT,
) -> float:
    """ATR(window) via True Range arithmetic mean (SMA) over the last `window` periods.

    True Range_t = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|).
    Requires len >= window + 1 (one prior close for the first TR).

    ABWEICHUNG VON WILDER — dokumentierte Abweichung, NICHT stillschweigend
    geaendert: Wilder (1978) definiert ATR als geglaetteten gleitenden
    Durchschnitt (RMA: atr_t = (atr_{t-1}*(w-1) + tr_t)/w), nicht als SMA ueber
    ein festes Fenster. Die Stop-Regeln schreiben nur "ATR(20)"/"ATR(14)" vor
    und legen das Glaettungsverfahren nicht fest; implementiert ist SMA.
    Numerisch verifiziert auf 60 synthetischen Bars: SMA(20)=3.1988 vs
    Wilder-RMA(20)=2.9606, rund 8 Prozent Unterschied. Da beide Stops
    proportional am ATR haengen (hard = entry - 1.5*ATR, soft = entry -
    mult*ATR), verschoebe eine Umstellung JEDE Stop-Marke um denselben Faktor.
    Das ist eine Verhaltensaenderung mit Kapitalfolge und braucht eine bewusste
    Entscheidung (Umstellung auf Wilder-RMA oder SMA als festgeschriebene
    Konvention). Bis dahin bleibt das Verhalten unveraendert; dieser Docstring
    ist die Offenlegung, damit die Abweichung nicht still bleibt.
    """
    h = list(highs)
    l = list(lows)
    c = list(closes)
    if not (len(h) == len(l) == len(c)):
        raise StopCalcError(
            f"OHLC length mismatch: highs={len(h)} lows={len(l)} closes={len(c)}"
        )
    if window < 2:
        raise StopCalcError(f"window must be >= 2, got {window}")
    if len(c) < window + 1:
        raise StopCalcError(
            f"need >= {window + 1} bars for ATR({window}), got {len(c)}"
        )

    trs: list[float] = []
    for i in range(1, len(c)):
        tr = max(
            h[i] - l[i],
            abs(h[i] - c[i - 1]),
            abs(l[i] - c[i - 1]),
        )
        trs.append(tr)
    last = trs[-window:]
    return sum(last) / float(window)


def _is_biotech(gics_sector: str | None, industry: str | None) -> bool:
    if (gics_sector or "").strip() != BIOTECH_GICS_SECTOR:
        return False
    ind = (industry or "").lower()
    return any(tok.lower() in ind for tok in BIOTECH_INDUSTRY_TOKENS)


def _is_defensive(gics_sector: str | None) -> bool:
    """True iff sector is in the canonical defensive set (Utilities, Consumer
    Staples). See ATR_WINDOW_DEFENSIVE."""
    return (gics_sector or "").strip() in DEFENSIVE_GICS_SECTORS


def get_atr_window(gics_sector: str | None) -> int:
    """Return the ATR window to use for this sector.

    Defensive (Utilities, Consumer Staples) → 14. Else → 20 default.
    None / empty / unknown sectors fall back to the default.
    """
    return ATR_WINDOW_DEFENSIVE if _is_defensive(gics_sector) else ATR_WINDOW_DEFAULT


def get_atr_multiplier(
    market_cap: float,
    gics_sector: str | None = None,
    industry: str | None = None,
) -> tuple[float, str]:
    """Return (multiplier, basis-label) for the Soft-Stop.

    Hard-Stop multiplier is invariant 1.5× (see HARD_MULT). Override precedence:
      1. Defensive (Utilities, Consumer Staples)  → 2.0×
      2. Biotech/Pharma in Health Care            → 3.75×
      3. Cap-size tiers                            → 2.25× / 2.75× / 3.25×
    """
    if market_cap is None or market_cap <= 0:
        raise StopCalcError(f"market_cap must be > 0, got {market_cap!r}")
    if _is_defensive(gics_sector):
        return SOFT_MULT_DEFENSIVE, "defensive-override"
    if _is_biotech(gics_sector, industry):
        return SOFT_MULT_BIOTECH, "biotech-override"
    if market_cap > LARGE_CAP_THRESHOLD:
        return SOFT_MULT_LARGE, "large-cap"
    if market_cap >= MID_CAP_THRESHOLD:
        return SOFT_MULT_MID, "mid-cap"
    return SOFT_MULT_SMALL, "small-cap"


def compute_hybrid_stop_from_ohlc(
    *,
    highs: Iterable[float],
    lows: Iterable[float],
    closes: Iterable[float],
    entry_price: float,
    market_cap: float,
    gics_sector: str | None = None,
    industry: str | None = None,
) -> dict:
    """Convenience: pick sector-aware ATR window, compute ATR, assemble hybrid stop.

    Wires `get_atr_window` + `calc_atr` + `compute_hybrid_stop` so callers don't
    have to know the sector-aware window selection. The standalone primitives
    remain available for callers that pre-compute ATR.
    """
    window = get_atr_window(gics_sector)
    atr = calc_atr(highs, lows, closes, window=window)
    return compute_hybrid_stop(
        entry_price=entry_price,
        atr=atr,
        market_cap=market_cap,
        gics_sector=gics_sector,
        industry=industry,
        atr_window=window,
    )


def compute_hybrid_stop(
    entry_price: float,
    atr: float,
    market_cap: float,
    gics_sector: str | None = None,
    industry: str | None = None,
    *,
    atr_window: int | None = None,
) -> dict:
    """Return {hard_stop, soft_stop, mult, basis, atr_at_entry, hard_mult, atr_window}.

    Raises StopCalcError unless the trading guardrail holds: both stops > 0
    AND soft_stop < hard_stop < entry_price.

    `atr_window` ist das Fenster, mit dem `atr` gerechnet wurde. Es geht rein
    durch, beeinflusst keine Rechnung und dient allein dem Beleg: der Aufrufer
    schreibt es in seine Stop-Begruendung, damit die Zahl im Beleg mit der
    Rechnung uebereinstimmt (statt hart "ATR(20)" zu formatieren, auch wenn
    fuer einen defensiven Sektor ATR(14) gerechnet wurde). None = Aufrufer hat
    das Fenster nicht mitgeteilt; dann steht "ATR(None)" im Beleg — sichtbar
    unbekannt ist besser als falsch belegt.

    Fail-closed: Die Defensive-Regel ist ein Paar — defensiver Sektor bekommt
    engeres Fenster (14) UND engeren Multiplikator (2.0x, siehe Kommentar bei
    ATR_WINDOW_DEFENSIVE). Ein Direktaufruf mit gics_sector="Utilities" und
    einem ATR aus einem Nicht-14-Fenster bekaeme sonst still einen
    plausiblen, aber nicht regelkonformen Soft-Stop. Deshalb wird
    `atr_window`, wenn mitgeteilt, gegen den Sektor geprueft — defensiver
    Sektor + ein Nicht-14-`atr_window` wirft `StopCalcError` statt still zu
    rechnen. `atr_window=None` (Aufrufer teilt das Fenster nicht mit) bleibt
    ungeprueft — keine Pflichtangabe, nur eine Konsistenzpruefung wenn
    angegeben. `compute_hybrid_stop_from_ohlc` waehlt das Fenster selbst
    passend zum Sektor und loest die Pruefung daher nie aus.
    """
    if entry_price is None or entry_price <= 0:
        raise StopCalcError(f"entry_price must be > 0, got {entry_price!r}")
    if atr is None or atr <= 0:
        raise StopCalcError(f"atr must be > 0, got {atr!r}")
    if (
        atr_window is not None
        and _is_defensive(gics_sector)
        and atr_window != ATR_WINDOW_DEFENSIVE
    ):
        raise StopCalcError(
            f"defensive sector {gics_sector!r} requires ATR({ATR_WINDOW_DEFENSIVE}), "
            f"got atr_window={atr_window!r} — the defensive window+multiplier pair "
            f"must not split"
        )

    mult, basis = get_atr_multiplier(market_cap, gics_sector, industry)
    hard_stop = entry_price - HARD_MULT * atr
    soft_stop = entry_price - mult * atr

    if hard_stop <= 0:
        raise StopCalcError(
            f"hard_stop <= 0 (entry={entry_price}, atr={atr}). "
            f"ATR exceeds entry/HARD_MULT — verify inputs."
        )
    if soft_stop <= 0:
        raise StopCalcError(
            f"soft_stop <= 0 (entry={entry_price}, atr={atr}, mult={mult})."
        )
    if not (soft_stop < hard_stop < entry_price):
        raise StopCalcError(
            f"stop-ordering invariant violated: "
            f"soft={soft_stop} hard={hard_stop} entry={entry_price}"
        )

    return {
        "hard_stop": round(hard_stop, 4),
        "soft_stop": round(soft_stop, 4),
        "mult": mult,
        "basis": basis,
        "atr_at_entry": round(atr, 4),
        "hard_mult": HARD_MULT,
        "atr_window": atr_window,
    }


def update_trailing_soft_stop(
    current_price: float,
    entry: float,
    current_soft: float,
) -> float:
    """Trail soft-stop upward only when gain > 15%.

    Lock at most 50% of unrealized profit. Never lowers the existing soft.
    """
    if current_price is None or entry is None or current_soft is None:
        raise StopCalcError("trailing inputs must all be defined")
    if entry <= 0 or current_price <= 0:
        raise StopCalcError("prices must be > 0")

    gain = (current_price - entry) / entry
    if gain <= TRAILING_TRIGGER_GAIN:
        return current_soft

    profit = current_price - entry
    proposed = entry + TRAILING_PROFIT_LOCK_RATIO * profit
    return round(max(current_soft, proposed), 4)

"""ES (Expected Shortfall) via Historical Simulation — 97.5%/99% × 756d lookback.

Teilweise aktiv:

  Aktiv ist compute_es_historical: im privaten System (nicht Teil dieses
  Auszugs) speist es eine On-Demand-ES-Anzeige je Ticker. Rein erklaerend, keine
  Trigger. Ein impliziter Screen ueber das ganze Universum ist bewusst nicht
  vorgesehen. In diesem Auszug ruft nur die Demo compute_es_historical auf, und
  zwar auf dem Basket.

  Nicht verdrahtet ist die Aktuator-Haelfte: check_es_block, scale_daily_to_horizon
  und die Trade-Block-Semantik der ES_TRIGGER_*-Konstanten. Bewusst NICHT aktiviert
  (das System erklaert, es handelt nicht). Im privaten System (nicht Teil dieses
  Auszugs) werden die ES_TRIGGER_*-Werte ausserhalb dieses Moduls nur als
  Anzeige-Schwelle gelesen (OK/BLOCK-Label), nicht als Aktuator.

Pure-function — caller liefert portfolio-return-series (daily, list[float], jeder
Wert = daily-return als decimal z.B. -0.01 for -1%).

Methode:
  1. Sortiere Returns aufsteigend
  2. Pick worst k Observations — k = floor(N*(1-alpha)) + 1 via der
     shared brain/risk/_tail_k.py::tail_k-Konvention (19 für N=756/alpha=0.975,
     8 für N=756/alpha=0.99). Acerbi/Tasche 2002 Prop. 4.1 verwenden floor
     ohne +1 — s. `_worst_k` docstring.
  3. ES = -mean(worst-k)  (positive % loss-magnitude)

Output ist daily-Loss als positive %; Caller skaliert zu 1M-Horizont via
sqrt(21)-Rule oder Monte-Carlo.

Nicht gebaut: EWMA-Weighting (λ=0.94) und eine DCC-GARCH-Pipeline (stattdessen
Pearson-Rolling, siehe correlation.py).
"""

from __future__ import annotations

import math
from typing import Sequence

from brain.risk._tail_k import tail_k


# KNOWN METHODOLOGY LIMITATION: ein hartes rollendes 756d-Rechteckfenster ohne
# Gewichtung ist prozyklisch -- der gemessene ES ist am kleinsten genau dann,
# wenn sich Risiko aufbaut ("low risk begets high risk", Danielsson et al.,
# Basel-Prozyklizitaets-Kritik) -- und leidet unter Cliff-/Ghost-Effekten,
# wenn alte Crash-Tage nach exakt 756 Tagen abrupt aus dem Fenster fallen
# (Pritsker 2006). Der Sortier-Schaetzer sieht keinen Verlust jenseits des
# beobachteten Fenster-Worst-Day; Vol-Clustering verletzt zudem dessen
# iid-Annahme (McNeil/Frey 2000 empfehlen GARCH-gefilterte EVT statt roher
# historischer Simulation). ES allein ist nicht elicitable (Gneiting 2011) --
# ein Backtest bräuchte das VaR-ES-Paar (Acerbi/Szekely 2014, Fissler/Ziegel
# 2016); ein solcher Backtest existiert nicht. Do NOT resolve by silently
# adding EWMA-/GARCH-Gewichtung -- das gehoert in eine bewusste Entscheidung
# vor einer Aktivierung der Aktuator-Haelfte, hier nicht implementiert
# (EWMA-Weighting/DCC-GARCH sind ohnehin nicht gebaut, s.o.).
ES_LOOKBACK_DAYS = 756            # 3 years of trading days
ES_PRIMARY_ALPHA = 0.975          # FRTB-aligned
ES_HARD_CAP_ALPHA = 0.99          # Hard-Cap-Trigger
ES_MIN_FOR_FULL_CONFIDENCE = 756  # Below → low_confidence flag
ES_MIN_OBSERVATIONS = 20          # Hard-floor: <20 returns → cannot estimate tail

# Trade-Block-Trigger (1M-Horizont, scaled). Bewusst NICHT aktiviert (s. Modul-
# Docstring). "Trade-Block" und "Hard-Stop + Auto-Cash" benennen die gemeinte
# Handlung des Investors; dieser Code fuehrt sie nicht aus und platziert keine
# Orders -- das System erklaert, es handelt nicht.
ES_TRIGGER_975_PCT = 15.0     # > 15% Portfolio-Loss → Trade-Block
ES_TRIGGER_990_PCT = 20.0     # > 20% Portfolio-Loss → Hard-Stop + Auto-Cash


class ESError(Exception):
    """Raised on invalid ES-input (e.g. too-few returns)."""


def _worst_k(alpha: float, n: int) -> int:
    """Worst-k for the ES tail mean.

    Delegates to brain/risk/_tail_k.py::tail_k, the shared floor+1 helper
    that risk_metrics.py's `_var_cvar` uses too, so both modules size the
    tail of the same metric identically. Acerbi/Tasche (2002), Prop. 4.1,
    use k = floor(N*(1-alpha)) instead; floor+1 is kept for that
    consistency (see _tail_k.py) and takes exactly one more observation
    (worst_k 19 instead of 18 / 8 instead of 7 at N=756), so the tail mean is
    never taken over fewer points than the textbook rule uses. The resulting
    values are pinned in brain/tests/test_numeric_regression.py.

    Feeds `compute_es_historical` (explain-only readout, no actuator).

    k is at least 1 (degenerate small-N -> point estimate, low_confidence-flag).
    """
    return tail_k(n, alpha)


def _es_at_alpha(sorted_returns: Sequence[float], alpha: float) -> tuple:
    """Return (var_pct, es_pct, k) — VaR and ES both as positive %.

    sorted_returns: ascending. VaR = -sorted[k-1] (the threshold loss).
    ES = -mean(sorted[:k]) (mean of worst-k tail).
    """
    n = len(sorted_returns)
    k = _worst_k(alpha, n)
    k = min(k, n)
    tail = sorted_returns[:k]
    var_value = -sorted_returns[k - 1]
    es_value = -(sum(tail) / k)
    return (var_value * 100.0, es_value * 100.0, k)


def compute_es_historical(
    returns: Sequence[float],
    *,
    lookback: int = ES_LOOKBACK_DAYS,
    primary_alpha: float = ES_PRIMARY_ALPHA,
    hard_cap_alpha: float = ES_HARD_CAP_ALPHA,
) -> dict:
    """Compute ES at primary (97.5%) and hard-cap (99%) levels via historical sim.

    Parameters
    ----------
    returns : Sequence[float]
        Daily portfolio-returns as decimals (e.g. -0.01 = -1% return).
        Last `lookback` entries are used; older entries discarded.
    lookback : int, optional
        Window length, default 756. Caller can shrink for tests.
    primary_alpha : float, optional
        Default 0.975 (FRTB-aligned).
    hard_cap_alpha : float, optional
        Default 0.99 (Hard-Cap-Trigger).

    Returns
    -------
    dict
        Keys: n_obs, lookback_used, low_confidence,
        var_975_pct, es_975_pct, worst_k_975,
        var_990_pct, es_990_pct, worst_k_990, method.
        ES + VaR as positive % daily-loss-magnitudes.
    """
    if returns is None:
        raise ESError("returns must not be None")
    returns_list = list(returns)
    if any(math.isnan(r) for r in returns_list):
        raise ESError("returns contain NaN — refusing tail estimate (fail-closed)")
    n_raw = len(returns_list)
    if n_raw < ES_MIN_OBSERVATIONS:
        raise ESError(
            f"need at least {ES_MIN_OBSERVATIONS} returns for ES, got {n_raw}"
        )

    if n_raw > lookback:
        window = returns_list[-lookback:]
    else:
        window = returns_list
    n = len(window)

    sorted_rets = sorted(window)

    var_975, es_975, k_975 = _es_at_alpha(sorted_rets, primary_alpha)
    var_990, es_990, k_990 = _es_at_alpha(sorted_rets, hard_cap_alpha)

    return {
        "n_obs": n,
        "lookback_used": min(lookback, n_raw),
        "low_confidence": n < ES_MIN_FOR_FULL_CONFIDENCE,
        "var_975_pct": round(var_975, 4),
        "es_975_pct": round(es_975, 4),
        "worst_k_975": k_975,
        "var_990_pct": round(var_990, 4),
        "es_990_pct": round(es_990, 4),
        "worst_k_990": k_990,
        "method": "historical_simulation",
    }


def scale_daily_to_horizon(daily_pct: float, horizon_days: int = 21) -> float:
    """Scale daily-loss-% to multi-day horizon via sqrt-time rule.

    The Trade-Block-Trigger is defined on a 1-Monats-Horizont (21 trading-days).
    sqrt-time-Scaling is the standard FRTB approximation.

    ACHTUNG: sqrt-t setzt iid voraus; unter Vol-Clustering und Jumps ist die
    Richtung der Verzerrung regimeabhaengig und Mehrtages-Tails koennen
    deutlich unterschaetzt werden (Danielsson/Zigrand 2006;
    McNeil/Frey/Embrechts, QRM). Vor einer Aktivierung gegen eine
    Overlapping-Window-Schaetzung pruefen. Diese Funktion hat bewusst keinen
    Produktions-Caller: die ES-Anzeige im privaten System zeigt nur Tageswerte und skaliert
    nicht auf den Monatshorizont.
    """
    if horizon_days <= 0:
        raise ESError(f"horizon_days must be positive, got {horizon_days!r}")
    return daily_pct * (horizon_days ** 0.5)


# KNOWN GAP — UNSICHERHEITS-KANAL FEHLT: check_es_block sieht nur die zwei
# ES-Werte und kann die Unsicherheit seiner eigenen Eingabe nicht sehen.
# compute_es_historical berechnet
# low_confidence (n < ES_MIN_FOR_FULL_CONFIDENCE=756) und gibt es im Dict
# zurueck — auf dem Weg hierher geht die Information strukturell verloren.
# Folge bei einer Aktivierung: bei n=ES_MIN_OBSERVATIONS=20 liefert
# _worst_k fuer alpha=0.99 k = floor(20*0.01) + 1 = 1; es_990 ist dann der
# schlechteste EINZELNE Tagesreturn, und dieser eine Tag traegt hier dieselbe
# Autoritaet wie eine 756-Tage-Schaetzung — er kann "HARD-STOP + Auto-Cash"
# ausloesen. Umgekehrt kann ein zufaellig ruhiges 20-Tage-Fenster einen echten
# Trade-Block verschlucken.
# Do NOT fix silently: der naheliegende Fix (ganzes Ergebnis-Dict statt zweier
# floats annehmen und bei low_confidence einen dritten Zustand "nicht belastbar
# schaetzbar" statt eines hard_stop/trade_block-Booleans liefern) veraendert eine
# Risiko-Entscheidungsgrenze mit Kapitalfolge. Die Funktion hat bewusst keinen
# Aufrufer (das System erklaert, es handelt nicht) — der Defekt wuerde erst bei
# einer Aktivierung wirksam, muss also VOR ihr entschieden sein.
def check_es_block(
    *,
    es_975_horizon_pct: float,
    es_990_horizon_pct: float,
    trigger_975: float = ES_TRIGGER_975_PCT,
    trigger_990: float = ES_TRIGGER_990_PCT,
) -> dict:
    """Apply the Trade-Block-Trigger to scaled (1M-horizon) ES values.

    Trade-Block if ES-97.5% > 15% portfolio loss.
    Hard-Stop + Auto-Cash if ES-99% > 20% portfolio loss.

    Not wired into any path (deliberately not activated). The function only
    returns flags and a reason string; it places no orders. If it were ever
    wired in, the flags would be a prompt to the investor, not an order.

    es_975_horizon_pct/es_990_horizon_pct are keyword-only: a silent
    argument swap here could trigger a wrong HARD-STOP decision.

    Returns dict with 'trade_block', 'hard_stop', 'reason'.
    """
    if math.isnan(es_975_horizon_pct) or math.isnan(es_990_horizon_pct):
        return {
            "trade_block": True,
            "hard_stop": True,
            "es_975_pct": es_975_horizon_pct,
            "es_990_pct": es_990_horizon_pct,
            "reason": "ES input is NaN -> fail-closed",
        }
    trade_block = es_975_horizon_pct > trigger_975
    hard_stop = es_990_horizon_pct > trigger_990
    if hard_stop:
        reason = (
            f"ES-99% (1M) = {es_990_horizon_pct:.2f}% > {trigger_990:.1f}% "
            f"→ HARD-STOP + Auto-Cash-Trigger"
        )
    elif trade_block:
        reason = (
            f"ES-97.5% (1M) = {es_975_horizon_pct:.2f}% > {trigger_975:.1f}% "
            f"→ Trade-Block"
        )
    else:
        reason = (
            f"ES OK — ES-97.5% (1M) = {es_975_horizon_pct:.2f}%, "
            f"ES-99% (1M) = {es_990_horizon_pct:.2f}%"
        )
    return {
        "trade_block": trade_block,
        "hard_stop": hard_stop,
        "es_975_pct": es_975_horizon_pct,
        "es_990_pct": es_990_horizon_pct,
        "reason": reason,
    }

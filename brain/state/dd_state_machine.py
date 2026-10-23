"""DD-State-Machine — 3 Stufen + 3pp-Cushion + Cooling-Off (7d UTC).

Pure-Python logic on a PortfolioState (`brain/state/portfolio_state.py`): reads
`tier`, `tier_entered_at`, `cooling_off_until`; mutates via `apply_transition()`
so the audit_log captures every state-change with timestamp + reason.

Regeln — Stadien zunehmender PRUEFTIEFE, nicht zunehmender Handlung; auf keiner
Stufe Verkauf, Zwangsverkauf oder Neukauf-Stopp. DD gemessen vom High-Water-Mark:
  Stufe 0  (DD <  22%):   Normal
  Stufe 1  (DD  22-32%):  Pflicht-Review der Kauf-These (ergebnisoffen)
  Stufe 2  (DD  32-42%):  + Pflicht-Bear-Case je Position, ungefragt
  Stufe 3  (DD >= 42%):   + haertester Bear-Case + Floor-Erinnerung (Cash-Reserve)

Der Anker: `state.peak_ytd` traegt den HWM (der Feldname ist historisch). Er
rastet nur nach oben, auf der EINZAHLUNGSBEREINIGTEN Kurve (adjusted =
gestempelter Kontostand - net_flows). Writer: ein manueller Reset, der
Auto-Ratchet beim Stempeln des Cash-Saldos (Persistenzschicht, nicht Teil
dieses Auszugs) und `ratchet_peak_to_measured_equity` unten, der
Mark-to-Market-Ratchet aus einer Equity-MESSUNG (nur vorwaerts, nur auf
gestempelten Eingaben).

Step-up (worsening drawdown): immediate.
Step-down (recovery): only if DD beats threshold + 3pp cushion (Re-Entry).
Stufe-2 sticky: even if DD recovers, stay at Stufe 2 until 7 days (wall-clock
UTC, = 7*24h, DST-immune) have elapsed since entering the -32%-zone. Cushion und
Cooling-off sind Anti-Whipsaw fuer das REVIEW-Staging (kein Handels-Block).

Defensive contract: dd_pct must be a finite, non-positive percent (e.g. -18.0
for -18% DD). Anything else raises DDStateError; a silently accepted sign error
would invalidate every transition downstream.

Stateful vs stateless tier-evaluation:
- `get_dd_tier_raw(dd_pct)`: pure mapping, no state, for display logic.
- `apply_tier_transition(state, dd_pct, ...)`: STATEFUL — applies cushion,
  sticky cooling-off, mutates state, audit-log. Calling `get_dd_tier_raw`
  alone does NOT persist anything, does NOT honour cooling-off, does NOT
  block flapping. The orchestrator (outside this excerpt) MUST call
  `apply_tier_transition` once per cycle, then persist the state.

Concurrency: apply_tier_transition mutates state in memory and takes no lock.
A caller that holds the state across a long run must save it with a
compare-and-swap against the version it loaded, otherwise the last writer wins.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Optional

from brain.state.portfolio_state import PortfolioState


# Stufen-Schwellen in % Drawdown vom High-Water-Mark (negativ).
# Index = Stufen-Nummer; die Stadien beginnen bei -22.
TIER_THRESHOLDS_PCT = {
    1: -22.0,
    2: -32.0,
    3: -42.0,
}

_MAX_TIER = max(TIER_THRESHOLDS_PCT)

# Re-Entry Cushion: DD must beat tier-floor by this many pp to step down.
# Prevents whipsaw flapping around a threshold.
CUSHION_PP = 3.0

# Cooling-off duration in HOURS (not days) — measured in UTC to be DST-immune.
# 7 days * 24h = 168h regardless of DST transitions. Armed on entering the
# -32%-Zone (Stufe 2).
#
# REICHWEITEN-GRENZE, bewusst unveraendert: das 168h-Fenster liegt auf derselben
# Groessenordnung wie die Abtastung. Der produktive Aufrufer von
# apply_tier_transition (ein Orchestrator, nicht Teil dieses Auszugs) laeuft
# woechentlich, ohne Auto-Trigger zwischen den Laeufen. Ein 7d-Fenster bei 7d-
# Abtastung bindet hoechstens EINEN Lauf; ob es den bindet, haengt am Jitter
# zwischen Arm-Zeitpunkt und naechstem Lauf. Cushion und Cooling-off sind fuer
# KONTINUIERLICHE DD-Messung entworfen — bei woechentlicher Punkt-Abtastung
# erledigt das Sampling selbst den groessten Teil der Anti-Whipsaw-Arbeit, die
# Schicht suggeriert also mehr Robustheit als sie heute liefert. Real wirksam
# wird sie erst bei taeglicher DD-Messung.
TIER_2_COOLING_OFF_HOURS = 7 * 24

# Stufe 0..3 action descriptors (used in transition-result + audit-log).
# Die Stufen erzwingen EINE Sache — den dokumentierten, ergebnisoffenen
# Review der Kauf-These in steigender Prueftiefe. Keine Handels-Actions.
TIER_ACTIONS = {
    0: "normal",
    1: "review_thesis",
    2: "review_bear_case",
    3: "review_bear_case_floor_reminder",
}


class DDStateError(Exception):
    """Raised on invalid DD-State-Machine input."""


def get_dd_tier_raw(dd_pct: float) -> int:
    """Map a drawdown percent to its raw tier number (0..3) ignoring all state.

    Stateless, so callers without portfolio-state can still compute the tier.

    Parameters
    ----------
    dd_pct : float
        Non-positive number, e.g. -25.0 for -25% drawdown from the HWM anchor.

    Returns
    -------
    int
        Stufe 0/1/2/3.
    """
    # Non-finite dd_pct must fail CLOSED. A NaN slips past `> 0` (NaN
    # comparisons are all False) AND every `<=` threshold below, falling
    # through to `return 0` — i.e. a broken equity curve would silently report
    # Tier 0 (full exposure, no de-risk). Reject NaN/±inf.
    if not math.isfinite(dd_pct):
        raise DDStateError(
            f"dd_pct must be a finite number (got {dd_pct!r}); a non-finite "
            "drawdown signals a broken equity curve and must NOT map to a tier"
        )
    if dd_pct > 0:
        raise DDStateError(
            f"dd_pct must be non-positive (got {dd_pct!r}); "
            "use a negative number for drawdowns (e.g. -18.0 for -18%)"
        )
    if dd_pct <= TIER_THRESHOLDS_PCT[3]:
        return 3
    if dd_pct <= TIER_THRESHOLDS_PCT[2]:
        return 2
    if dd_pct <= TIER_THRESHOLDS_PCT[1]:
        return 1
    return 0


def _target_tier_with_cushion(dd_pct: float, prev_tier: int) -> int:
    """Compute tier with step-down cushion (step-up has no cushion).

    Step-up (raw_tier > prev_tier): use raw_tier immediately. DD just got worse,
    no reason to delay the stricter regime.

    Step-down (raw_tier < prev_tier): require dd_pct above (prev_tier_floor +
    CUSHION_PP). At Stufe 1 (floor -22%), dropping to Stufe 0 needs dd_pct
    above -19% (-22 + 3). At dd_pct = -20%, we stay at Stufe 1.

    Generalized for ALL step-downs, including Stufe 3 -> 2. Without this, a
    Stufe-3 portfolio that bounces to -40% would step down on raw_tier
    alone, contradicting the cushion-rule.
    """
    raw_tier = get_dd_tier_raw(dd_pct)

    if raw_tier >= prev_tier:
        return raw_tier

    # Richtung beachten: der Step-DOWN verlangt Erholung UEBER die Cushion-Linie
    # hinaus (Floor + 3pp, z.B. -22% + 3pp = -19%). Eine Formel der Form
    # `neuer_DD <= tier_threshold - 3pp Cushion` waere in Vorzeichen UND
    # Vergleichsoperator gegenlaeufig: sie wuerde den Step-DOWN erst ausloesen,
    # wenn der Drawdown SCHLIMMER wird, und damit die Anti-Whipsaw-Mechanik
    # invertieren.
    #
    # Stepping down: walk down ONE tier at a time, each step gated by THAT
    # tier's cushion line (floor + 3pp). Checking only prev_tier's floor and
    # then jumping straight to raw_tier would let a multi-tier recovery skip
    # every intermediate cushion and de-escalate too fast. Example: Stufe 2 at
    # -20% must hold at Stufe 1, because -20% is inside Stufe-1's cushion (-19).
    tier = prev_tier
    while tier > raw_tier:
        cushion_line = TIER_THRESHOLDS_PCT[tier] + CUSHION_PP
        if dd_pct <= cushion_line:
            # Not enough recovery to leave `tier` yet.
            break
        tier -= 1
    return tier


def _parse_iso_utc(ts: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp; assume UTC if naive (migration safety).

    Returns None if input is None or empty. Raises ValueError on garbage input
    rather than silently passing wrong data downstream.
    """
    if ts in (None, ""):
        return None
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _cooling_off_active(
    cooling_off_until_iso: Optional[str],
    now_utc: datetime,
) -> bool:
    """True if cooling-off-window is still in the future."""
    until = _parse_iso_utc(cooling_off_until_iso)
    if until is None:
        return False
    return now_utc < until


def apply_tier_transition(
    state: PortfolioState,
    current_dd_pct: float,
    *,
    now_utc: Optional[datetime] = None,
) -> dict:
    """Evaluate DD against state, persist tier-change through audit-log.

    Mutates `state` in-place via `state.apply_transition(...)`. The caller is
    responsible for persisting `state`. Does NOT do the persistence itself —
    that keeps the state-machine pure-logic and lets callers batch multiple
    transitions inside one save.

    Parameters
    ----------
    state : PortfolioState
        Current state (read tier/tier_entered_at/cooling_off_until).
    current_dd_pct : float
        Current DD as non-positive percent (e.g. -18.0).
    now_utc : datetime, optional
        Injected wall-clock for testability. Defaults to datetime.now(UTC).
        Must be tz-aware.

    Returns
    -------
    dict
        {
          'old_tier': int,
          'new_tier': int,
          'tier_action': str,
          'cooling_off_until': str | None,   # ISO-UTC if active
          'blocked_by_cooling_off': bool,    # True if step-down suppressed
          'tier_entered_at': str | None,     # ISO-UTC of last entry (or pre-existing)
          'current_dd_pct': float,
        }
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        raise DDStateError("now_utc must be timezone-aware")

    old_tier = int(state.tier)
    if old_tier > _MAX_TIER:
        # Legacy-Clamp: ein persistierter State aus der frueheren
        # Vierstufen-Leiter kann tier=4 tragen. Die -42er-Zone ist heute
        # Stufe 3 — semantisch derselbe Zustand, kein Informationsverlust. Ohne
        # Clamp wuerde der Cushion-Walk unten TIER_THRESHOLDS_PCT[4]
        # subscripten -> KeyError beim ersten Lauf.
        old_tier = _MAX_TIER
    target_tier = _target_tier_with_cushion(current_dd_pct, old_tier)

    # Stufe-2 Sticky-Cooling-Off: even if DD recovers, hold at Stufe 2 until
    # cooling-off-window expires. Anti-Whipsaw fuer das Review-Staging — the
    # cushion alone is not enough.
    # The window itself is the authority, NOT `old_tier == 2`. A portfolio that entered Stufe 2, stepped UP to Stufe 3,
    # then recovered would have `old_tier == 3` and bypass the still-active pact
    # straight to Stufe 1. Gate on the window being future-dated regardless of
    # which tier we sit at right now.
    cooling_active = _cooling_off_active(state.cooling_off_until, now_utc)
    blocked_by_cooling_off = False
    if cooling_active and target_tier < 2:
        blocked_by_cooling_off = True
        target_tier = 2

    # Idempotency-Guard: if nothing stateful changed (same DD, same tier, no
    # cooling-off-action), DO NOT write an audit-log entry; otherwise the log
    # would grow linearly with every run.
    # Gegen den PERSISTIERTEN tier vergleichen, nicht die geklemmte Lokale —
    # sonst bleibt ein Legacy-tier=4 bei tiefem DD im State stehen
    # (target==geklemmt 3 -> kein Write) und crasht am Return/downstream.
    #
    # BEKANNTE LUECKE, bewusst nicht gefixt: solange das Cooling-off-Fenster
    # laeuft UND die rohe Stufe unter 2 liegt, ist `blocked_by_cooling_off` bei
    # JEDEM Aufruf True. `nothing_changed` ist dann per Konstruktion False, und
    # es wird auch bei identischem dd_pct und identischem state.tier ein
    # audit_log-Eintrag mit applied={'current_dd': <unveraendert>} geschrieben —
    # genau die Bloat-Form, die der Guard sonst verhindert. Wirkung klein
    # (7d-Fenster gegen woechentlichen Lauf -> ein bis zwei Zusatzeintraege),
    # aber das audit_log ist append-only und Leser laufen linear darueber. Nicht
    # geaendert, weil der Fix eine BEWEIS-Frage ist und keine Aufraeumarbeit:
    # unterdrueckt man den Write, taucht der aktiv gehaltene Pakt in keinem
    # Log-Eintrag mehr auf, und der Early-Return unten muesste
    # blocked_by_cooling_off weitertragen (er liefert hart False) — sichtbares
    # Verhalten fuer Aufrufer.
    tier_changed = (target_tier != int(state.tier))
    dd_changed = abs(float(current_dd_pct) - float(state.current_dd)) > 1e-9
    nothing_changed = (
        not tier_changed
        and not dd_changed
        and not blocked_by_cooling_off
    )
    if nothing_changed:
        return {
            "old_tier": old_tier,
            "new_tier": old_tier,
            "tier_action": TIER_ACTIONS[old_tier],
            "cooling_off_until": state.cooling_off_until,
            "blocked_by_cooling_off": False,
            "tier_entered_at": state.tier_entered_at,
            "current_dd_pct": float(current_dd_pct),
        }

    changes: dict = {"current_dd": float(current_dd_pct)}

    if tier_changed:
        # Tier change — refresh tier_entered_at + cooling_off_until.
        changes["tier"] = target_tier
        changes["tier_entered_at"] = now_utc.isoformat()
        # The cooling-off window is armed ONCE when crossing into
        # the danger zone (tier >= 2, die -32%-Zone) from below, and survives any
        # movement that stays inside the zone (2→3, 3→2). It is cleared only on a
        # genuine step-DOWN out of the zone (target_tier < 2), which the
        # cooling-gate above only permits once the window has expired. This
        # prevents both the 2→3→1 bypass and an unintended re-extension of the
        # window on every recovery attempt while held.
        entering_danger_from_below = old_tier < 2 and target_tier >= 2
        if entering_danger_from_below:
            until = now_utc + timedelta(hours=TIER_2_COOLING_OFF_HOURS)
            changes["cooling_off_until"] = until.isoformat()
        elif target_tier < 2 and state.cooling_off_until is not None:
            # Stepping out of the danger zone (window already expired) — clear
            # the stale value so it does not bleed across cycles.
            changes["cooling_off_until"] = None
        # else: staying inside the danger zone — preserve the existing window.

    reason = (
        f"DD-State-Machine: dd={current_dd_pct:.2f}%, "
        f"tier {old_tier} -> {target_tier}"
        + (" (cooling-off-sticky)" if blocked_by_cooling_off else "")
    )
    state.apply_transition(reason, **changes)

    return {
        "old_tier": old_tier,
        "new_tier": int(state.tier),
        "tier_action": TIER_ACTIONS[int(state.tier)],
        "cooling_off_until": state.cooling_off_until,
        "blocked_by_cooling_off": blocked_by_cooling_off,
        "tier_entered_at": state.tier_entered_at,
        "current_dd_pct": float(current_dd_pct),
    }


PEAK_MTM_RATCHET_REASON_PREFIX = "peak_mtm_ratchet"


def peak_ratchet_block_reason(state: PortfolioState, equity_result: dict) -> Optional[str]:
    """Why this measurement may NOT lift the anchor, or None if it may.

    The single source of the ratchet conditions — ratchet_peak_to_measured_equity
    applies them, a reporting caller can print them. Every condition fails
    CLOSED: a measurement that is incomplete, priced on a stale close, or
    computed against another anchor / another flow ledger than `state` carries
    never moves the HWM.
    """
    if equity_result.get("incomplete") or equity_result.get("missing_prices"):
        return "Messung unvollstaendig (Kurs fuer offene Position fehlt)"
    stale = equity_result.get("stale_tickers") or []
    if stale:
        return f"Messung auf veralteten Kursen ({', '.join(stale)})"
    adjusted = equity_result.get("adjusted_equity_eur")
    if (isinstance(adjusted, bool) or not isinstance(adjusted, (int, float))
            or not math.isfinite(adjusted)):
        return "Messung ohne bereinigte Equity"
    if equity_result.get("peak_ytd") != state.peak_ytd:
        return "Messung gegen einen anderen Anker gerechnet als der State traegt"
    if equity_result.get("net_flows_eur") != state.net_flows_eur:
        return "Messung mit einem anderen Flow-Ledger gerechnet als der State traegt"
    if not equity_result.get("above_peak") or adjusted <= state.peak_ytd:
        return (f"bereinigte Equity {adjusted:.2f} EUR liegt nicht ueber dem "
                f"Anker {state.peak_ytd:.2f} EUR")
    return None


def ratchet_peak_to_measured_equity(state: PortfolioState, equity_result: dict) -> Optional[dict]:
    """Lift state.peak_ytd to the MEASURED deposit-adjusted equity, forward only.

    The HWM ratchets only upward on the adjusted return curve. If only a
    stamped cash balance could lift it, a book that rose on prices and fell
    back without a cash stamp would be measured against a too-low anchor —
    the tier would read too mild.

    FORWARD ONLY BY CONSTRUCTION: compares ONE current measurement with the
    current anchor. Reads no audit_log, no price history, no equity history;
    rewrites nothing that already exists. A ratchet appends exactly one
    audit_log row (prefix PEAK_MTM_RATCHET_REASON_PREFIX, all inputs in the
    reason) — historical peak_ytd, current_dd and tier rows stay as stored.

    STAMP-ONLY IS THE CALLER'S DUTY: cash must be the state's stamp and FX a
    fetched reference rate (FRED), never a typed CLI value — a mistyped or
    undeclared deposit would lift the anchor for good.

    Irreversible; correction only through a manual anchor reset. Residual
    risks, accepted: a fat-fingered `shares` inside the ingest band, or a close
    outlier in the daily prices, ratchets too high — the anchor then errs
    STRICT (more reviews, never fewer). The reason line carries every input so
    such a correction is traceable.

    No tier/drawdown guard (unlike the drawdown guard on the cash-stamping
    path): above_peak means the drawdown against the CURRENT anchor is already
    0, so lifting the anchor can only make later readings stricter, never
    erase an active stage.

    Readers that compare raw equity (cash + positions) with peak_ytd measure
    UNADJUSTED, while this ratchet writes the ADJUSTED value. Both agree while
    net_flows_eur == 0.0 and diverge with the first recorded deposit.

    A higher anchor also widens the band the cash-stamping helper checks
    against peak_ytd — intended, since that band tracks the book's real size.

    Returns {"old_peak_ytd", "new_peak_ytd"} when it lifted, else None.
    """
    if peak_ratchet_block_reason(state, equity_result) is not None:
        return None
    old = float(state.peak_ytd)
    new = float(equity_result["adjusted_equity_eur"])
    state.apply_transition(
        reason=(f"{PEAK_MTM_RATCHET_REASON_PREFIX}: {new:.2f} EUR adjusted "
                f"(was {old:.2f}; equity {float(equity_result['equity_eur']):.2f}, "
                f"net_flows {float(equity_result['net_flows_eur']):.2f}, "
                f"fx_usd_eur {float(equity_result['fx_usd_eur']):.6f}, "
                f"priced_at {equity_result.get('priced_at')})"),
        peak_ytd=new,
    )
    return {"old_peak_ytd": old, "new_peak_ytd": new}


def wire_current_dd_from_fraction(state, dd_fraction, *, now_utc=None):
    """Bridge an equity-curve drawdown FRACTION (<=0) into the PERCENT tier
    ladder and apply the transition. The ×100 units-bridge is mandatory:
    TIER_THRESHOLDS_PCT is percent, the equity reader yields a fraction; without
    it a -25% drawdown (-0.25) would read as -0.25% and never leave Tier 0.
    Rounded to 4 decimals. Returns the transition result.
    """
    current_dd_pct = round(float(dd_fraction) * 100.0, 4)
    return apply_tier_transition(state, current_dd_pct, now_utc=now_utc)

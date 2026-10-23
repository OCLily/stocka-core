"""Data-Quality Score v2 — 7-component composite + Hard-Vetos.

Components: report age, auditor tier, restatements (36 months), revenue-
recognition restatement, Sloan accruals quintile, Beneish M-Score, going
concern. Composite is bounded 0..100. Hard-Vetos are live, not advisory.

`audit_signals` is REQUIRED — the caller must explicitly provide the audit
state (auditor_tier, going_concern, restatements_36m,
revenue_recognition_restatement). Sourcing these signals lives outside this
module (e.g. a full-text search over regulatory filings upstream).
Calling without this dict raises ValueError — there is NO Soft-Mode.

The `*_phase1` veto/flag/note strings are historical names from the first
build phase, kept unchanged because downstream code and tests match on them.

Pure-function — no provider imports.
"""

from __future__ import annotations

from datetime import date, datetime
from types import MappingProxyType
from typing import Any, TypedDict

from brain.scoring.beneish import (
    BENEISH_THRESHOLD_RED,
    BENEISH_THRESHOLD_GRAY_LOW,
    compute_beneish_m_score,
)
from brain.scoring.sloan import compute_sloan_accruals, compute_sloan_quintile


AUDIT_SIGNALS_REQUIRED_KEYS = {"going_concern", "restatements_36m"}
AUDIT_SIGNALS_OPTIONAL_KEYS = {
    "auditor_tier", "revenue_recognition_restatement", "filing_date",
}

AUDITOR_TIER_SCORES = {"A": 15, "B": 10, "C": 0, "X": -20}

# Sector-Bypass-Policy: zentrale Quelle für sektor-spezifische
# Beneish/Sloan-Bypass-Entscheidungen.
#
# Felder pro Sektor (TypedDict-Vertrag):
#   - sloan_bypass:   wenn True, Sloan-Quintile wird NICHT klassifiziert
#                     (Banks: deposits dominieren BS-Deltas).
#   - beneish_bypass: wenn True, Beneish-incomplete + Beneish-red führen
#                     NICHT zu Veto, sondern zu not_applicable-Flag.
#   - flag_suffix:    String für Flag-Naming
#                     ("beneish_not_applicable_<suffix>" etc.).
#
# Invariante: flag_suffix is None ⇔ both bypass-flags are False
# (no-bypass-Policy hat keinen sinnvollen suffix für f-strings).
# Wird durch _NO_BYPASS_POLICY-Definition + dieser Comment-Note encoded.
#
# Immutability: Policies sind via MappingProxyType read-only — verhindert
# accidental Mutation durch Caller, die _get_sector_policy()-Returnwerte
# weiterreichen.
#
# Unknown sectors erhalten _NO_BYPASS_POLICY via _get_sector_policy.
#
# Begründung:
#   - Hard-Vetos sind scharf; jeder Bypass ist eine explizite Ausnahme.
#   - Real-Estate-Bypass evaluiert und VERWORFEN (Beneish funktioniert bei
#     REITs als Signal).
#   - Utilities: SGA-Feld-Drift beim Provider (SGA in totalOperatingExpenses
#     aggregiert) macht Beneish dort unzuverlässig.


class _SectorPolicy(TypedDict):
    sloan_bypass: bool
    beneish_bypass: bool
    flag_suffix: str | None


_SECTOR_BYPASS_POLICY: dict[str, _SectorPolicy] = {
    "Financials": MappingProxyType({"sloan_bypass": True,  "beneish_bypass": True, "flag_suffix": "financial"}),
    "Utilities":  MappingProxyType({"sloan_bypass": False, "beneish_bypass": True, "flag_suffix": "utility"}),
}
_NO_BYPASS_POLICY: _SectorPolicy = MappingProxyType(
    {"sloan_bypass": False, "beneish_bypass": False, "flag_suffix": None}
)


def _get_sector_policy(sector: str | None) -> _SectorPolicy:
    """Return sector bypass policy or no-bypass-default for unknown sectors.

    Returned mappings are MappingProxyType — read-only views. Callers may
    only read fields, never assign (TypeError on mutation attempt).
    """
    return _SECTOR_BYPASS_POLICY.get(sector or "", _NO_BYPASS_POLICY)

REPORT_AGE_BANDS = (
    (60, 30), (120, 20), (180, 10), (10_000, 0),
)

# Sloan-Q5-Boundary-Threshold.
# accruals_ratio  > 0.05 → real anomaly territory (hard-veto "sloan_q5_anomaly")
# accruals_ratio <= 0.05 → universe-distribution boundary-effect (soft-flag
#                          "sloan_q5_boundary"; no veto, sloan_pts=-10).
# Threshold is heuristic, not lit-derived: Sloan (1996) uses universe-deciles,
# not absolute thresholds. 0.05 anchors on a natural gap observed in a
# mega/large-cap calibration snapshot (observed in the author's private data,
# not reproducible from this excerpt); round number for communicability.
# Should be re-calibrated once enough live data has accumulated.
# sloan_q5_boundary is a soft boundary flag for downstream consumers.
SLOAN_Q5_ABSOLUTE_VETO_THRESHOLD = 0.05


class DQContractError(ValueError):
    """audit_signals contract violation (Hard-Vetos are live, no Soft-Mode)."""


def _validate_audit_signals(audit_signals: dict | None) -> dict:
    if audit_signals is None:
        raise DQContractError(
            "audit_signals required: caller must provide going_concern and "
            "restatements_36m (Hard-Vetos enforced)."
        )
    if not isinstance(audit_signals, dict):
        raise DQContractError(
            f"audit_signals must be dict, got {type(audit_signals).__name__}"
        )
    missing = AUDIT_SIGNALS_REQUIRED_KEYS - set(audit_signals.keys())
    if missing:
        raise DQContractError(
            f"audit_signals missing required keys: {sorted(missing)}. "
            f"No Soft-Mode tolerated."
        )
    # None is an explicit third state (unverified — a real provider-degraded
    # signal, e.g. the upstream audit-data source was unavailable or could not
    # verify the value), not a contract violation. The None-branches below
    # (restatements + going_concern) run BEFORE any bool()/int() coercion, so
    # this None-tolerance never reaches a silent-clean or a crash.
    if audit_signals["going_concern"] is not None and not isinstance(audit_signals["going_concern"], bool):
        raise DQContractError("going_concern must be bool or None")
    if audit_signals["restatements_36m"] is not None and not isinstance(audit_signals["restatements_36m"], int):
        raise DQContractError("restatements_36m must be int or None")
    return audit_signals


def _resolve_filing_date(fundamentals: dict, audit_signals: dict) -> tuple[date | None, str]:
    raw = audit_signals.get("filing_date")
    if raw:
        try:
            return datetime.strptime(raw, "%Y-%m-%d").date(), "audit_signals.filing_date"
        except ValueError:
            pass

    inc = (fundamentals or {}).get("Financials", {}).get("Income_Statement", {}).get("yearly", {})
    if inc:
        latest_key = sorted(inc.keys(), reverse=True)[0]
        fd = inc[latest_key].get("filing_date")
        if fd and fd != "0000-00-00":
            try:
                return datetime.strptime(fd, "%Y-%m-%d").date(), "Income.filing_date"
            except ValueError:
                pass

    mrq = (fundamentals or {}).get("Highlights", {}).get("MostRecentQuarter")
    if mrq and mrq != "0000-00-00":
        try:
            return datetime.strptime(mrq, "%Y-%m-%d").date(), "Highlights.MostRecentQuarter"
        except ValueError:
            pass

    return None, "none"


def _age_points(filing_dt: date | None, as_of: date) -> tuple[int, int | None]:
    if filing_dt is None:
        return 0, None
    days = (as_of - filing_dt).days
    if days < 0:
        days = 0
    for max_days, pts in REPORT_AGE_BANDS:
        if days <= max_days:
            return pts, days
    return 0, days


def _apply_audit_source_flags(audit: dict, allow_mock: bool, flags: list[str]) -> None:
    # Mock-Guard: synthetische Audit-Daten dürfen nie unbemerkt ins Scoring;
    # ohne explizites allow_mock wird hart abgebrochen statt weitergerechnet.
    if "_mock_warning" in audit:
        if not allow_mock:
            raise DQContractError(
                "_mock_warning present: synthetic audit data must not "
                "enter scoring (pass allow_mock=True only for explicit "
                "experiments)"
            )
        flags.append("mock_audit_data")

    # Placeholder transparency, a SEPARATE channel from the mock-guard
    # above: mock = fabricated data -> hard-fail; placeholder =
    # absent-by-design (no audit source wired yet) -> flag-only, score-neutral.
    # Do not merge these flags; their fail semantics differ on purpose.
    if audit.get("source") == "phase1_placeholder":
        flags.append("audit_signals_unverified")


def _score_report_age(
    fundamentals: dict, audit: dict, as_of: date,
    components: dict[str, Any], flags: list[str],
) -> int:
    filing_dt, age_source = _resolve_filing_date(fundamentals, audit)
    age_pts, age_days = _age_points(filing_dt, as_of)
    components["report_age"] = {
        "days": age_days, "points": age_pts, "source": age_source,
        "filing_date": filing_dt.isoformat() if filing_dt else None,
    }
    # An unresolvable filing_date yields 0 age-points — numerically identical
    # to a filing older than the last REPORT_AGE_BAND. Without this flag the 30
    # forgone points read as "measured and bad" instead of "never measured".
    # Flag only, Punktelogik unverändert — symmetric to going_concern_unverified
    # / restatements_unverified. DROP-vs-KEEP stays the caller's call.
    if filing_dt is None:
        flags.append("report_age_unresolvable")
    return age_pts


def _score_auditor_tier(audit: dict, components: dict[str, Any], flags: list[str]) -> int:
    tier = audit.get("auditor_tier")
    tier_pts = AUDITOR_TIER_SCORES.get(tier, 0) if tier in AUDITOR_TIER_SCORES else 0
    components["auditor_tier"] = {"tier": tier, "points": tier_pts}
    if tier == "X":
        flags.append("auditor_tier_X_sanctioned")
    elif tier not in AUDITOR_TIER_SCORES:
        # Same reasoning: a missing key, None, or an unknown value all score
        # 0 — indistinguishable from a measured tier "C". Flag only; the 15
        # forgone A-points stay forgone.
        flags.append("auditor_tier_unverified")
    return tier_pts


def _score_restatements(
    audit: dict, composite_cap: int | None,
    components: dict[str, Any], vetos: list[str], flags: list[str],
) -> tuple[int, int | None]:
    raw_rest = audit["restatements_36m"]
    if raw_rest is None:
        # Unverified — no int(None) crash, no veto, no silent clean.
        # DROP-vs-KEEP is the caller's (screening stage) decision.
        rest_pts = 0
        components["restatements_36m"] = {"count": None, "points": 0}
        flags.append("restatements_unverified")
    else:
        # DOKUMENTIERT, NICHT GEFIXT: die Abstufung unten (rest==1 → -20,
        # rest>=2 → -100 + cap 40) ist downstream unerreichbar. Der Veto-String
        # "restatement_36m" (u.) fuehrt in der nachgelagerten Screening-Stufe
        # (nicht Teil dieses Auszugs) zum Drop, d.h. jeder Kandidat mit auch
        # nur EINEM Restatement wird gedroppt, bevor sein Composite einen
        # Consumer erreicht. Die Abstufung liest sich wie "ein Restatement ist
        # tragbar" — faktisch ist sie es nicht.
        # Ob das Veto auf rest>=2 gehoert (Abstufung wird wirksam) oder die
        # Abstufung weg soll ("jedes Restatement = Drop"), ist eine POLICY-
        # Frage (Kapitalschutz vor Rendite), kein Refactoring. Bis dahin
        # bleibt der konservative Ist-Zustand (fail-closed) bewusst unveraendert.
        rest = int(raw_rest)
        if rest <= 0:
            rest_pts = 0
        elif rest == 1:
            rest_pts = -20
        else:
            rest_pts = -100
            composite_cap = 40
        components["restatements_36m"] = {"count": rest, "points": rest_pts}
        if rest >= 1:
            vetos.append("restatement_36m")
    return rest_pts, composite_cap


def _score_revenue_recognition(audit: dict, components: dict[str, Any]) -> int:
    revrec = bool(audit.get("revenue_recognition_restatement", False))
    revrec_pts = -10 if revrec else 0
    components["revenue_recognition_restatement"] = {
        "present": revrec, "points": revrec_pts,
    }
    return revrec_pts


def _score_sloan(
    fundamentals: dict, sloan_universe: list[float] | None, sector_policy: _SectorPolicy,
    composite_cap: int | None,
    components: dict[str, Any], vetos: list[str], flags: list[str],
) -> tuple[int, int | None]:
    sloan = compute_sloan_accruals(fundamentals)
    quintile = None
    # Financial-sector bypass MUST precede quintile-classification — banks
    # have large balance-sheet swings that score as high accruals under
    # Sloan-formula but reflect normal banking operations, not earnings-
    # quality concerns (hence the explicit not_applicable_financial_sector note).
    # Without this guard, large banks get Q5-vetoed in any universe-sized
    # ≥10 run — an observed false positive in multi-sector runs (observed in
    # the author's private data, not reproducible from this excerpt).
    if not sector_policy["sloan_bypass"] and sloan["complete"] and sloan_universe:
        quintile = compute_sloan_quintile(sloan["accruals_ratio"], sloan_universe)
    if quintile is None:
        return _score_sloan_unclassified(
            sloan, sloan_universe, sector_policy, composite_cap, components, vetos, flags,
        )
    return _score_sloan_classified(sloan, quintile, components, vetos, flags), composite_cap


def _score_sloan_unclassified(
    sloan: dict, sloan_universe: list[float] | None, sector_policy: _SectorPolicy,
    composite_cap: int | None,
    components: dict[str, Any], vetos: list[str], flags: list[str],
) -> tuple[int, int | None]:
    sloan_pts = 0
    if sector_policy["sloan_bypass"]:
        suffix = sector_policy["flag_suffix"]
        components["sloan_quintile"] = {
            "ratio": sloan["accruals_ratio"], "quintile": None, "points": 0,
            "note": f"not_applicable_{suffix}_sector",
        }
        flags.append(f"sloan_not_applicable_{suffix}")
    elif sloan_universe is None:
        # No universe snapshot supplied (known gap: the caller wires no
        # snapshot path; outside this excerpt). Soft-flag
        # without veto for non-financial — caller must surface in render.
        components["sloan_quintile"] = {
            "ratio": sloan["accruals_ratio"], "quintile": None, "points": 0,
            "note": "universe_snapshot_deferred_phase1",
        }
        flags.append("sloan_quintile_unavailable")
    else:
        # Universe provided but sloan-inputs partial = suspicious in non-financial.
        components["sloan_quintile"] = {
            "ratio": sloan["accruals_ratio"], "quintile": None, "points": 0,
            "note": "incomplete_phase1_veto",
        }
        flags.append("sloan_incomplete")
        vetos.append("sloan_data_incomplete_phase1")
        composite_cap = min(composite_cap, 30) if composite_cap is not None else 30
    return sloan_pts, composite_cap


def _score_sloan_classified(
    sloan: dict, quintile: int,
    components: dict[str, Any], vetos: list[str], flags: list[str],
) -> int:
    ratio = sloan["accruals_ratio"]
    sloan_component: dict[str, Any] = {"ratio": ratio, "quintile": quintile}
    if quintile == 5:
        # Boundary-vs-Anomaly split. Relative-Q5 with absolute
        # ratio below SLOAN_Q5_ABSOLUTE_VETO_THRESHOLD is a universe-skew
        # boundary-effect, not an earnings-quality anomaly. Hard-veto only
        # when both relative AND absolute signals agree (ratio > threshold).
        # <= semantics: exact threshold counts as boundary, not anomaly.
        if ratio > SLOAN_Q5_ABSOLUTE_VETO_THRESHOLD:
            sloan_pts = -30
            vetos.append("sloan_q5_anomaly")
            sloan_component["classification"] = "anomaly"
        else:
            sloan_pts = -10
            flags.append("sloan_q5_boundary")
            sloan_component["classification"] = "boundary"
        sloan_component["absolute_threshold"] = SLOAN_Q5_ABSOLUTE_VETO_THRESHOLD
    else:
        sloan_pts = 5
    sloan_component["points"] = sloan_pts
    components["sloan_quintile"] = sloan_component
    return sloan_pts


def _score_beneish(
    fundamentals: dict, sector_policy: _SectorPolicy, composite_cap: int | None,
    components: dict[str, Any], vetos: list[str], flags: list[str],
) -> tuple[int, int | None]:
    beneish = compute_beneish_m_score(fundamentals)
    if not beneish["complete"]:
        return _score_beneish_incomplete(
            beneish, sector_policy, composite_cap, components, vetos, flags,
        )
    return _score_beneish_complete(
        beneish, sector_policy, composite_cap, components, vetos, flags,
    )


def _score_beneish_incomplete(
    beneish: dict, sector_policy: _SectorPolicy, composite_cap: int | None,
    components: dict[str, Any], vetos: list[str], flags: list[str],
) -> tuple[int, int | None]:
    beneish_pts = 0
    if sector_policy["beneish_bypass"]:
        suffix = sector_policy["flag_suffix"]
        components["beneish"] = {
            "m_score": None, "flag": f"not_applicable_{suffix}_sector", "points": 0,
            "missing_inputs": beneish["missing_inputs"],
        }
        flags.append(f"beneish_not_applicable_{suffix}")
    else:
        components["beneish"] = {
            "m_score": None, "flag": "incomplete_phase1_veto", "points": 0,
            "missing_inputs": beneish["missing_inputs"],
        }
        flags.append("beneish_incomplete")
        vetos.append("beneish_data_incomplete_phase1")
        composite_cap = min(composite_cap, 30) if composite_cap is not None else 30
    return beneish_pts, composite_cap


def _score_beneish_complete(
    beneish: dict, sector_policy: _SectorPolicy, composite_cap: int | None,
    components: dict[str, Any], vetos: list[str], flags: list[str],
) -> tuple[int, int | None]:
    m = beneish["m_score"]
    if m < BENEISH_THRESHOLD_GRAY_LOW:
        beneish_pts = 30
    elif m <= BENEISH_THRESHOLD_RED:
        beneish_pts = 5
    elif sector_policy["beneish_bypass"]:
        # Sektor-Bypass für Beneish-Red — Financials + Utilities haben
        # structural balance-sheet/cost-structure-Patterns die Beneish
        # 1999 industrial-calibrated Heuristik falsch positiv triggern
        # (Banks: deposits/AFS-Securities/trading-book; Utilities:
        # regulated-revenue + large CapEx-D&A). REITs explizit
        # ausgenommen (Beneish funktioniert dort als Signal).
        beneish_pts = 5
        flags.append(f"beneish_red_not_applicable_{sector_policy['flag_suffix']}")
    else:
        beneish_pts = 0
        composite_cap = min(composite_cap, 30) if composite_cap is not None else 30
        vetos.append("beneish_red_flag")
    components["beneish"] = {
        "m_score": m, "flag": beneish["flag"], "points": beneish_pts,
    }
    return beneish_pts, composite_cap


def _score_going_concern(
    audit: dict, components: dict[str, Any], vetos: list[str], flags: list[str],
) -> int:
    raw_gc = audit["going_concern"]
    if raw_gc is None:
        # Unverified — no bool(None)=False silent-clean, no veto.
        gc_pts = 0
        components["going_concern"] = {"present": None, "points": 0}
        flags.append("going_concern_unverified")
    elif raw_gc is True:
        gc_pts = -100
        components["going_concern"] = {"present": True, "points": gc_pts}
        vetos.append("going_concern")
    else:
        gc_pts = 20
        components["going_concern"] = {"present": False, "points": gc_pts}
    return gc_pts


def _apply_composite_cap(points: int, composite_cap: int | None) -> int:
    composite = max(0, min(100, points))
    if composite_cap is not None:
        composite = min(composite, composite_cap)
    return composite


def compute_dq_score(
    *,
    ticker: str,
    fundamentals: dict,
    audit_signals: dict,
    sloan_universe: list[float] | None = None,
    as_of: date | None = None,
    allow_mock: bool = False,
) -> dict:
    """Composite DQ-Score with live (non-advisory) Hard-Vetos.

    Returns dict with `composite_score`, `components`, `vetos`, `flags`, `ticker`.
    Composite bounded 0..100. Hard caps (M>-1.78 caps 30, >=2 restatements
    caps 40) are applied directly. Vetos list is non-empty if any hard-flag fires.
    """
    if not ticker or not isinstance(ticker, str):
        raise DQContractError("ticker must be a non-empty string")

    audit = _validate_audit_signals(audit_signals)
    as_of = as_of or date.today()

    points = 0
    components: dict[str, Any] = {}
    vetos: list[str] = []
    flags: list[str] = []
    composite_cap: int | None = None

    # Phase order is behavior: components-dict key order, flags order and
    # vetos order are serialized downstream, and each phase appends in place.
    _apply_audit_source_flags(audit, allow_mock, flags)
    points += _score_report_age(fundamentals, audit, as_of, components, flags)
    points += _score_auditor_tier(audit, components, flags)
    rest_pts, composite_cap = _score_restatements(audit, composite_cap, components, vetos, flags)
    points += rest_pts
    points += _score_revenue_recognition(audit, components)

    # Sector-Policy via _SECTOR_BYPASS_POLICY.
    # Financials: Sloan+Beneish bypass (BS-Deltas dominated by deposits;
    #             Beneish industrial-calibrated).
    # Utilities:  Beneish-only bypass (SGA aggregated in totalOperatingExpenses;
    #             Sloan still applies because BS-deltas are universally
    #             meaningful — conservative default).
    # Unknown sectors: no bypass (prevents a Wirecard-style Soft-Mode).
    general = (fundamentals or {}).get("General") or {}
    sector = general.get("GicSector")
    sector_policy = _get_sector_policy(sector)

    sloan_pts, composite_cap = _score_sloan(
        fundamentals, sloan_universe, sector_policy, composite_cap, components, vetos, flags,
    )
    points += sloan_pts
    beneish_pts, composite_cap = _score_beneish(
        fundamentals, sector_policy, composite_cap, components, vetos, flags,
    )
    points += beneish_pts
    points += _score_going_concern(audit, components, vetos, flags)

    composite = _apply_composite_cap(points, composite_cap)

    return {
        "ticker": ticker,
        "composite_score": composite,
        "raw_points": points,
        "components": components,
        "vetos": vetos,
        "flags": flags,
        "composite_cap_applied": composite_cap,
        "as_of": as_of.isoformat(),
    }


def get_dq_vetos(score_result: dict) -> list[str]:
    """Return the veto-list of a `compute_dq_score` result. Convenience accessor."""
    if not isinstance(score_result, dict):
        raise DQContractError("score_result must be a dict from compute_dq_score")
    return list(score_result.get("vetos", []))

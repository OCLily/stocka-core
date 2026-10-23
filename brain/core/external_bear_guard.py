"""Shared external-bear-case guard.

The pattern: a judgment-supporting bear_case must be EXTERNALLY researched
(source_type='external' + url + source_date), never sourced from the system's
own corpus or prose — otherwise a single-user corpus reinforces its own biases
instead of correcting them. Several downstream consumers (outside this excerpt)
share this one function rather than duplicating the three checks.

Deliberately NOT checked here: `these` (the counter-thesis text itself). This
function is a PROVENANCE gate (external source, not the system's own
corpus/prose) — content completeness is a CONSUMER concern, not a provenance
one. Some callers pass bear_case objects that never carry a `these` field at
all; a `these`-not-empty check here would break them, unrelated to their own
gate. Where content completeness DOES matter (a consumer that shows `these`
verbatim), the check lives in that consumer's own validation.
"""
from __future__ import annotations

_BEAR_SOURCE_REQUIRED = "external"


def require_external_bear_case(bear_case: dict, *, context: str) -> None:
    """Raise ValueError unless bear_case carries source_type='external' + url +
    source_date. `context` is prefixed into every message so each caller keeps
    its own identity (e.g. "report", "check")."""
    if not isinstance(bear_case, dict):
        raise ValueError(
            f"{context}: bear_case fehlt — ohne extern recherchierten Bear-Case "
            "(source_type='external' + url + source_date) wird NICHT akzeptiert.")
    if bear_case.get("source_type") != _BEAR_SOURCE_REQUIRED:
        raise ValueError(
            f"{context}: bear_case.source_type={bear_case.get('source_type')!r} — "
            "erlaubt ist NUR 'external' (kein Korpus-/Analysis-Echo).")
    if not bear_case.get("url"):
        raise ValueError(f"{context}: bear_case.url fehlt — externer Bear braucht belegbare Quelle.")
    if not bear_case.get("source_date"):
        raise ValueError(f"{context}: bear_case.source_date fehlt — externer Bear braucht ein Datum.")

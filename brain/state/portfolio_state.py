"""PortfolioState — single source-of-truth for the book.

Holds positions (per-ticker dicts), peak_ytd (the high-water-mark anchor),
current_dd, tier (DD-State-Machine Stufe 0-3), cooling-off window timestamps,
the EUR cash balance, a flow ledger of deposits/withdrawals, and an append-only
audit log. JSON round-trip via to_dict / from_dict; apply_transition mutates
and appends a timestamped audit entry.

Das Positions-Dict-Schema spiegelt die Pflichtfelder des Trade-File-Validators
(ein separates Skript, nicht Teil dieses Auszugs). `to_trade_frontmatter()`
emittiert genau die Keys, die dieser Validator erwartet; der produktive Weg
vom State zum Trade-File baut sein Frontmatter allerdings selbst. Methode und
REQUIRED/OPTIONAL_POSITION_FIELDS sind daher Schema-Dokumentation, nicht als
aktiv verdrahtet annehmen.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional


# Schema-Dokumentation, KEIN Enforcement: beide Tupel sind unreferenziert.
# Nichts prueft eingehende Positions-Dicts dagegen — s. Modul-Docstring oben.
REQUIRED_POSITION_FIELDS = (
    "entry_price",
    "shares",
    "entry_date",
    "status",
    "stop_loss",
    "stop_loss_reason",
    "linked_research",
    "score_at_entry",
    "slot",
    "sector",
)

OPTIONAL_POSITION_FIELDS = (
    "exit_date",
    "position_size_eur",
    "position_size_pct",
    "soft_stop",
    "trailing_soft",
    "atr_at_entry",
    # Ebenfalls nur Doku: der FX-Kurs, mit dem der Kauf gestempelt wurde
    # (position_size_eur = shares x entry_price x entry_fx_usd_eur = der
    # gebuchte Kauf-Cash), und Fill/FX des Ausstiegs. Ein Ingest-Schritt muss
    # alle drei aus dem Trade-File mitkopieren, weil apply_transition das
    # Positions-Dict des Tickers ERSETZT.
    "entry_fx_usd_eur",
    "exit_price",
    "exit_fx_usd_eur",
)


# YAML chars that force quoting (incomplete vs full YAML spec, but covers our cases).
# `#` is special — the trade-file validator's awk parser strips `#.*$` regardless
# of quoting, so we REJECT `#` in user-controlled string fields instead of trying
# to escape it.
_YAML_SPECIAL = frozenset(':\n\r\t"\'{}[]&*!|>%@`,')


def _yaml_str(value) -> str:
    """Return YAML-safe scalar. Quotes only when needed; uses single-quote style.

    `#` triggers ValueError because the trade-file validator's awk parser
    strips `#.*$` before quote-awareness, which would silently truncate the value.
    """
    s = str(value)
    if "#" in s:
        raise ValueError(
            f"value contains '#' which the trade-file validator would silently strip: {s!r}"
        )
    if s == "":
        return "''"
    if s.lstrip() != s or s.rstrip() != s:
        return "'" + s.replace("'", "''") + "'"
    if any(c in _YAML_SPECIAL for c in s):
        return "'" + s.replace("'", "''") + "'"
    if s.lower() in {"true", "false", "yes", "no", "null", "~", "on", "off"}:
        return "'" + s + "'"
    if s[0] in "-?":
        return "'" + s + "'"
    return s


class SchemaError(Exception):
    """Raised when serialized state does not match the current schema."""


class PortfolioState:
    """Mutable state-of-the-book. JSON-round-trip via to_dict / from_dict."""

    SCHEMA_VERSION = 1
    DEFAULT_PEAK_YTD = 10_000.0

    def __init__(
        self,
        *,
        positions: Optional[dict] = None,
        peak_ytd: float = DEFAULT_PEAK_YTD,
        current_dd: float = 0.0,
        tier: int = 0,
        tier_entered_at: Optional[str] = None,
        cooling_off_until: Optional[str] = None,
        audit_log: Optional[list] = None,
        cash_eur: Optional[float] = None,
        cash_eur_updated_at: Optional[str] = None,
        net_flows_eur: float = 0.0,
        flow_log: Optional[list] = None,
    ) -> None:
        self.positions: dict = positions if positions is not None else {}
        self.peak_ytd: float = float(peak_ytd)
        self.current_dd: float = float(current_dd)
        self.tier: int = int(tier)
        self.tier_entered_at: Optional[str] = tier_entered_at
        self.cooling_off_until: Optional[str] = cooling_off_until
        self.audit_log: list = audit_log if audit_log is not None else []
        # EUR-Cash-Saldo. Default None, NEVER 0.0 — 0.0 is a
        # meaningful "all-in, no cash" value that would silently mis-measure
        # DD as if cash were known-zero. None = "never set" (fail-closed).
        self.cash_eur: Optional[float] = cash_eur
        self.cash_eur_updated_at: Optional[str] = cash_eur_updated_at
        # Flow-Ledger: kumulative Netto-Zufluesse SEIT der Anker-Setzung
        # (Deposits +, Withdrawals -). Einzahlungen zaehlen nie zur
        # Performance — HWM/DD laufen auf der bereinigten Kurve
        # adjusted = equity - net_flows. flow_log ist die append-only
        # Historie; sie wird NIE geloescht.
        self.net_flows_eur: float = float(net_flows_eur)
        self.flow_log: list = flow_log if flow_log is not None else []

    def to_dict(self) -> dict:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "positions": self.positions,
            "peak_ytd": self.peak_ytd,
            "current_dd": self.current_dd,
            "tier": self.tier,
            "tier_entered_at": self.tier_entered_at,
            "cooling_off_until": self.cooling_off_until,
            "audit_log": self.audit_log,
            "cash_eur": self.cash_eur,
            "cash_eur_updated_at": self.cash_eur_updated_at,
            "net_flows_eur": self.net_flows_eur,
            "flow_log": self.flow_log,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PortfolioState":
        if not isinstance(data, dict):
            raise SchemaError("state-data must be dict")
        version = data.get("schema_version")
        if version != cls.SCHEMA_VERSION:
            raise SchemaError(
                f"schema_version mismatch (got {version!r}, expect {cls.SCHEMA_VERSION})"
            )
        return cls(
            positions=data.get("positions") or {},
            peak_ytd=data.get("peak_ytd", cls.DEFAULT_PEAK_YTD),
            current_dd=data.get("current_dd", 0.0),
            tier=data.get("tier", 0),
            tier_entered_at=data.get("tier_entered_at"),
            cooling_off_until=data.get("cooling_off_until"),
            audit_log=data.get("audit_log") or [],
            cash_eur=data.get("cash_eur"),
            cash_eur_updated_at=data.get("cash_eur_updated_at"),
            net_flows_eur=data.get("net_flows_eur", 0.0),
            flow_log=data.get("flow_log") or [],
        )

    def apply_transition(self, reason: str, **changes: Any) -> None:
        """Mutate state; append timestamped audit entry.

        Recognized change-keys: `positions_update` (dict ticker→pos-dict),
        plus any direct attribute on self (peak_ytd, current_dd, tier,
        tier_entered_at, cooling_off_until, cash_eur, cash_eur_updated_at,
        net_flows_eur, flow_log). Unknown keys raise ValueError.
        """
        ts = datetime.now(timezone.utc).isoformat()
        applied: dict = {}
        for key, val in changes.items():
            if key == "positions_update" and isinstance(val, dict):
                self.positions.update(val)
                applied[key] = sorted(val.keys())
            elif key in ("peak_ytd", "current_dd", "tier",
                         "tier_entered_at", "cooling_off_until",
                         "cash_eur", "cash_eur_updated_at",
                         "net_flows_eur", "flow_log"):
                setattr(self, key, val)
                applied[key] = val
            else:
                raise ValueError(f"unknown change-key: {key!r}")
        self.audit_log.append({"ts": ts, "reason": reason, "applied": applied})

    def to_trade_frontmatter(self, ticker: str) -> str:
        """Emit YAML-frontmatter matching the trade-file validator's expectations.

        Emits exactly the validator-relevant keys (plus exit_date for closed
        positions). Sizing: position_size_eur ODER position_size_pct, exklusiv.
        Wenn beide gesetzt: ValueError (Trader-Guardrail).
        """
        pos = self.positions.get(ticker)
        if not pos:
            raise KeyError(f"position {ticker!r} not in state")

        for f in ("status", "entry_date", "stop_loss", "stop_loss_reason",
                  "linked_research", "score_at_entry"):
            if pos.get(f) in (None, ""):
                raise ValueError(f"{ticker} missing required field: {f}")

        size_eur = pos.get("position_size_eur")
        size_pct = pos.get("position_size_pct")
        if size_eur is None and size_pct is None:
            raise ValueError(f"{ticker} missing position_size_eur OR position_size_pct")
        if size_eur is not None and size_pct is not None:
            raise ValueError(f"{ticker} has both position_size_eur and _pct (must be exclusive)")

        lines = [
            "---",
            f"ticker: {ticker}",
            f"status: {pos['status']}",
            f"entry_date: {pos['entry_date']}",
        ]
        if pos["status"] == "closed":
            if not pos.get("exit_date"):
                raise ValueError(f"{ticker} status=closed but exit_date missing")
            lines.append(f"exit_date: {pos['exit_date']}")
        lines.extend([
            f"stop_loss: {pos['stop_loss']}",
            f"stop_loss_reason: {_yaml_str(pos['stop_loss_reason'])}",
            f"linked_research: {_yaml_str(pos['linked_research'])}",
        ])
        if size_eur is not None:
            lines.append(f"position_size_eur: {size_eur}")
        else:
            lines.append(f"position_size_pct: {size_pct}")
        lines.append(f"score_at_entry: {pos['score_at_entry']}")
        lines.append("---")
        return "\n".join(lines) + "\n"

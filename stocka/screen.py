"""Illustrative screen: config thresholds applied to per-ticker engine results.

This is an example of how the engine's outputs can be combined into a
pass/fail decision with stated reasons. The thresholds are configuration,
not research findings, and the screen is not a buy list.
"""

from __future__ import annotations

from dataclasses import dataclass

from stocka.config import ScreenSettings


@dataclass(frozen=True)
class EngineResult:
    ticker: str
    sector: str
    dq: dict | None
    dq_error: str | None
    beneish_m: float | None
    sloan_ratio: float | None
    sloan_quintile: int | None


@dataclass(frozen=True)
class ScreenDecision:
    ticker: str
    passed: bool
    reasons: tuple[str, ...]
    notes: tuple[str, ...]


def _beneish_checks(result: EngineResult, cfg: ScreenSettings, reasons: list[str], notes: list[str]) -> None:
    flags = result.dq["flags"]
    if any(f.startswith(("beneish_not_applicable", "beneish_red_not_applicable")) for f in flags):
        notes.append("Beneish not applied (sector bypass)")
    elif (
        result.beneish_m is not None
        and result.beneish_m > cfg.max_beneish_m
        and "beneish_red_flag" not in result.dq["vetos"]
    ):
        reasons.append(f"Beneish M {result.beneish_m:.2f} > {cfg.max_beneish_m:.2f}")


def _sloan_checks(result: EngineResult, cfg: ScreenSettings, reasons: list[str], notes: list[str]) -> None:
    flags = result.dq["flags"]
    if any(f.startswith("sloan_not_applicable") for f in flags):
        notes.append("Sloan not applied (sector bypass)")
    elif result.sloan_quintile is None:
        notes.append("Sloan quintile unavailable")
    elif cfg.exclude_sloan_q5 and result.sloan_quintile == 5 and "sloan_q5_anomaly" not in result.dq["vetos"]:
        reasons.append("Sloan accruals in Q5")


def screen_ticker(result: EngineResult, cfg: ScreenSettings) -> ScreenDecision:
    if result.dq is None:
        return ScreenDecision(result.ticker, False, (f"audit input rejected: {result.dq_error}",), ())
    reasons = [f"engine veto: {v}" for v in result.dq["vetos"]]
    notes: list[str] = []
    score = result.dq["composite_score"]
    if score < cfg.min_dq_score:
        reasons.append(f"DQ score {score} < {cfg.min_dq_score}")
    _beneish_checks(result, cfg, reasons, notes)
    _sloan_checks(result, cfg, reasons, notes)
    if cfg.require_verified_audit:
        reasons.extend(f"unverified input: {f}" for f in result.dq["flags"] if f.endswith("_unverified"))
    return ScreenDecision(result.ticker, not reasons, tuple(reasons), tuple(notes))


def run_screen(results: list[EngineResult], cfg: ScreenSettings) -> list[ScreenDecision]:
    return [screen_ticker(r, cfg) for r in results]

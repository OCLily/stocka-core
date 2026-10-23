"""Sloan (1996) Total-Accruals + Quintile-Classifier on EODHD fundamentals.

Balance-Sheet method:
  Accruals_t = (DCA - DCL) - (DCash - DSTD) - Dep where:
    DCA   = Delta Current Assets
    DCL   = Delta Current Liabilities
    DCash = Delta (cash + short-term investments)
    DSTD  = Delta short-term debt (part of current liabilities — netted out)
    Dep   = depreciation+amortization (income-statement-period)

Scaled by average totalAssets to yield `accruals_ratio`. High accruals (Q5) =
earnings quality concern (Sloan anomaly). Quintile-classifier requires a
universe-sample list; without it the function returns ratio only.

Pure-function — no provider imports.
"""

from __future__ import annotations

from typing import Any


def _as_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# Sectors where cash-rich-heuristic is unsafe — balance-sheet structure makes
# the heuristic produce false-positives (Banks: cash+STI dominates BUT deposits
# in totalLiab are much larger; Insurer: reserves count as cash, premiums as
# liabilities). Explicit blocklist preserves the Sloan hard-veto integrity
# even if a caller skips the financial-sector bypass.
_CASH_RICH_BANNED_SECTORS = frozenset({"Financials"})


def _is_cash_rich(bal: dict | None, sector: str | None = None) -> bool:
    """Balance-sheet plausibility for the shortTermDebt null-fallback.

    Returns True iff the filer's cash-equivalents (cash + shortTermInvestments,
    or pre-aggregated cashAndShortTermInvestments) exceed longTermDebt.
    Used to justify treating a null shortTermDebt as zero (vs missing).

    Sector-Gate: Banks/Insurers (`sector == "Financials"`) fail this test
    unconditionally — their balance sheet appears cash-rich but is funded by
    deposits/premiums (in totalLiab). Without this gate, a large bank
    with null shortTermDebt would be falsely treated as zero, manipulating
    Sloan-accruals and bypassing the explicit Financial-Bypass in dq_score.py.

    Cash-rich filers (tech mega-caps) often genuinely have zero
    short-term debt — no commercial-paper program. False-Negative-Window:
    if a filer historically reported short-term debt and the provider stops
    delivering the field for the current year, the fallback will mask it
    (silent zero). Not yet mitigated; a drift flag for std_p > 0 with a
    null std_t would close the gap.
    """
    if sector in _CASH_RICH_BANNED_SECTORS:
        return False
    if not bal:
        return False
    csti = _as_float(bal.get("cashAndShortTermInvestments"))
    if csti is None:
        cash = _as_float(bal.get("cash")) or 0.0
        sti = _as_float(bal.get("shortTermInvestments")) or 0.0
        csti = cash + sti if (cash or sti) else None
    ltdebt = _as_float(bal.get("longTermDebt"))
    if csti is None or ltdebt is None:
        return False
    return csti > ltdebt


def _two_period(financials: dict, branch: str) -> tuple[dict | None, dict | None]:
    yearly = (financials or {}).get(branch, {}).get("yearly", {}) or {}
    dates = sorted(yearly.keys(), reverse=True)
    if len(dates) < 2:
        return None, None
    return yearly[dates[0]], yearly[dates[1]]


def compute_sloan_accruals(fundamentals: dict) -> dict:
    """Balance-sheet accruals from a 2-year EODHD fundamentals dict.

    Returns:
        {
            "accruals": float | None,
            "avg_total_assets": float | None,
            "accruals_ratio": float | None,
            "complete": bool,
            "missing_inputs": list[str],
        }
    """
    fin = (fundamentals or {}).get("Financials", {}) or {}
    bal_t, bal_p = _two_period(fin, "Balance_Sheet")
    inc_t, _inc_p = _two_period(fin, "Income_Statement")

    missing: list[str] = []

    def f(d: dict | None, key: str, label: str) -> float | None:
        if d is None:
            missing.append(label)
            return None
        v = _as_float(d.get(key))
        if v is None:
            missing.append(label)
        return v

    ca_t = f(bal_t, "totalCurrentAssets", "Balance.totalCurrentAssets[t]")
    ca_p = f(bal_p, "totalCurrentAssets", "Balance.totalCurrentAssets[t-1]")
    cl_t = f(bal_t, "totalCurrentLiabilities", "Balance.totalCurrentLiabilities[t]")
    cl_p = f(bal_p, "totalCurrentLiabilities", "Balance.totalCurrentLiabilities[t-1]")

    cash_t = _as_float((bal_t or {}).get("cash"))
    cash_p = _as_float((bal_p or {}).get("cash"))
    sti_t = _as_float((bal_t or {}).get("shortTermInvestments"))
    sti_p = _as_float((bal_p or {}).get("shortTermInvestments"))

    def _cash_eq(cash, sti):
        if cash is None and sti is None:
            return None
        return (cash or 0.0) + (sti or 0.0)

    ce_t = _cash_eq(cash_t, sti_t)
    ce_p = _cash_eq(cash_p, sti_p)
    if ce_t is None:
        missing.append("Balance.cashEq[t]")
    if ce_p is None:
        missing.append("Balance.cashEq[t-1]")

    std_t = _as_float((bal_t or {}).get("shortTermDebt"))
    std_p = _as_float((bal_p or {}).get("shortTermDebt"))

    # Cash-rich filer fallback, an EODHD field-drift workaround. If
    # shortTermDebt is null for a year AND the filer is cash-rich
    # (cashAndShortTermInvestments > longTermDebt), treat shortTermDebt as 0.
    # Plausibility: cash-rich filers with reported longTermDebt but null
    # shortTermDebt typically have genuinely zero short-term debt (no
    # commercial-paper program). Banks/Insurers fail the cash-rich test
    # (deposits dominate balance sheet), preventing false-zero.
    sector = (fundamentals or {}).get("General", {}).get("GicSector")
    if std_t is None and _is_cash_rich(bal_t, sector):
        std_t = 0.0
    elif std_t is None:
        missing.append("Balance.shortTermDebt[t]")
    if std_p is None and _is_cash_rich(bal_p, sector):
        std_p = 0.0
    elif std_p is None:
        missing.append("Balance.shortTermDebt[t-1]")

    dep = f(inc_t, "depreciationAndAmortization", "Income.D&A[t]")
    ta_t = f(bal_t, "totalAssets", "Balance.totalAssets[t]")
    ta_p = f(bal_p, "totalAssets", "Balance.totalAssets[t-1]")

    if any(v is None for v in (ca_t, ca_p, cl_t, cl_p, ce_t, ce_p,
                                std_t, std_p, dep, ta_t, ta_p)):
        return {
            "accruals": None,
            "avg_total_assets": None,
            "accruals_ratio": None,
            "complete": False,
            "missing_inputs": missing,
        }

    d_ca = ca_t - ca_p
    d_cl = cl_t - cl_p
    d_cash = ce_t - ce_p
    d_std = std_t - std_p

    accruals = (d_ca - d_cl) - (d_cash - d_std) - dep
    avg_ta = (ta_t + ta_p) / 2.0
    ratio = accruals / avg_ta if avg_ta != 0 else None

    return {
        "accruals": round(accruals, 2),
        "avg_total_assets": round(avg_ta, 2),
        "accruals_ratio": round(ratio, 6) if ratio is not None else None,
        "complete": True,
        "missing_inputs": [],
    }


def compute_sloan_quintile(
    target_accruals_ratio: float,
    universe_samples: list[float],
) -> int | None:
    """Return 1..5 (Q1 = lowest accruals, Q5 = highest, the anomaly).

    Requires >= 10 numeric samples in the universe. Returns None on insufficient.

    MIN_SAMPLE = 10 (single-ticker-classifier-min):
    Statistisch sinnvoll für Quintile-Klassifikation eines SINGLE Tickers gegen
    Universe-Samples — Quintile partitionieren in 5 Buckets, mit 10 Samples
    bekommt jeder Bucket ≥2 Datapoints (Boundary-Computation aussagekräftig).
    Bei <10 wäre Quintile-Boundary von einzelnen Outliers dominiert.

    Semantisch unterschiedlich von sloan_universe._MIN_SAMPLE_SIZE=5:
    - dieser Wert (10) = classifier-min für valide quintile-Klassifikation
    - jener Wert (5)   = snapshot-publication-min (5 boundaries computable
                         mit 5 samples, aber statistisch dünn)
    Deshalb werden die beiden Werte nicht vereinheitlicht: mit 10 als
    gemeinsamem Minimum hätte der Universe-Snapshot bei kleinen Sektor-
    Teilmengen (z.B. 6 Ticker) still quintiles=None, und ein darauf
    gestütztes Hard-Veto würde unbemerkt entfallen.
    """
    if target_accruals_ratio is None:
        return None
    clean = [s for s in (universe_samples or []) if isinstance(s, (int, float))]
    if len(clean) < 10:
        return None
    s = sorted(clean)
    n = len(s)
    boundaries = [s[int(n * q)] for q in (0.2, 0.4, 0.6, 0.8)]
    for i, b in enumerate(boundaries, start=1):
        if target_accruals_ratio <= b:
            return i
    return 5

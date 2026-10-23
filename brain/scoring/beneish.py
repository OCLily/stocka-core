"""Beneish M-Score (8-component) on EODHD fundamentals.

Pure function on a pre-fetched fundamentals dict (no provider import; the
caller fetches the EODHD fundamentals payload). Returns components +
composite + completeness flag.

Beneish (1999): M = -4.84 + 0.92*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI
                    + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI

Hard-flag: M > -1.78. Gray-zone: -2.22 <= M <= -1.78. Clear: M < -2.22.
Gray-floor -2.22: secondary-literature convention, not in Beneish (1999).
"""

from __future__ import annotations

from typing import Any


BENEISH_THRESHOLD_RED = -1.78
BENEISH_THRESHOLD_GRAY_LOW = -2.22


def _as_float(v: Any) -> float | None:
    """EODHD numeric values arrive as strings; cast defensively. Empty/None -> None."""
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f


def _safe_div(num: float | None, den: float | None) -> float | None:
    if num is None or den is None:
        return None
    if den == 0:
        return None
    return num / den


def _two_period_yearly(financials: dict, branch: str) -> tuple[dict | None, dict | None, str | None, str | None]:
    """Return (current_year_dict, prior_year_dict, cur_date, prior_date) sorted desc."""
    yearly = (financials or {}).get(branch, {}).get("yearly", {}) or {}
    dates = sorted(yearly.keys(), reverse=True)
    if len(dates) < 2:
        return None, None, (dates[0] if dates else None), None
    return yearly[dates[0]], yearly[dates[1]], dates[0], dates[1]


def compute_beneish_m_score(fundamentals: dict) -> dict:
    """Compute the 8-component Beneish M-Score from an EODHD fundamentals dict.

    Returns:
        {
            "m_score": float | None,
            "components": {DSRI, GMI, AQI, SGI, DEPI, SGAI, TATA, LVGI: float | None},
            "complete": bool,
            "missing_inputs": list[str],
            "flag": "clear" | "gray" | "red" | "incomplete",
            "current_year": str | None, "prior_year": str | None,
        }
    """
    fin = (fundamentals or {}).get("Financials", {}) or {}
    inc_cur, inc_prior, cur_date, prior_date = _two_period_yearly(fin, "Income_Statement")
    bal_cur, bal_prior, _, _ = _two_period_yearly(fin, "Balance_Sheet")
    cf_cur, _cf_prior, _, _ = _two_period_yearly(fin, "Cash_Flow")

    missing: list[str] = []

    def f(d: dict | None, key: str, label: str) -> float | None:
        if d is None:
            missing.append(label)
            return None
        v = _as_float(d.get(key))
        if v is None:
            missing.append(label)
        return v

    def f_alt(d: dict | None, keys: tuple, label: str) -> float | None:
        """Try keys in order; return first non-null. EODHD field-drift workaround.

        Beneish-paper specifies "Property, Plant and Equipment (net)". EODHD
        populates `propertyPlantAndEquipmentNet` consistently across filers;
        legacy `propertyPlantEquipment` is null for some large-cap filers.
        Prefer the spec-correct net-field, fall back for payloads where only
        the legacy field is set.
        """
        if d is None:
            missing.append(label)
            return None
        for k in keys:
            v = _as_float(d.get(k))
            if v is not None:
                return v
        missing.append(label)
        return None

    rev_t = f(inc_cur, "totalRevenue", "Income.totalRevenue[t]")
    rev_p = f(inc_prior, "totalRevenue", "Income.totalRevenue[t-1]")
    gp_t = f(inc_cur, "grossProfit", "Income.grossProfit[t]")
    gp_p = f(inc_prior, "grossProfit", "Income.grossProfit[t-1]")
    sga_t = f(inc_cur, "sellingGeneralAdministrative", "Income.SGA[t]")
    sga_p = f(inc_prior, "sellingGeneralAdministrative", "Income.SGA[t-1]")
    da_t = f(inc_cur, "depreciationAndAmortization", "Income.D&A[t]")
    da_p = f(inc_prior, "depreciationAndAmortization", "Income.D&A[t-1]")
    ni_t = f(inc_cur, "netIncome", "Income.netIncome[t]")

    ar_t = f(bal_cur, "netReceivables", "Balance.netReceivables[t]")
    ar_p = f(bal_prior, "netReceivables", "Balance.netReceivables[t-1]")
    ca_t = f(bal_cur, "totalCurrentAssets", "Balance.totalCurrentAssets[t]")
    ca_p = f(bal_prior, "totalCurrentAssets", "Balance.totalCurrentAssets[t-1]")
    ppe_t = f_alt(
        bal_cur,
        ("propertyPlantAndEquipmentNet", "propertyPlantEquipment"),
        "Balance.PPE[t]",
    )
    ppe_p = f_alt(
        bal_prior,
        ("propertyPlantAndEquipmentNet", "propertyPlantEquipment"),
        "Balance.PPE[t-1]",
    )
    ta_t = f(bal_cur, "totalAssets", "Balance.totalAssets[t]")
    ta_p = f(bal_prior, "totalAssets", "Balance.totalAssets[t-1]")

    cl_t = f(bal_cur, "totalCurrentLiabilities", "Balance.totalCurrentLiabilities[t]")
    cl_p = f(bal_prior, "totalCurrentLiabilities", "Balance.totalCurrentLiabilities[t-1]")

    def f_opt0_alt(d: dict | None, keys: tuple) -> float:
        # LVGI-LTD null->0: fehlende LTD = keine Long-Term-Debt.
        # Bekannte Rest-Abweichung: echte Schuldner mit EODHD-Luecke.
        if d is None:
            return 0.0
        for k in keys:
            v = _as_float(d.get(k))
            if v is not None:
                return v
        return 0.0

    ltd_t = f_opt0_alt(bal_cur, ("longTermDebt", "longTermDebtTotal"))
    ltd_p = f_opt0_alt(bal_prior, ("longTermDebt", "longTermDebtTotal"))

    cfo_t = f(cf_cur, "totalCashFromOperatingActivities", "CashFlow.CFO[t]")

    dsri = _safe_div(_safe_div(ar_t, rev_t), _safe_div(ar_p, rev_p))

    gm_t = _safe_div(gp_t, rev_t)
    gm_p = _safe_div(gp_p, rev_p)
    gmi = _safe_div(gm_p, gm_t)

    def f_opt0(d: dict | None, key: str) -> float:
        # Beneish-"Securities": fehlendes Feld = Position existiert nicht (0.0),
        # KEINE Datenluecke (null->0).
        if d is None:
            return 0.0
        v = _as_float(d.get(key))
        return 0.0 if v is None else v

    sec_t = f_opt0(bal_cur, "longTermInvestments")
    sec_p = f_opt0(bal_prior, "longTermInvestments")

    def aqi_term(ca, ppe, sec, ta):
        if ca is None or ppe is None or ta is None or ta == 0:
            return None
        return 1.0 - (ca + ppe + sec) / ta
    aqi = _safe_div(aqi_term(ca_t, ppe_t, sec_t, ta_t), aqi_term(ca_p, ppe_p, sec_p, ta_p))

    sgi = _safe_div(rev_t, rev_p)

    def depi_term(da, ppe):
        if da is None or ppe is None:
            return None
        denom = da + ppe
        if denom == 0:
            return None
        return da / denom
    depi = _safe_div(depi_term(da_p, ppe_p), depi_term(da_t, ppe_t))

    sgai = _safe_div(_safe_div(sga_t, rev_t), _safe_div(sga_p, rev_p))

    def lvgi_term(cl, ltd, ta):
        if cl is None or ta is None or ta == 0:
            return None
        return (cl + ltd) / ta
    lvgi = _safe_div(lvgi_term(cl_t, ltd_t, ta_t), lvgi_term(cl_p, ltd_p, ta_p))

    if ni_t is not None and cfo_t is not None and ta_t is not None and ta_t != 0:
        # netIncome proxies "income from continuing operations" — accepted convention.
        tata = (ni_t - cfo_t) / ta_t
    else:
        tata = None

    components = {
        "DSRI": dsri, "GMI": gmi, "AQI": aqi, "SGI": sgi,
        "DEPI": depi, "SGAI": sgai, "TATA": tata, "LVGI": lvgi,
    }

    if all(v is not None for v in components.values()):
        m = (
            -4.84
            + 0.92 * dsri
            + 0.528 * gmi
            + 0.404 * aqi
            + 0.892 * sgi
            + 0.115 * depi
            - 0.172 * sgai
            + 4.679 * tata
            - 0.327 * lvgi
        )
        if m > BENEISH_THRESHOLD_RED:
            flag = "red"
        elif m >= BENEISH_THRESHOLD_GRAY_LOW:
            flag = "gray"
        else:
            flag = "clear"
        return {
            "m_score": round(m, 4),
            "components": {k: round(v, 6) for k, v in components.items()},
            "complete": True,
            "missing_inputs": [],
            "flag": flag,
            "current_year": cur_date,
            "prior_year": prior_date,
        }

    return {
        "m_score": None,
        "components": components,
        "complete": False,
        "missing_inputs": missing,
        "flag": "incomplete",
        "current_year": cur_date,
        "prior_year": prior_date,
    }

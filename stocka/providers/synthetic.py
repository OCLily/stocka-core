"""Deterministic synthetic data: no network, no files, no real companies.

Tickers are SYN01..SYNnn. Prices come from a one-factor-per-sector model
(market factor + sector factor + idiosyncratic noise) drawn with
`numpy.random.default_rng`, so a seed always yields the same data. One
market-wide stress window and one single-name crash give the drawdown
measures something to find. Dates are business days (weekdays) ending on a
fixed date, without an exchange holiday calendar. Fundamentals are generated
from revenue ratios in the EODHD shape described in `stocka.providers.base`.

The first twenty tickers follow a fixed profile list. Most are plain, but
some are built to trip the engine's guards: a Beneish red-zone ticker, a
gray-zone ticker, a high-accruals ticker (Sloan Q5 anomaly), a bank (sector
bypass), missing or unverified audit data, a restatement, a going-concern
opinion and a stale report. Tickers beyond twenty are plain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

from stocka.providers.base import Ticker

END_DATE = date(2025, 6, 30)
DEFAULT_DAYS = 780
MAX_TICKERS = 99
MARKET_VOL = 0.009
MARKET_DRIFT = 0.0003

# sector -> (market beta, sector beta, sector factor vol, idiosyncratic vol), daily
SECTOR_MODEL: dict[str, tuple[float, float, float, float]] = {
    "Information Technology": (1.1, 1.0, 0.012, 0.005),
    "Health Care": (0.8, 0.6, 0.008, 0.011),
    "Financials": (1.0, 0.7, 0.009, 0.009),
    "Consumer Discretionary": (1.1, 0.6, 0.009, 0.013),
    "Consumer Staples": (0.6, 0.5, 0.006, 0.008),
    "Industrials": (1.0, 0.6, 0.008, 0.011),
    "Energy": (0.9, 1.0, 0.014, 0.016),
    "Materials": (1.0, 0.7, 0.010, 0.013),
    "Utilities": (0.5, 0.6, 0.006, 0.007),
    "Communication Services": (1.0, 0.6, 0.009, 0.012),
    "Real Estate": (0.8, 0.7, 0.009, 0.010),
}
SECTORS = tuple(SECTOR_MODEL)


@dataclass(frozen=True)
class Profile:
    sector: str
    kind: str = "plain"
    crash: bool = False


PROFILES: tuple[Profile, ...] = (
    Profile("Information Technology"),
    Profile("Health Care"),
    Profile("Industrials", "beneish_red"),
    Profile("Information Technology"),
    Profile("Consumer Discretionary", "high_accruals"),
    Profile("Consumer Staples"),
    Profile("Information Technology"),
    Profile("Energy", crash=True),
    Profile("Financials", "bank"),
    Profile("Materials", "missing_audit"),
    Profile("Utilities"),
    Profile("Communication Services", "beneish_gray"),
    Profile("Industrials", "unverified_audit"),
    Profile("Health Care", "restatement"),
    Profile("Real Estate", "stale_report"),
    Profile("Financials"),
    Profile("Information Technology"),
    Profile("Energy", "going_concern"),
    Profile("Consumer Discretionary"),
    Profile("Materials"),
)

# Share of revenue, drawn uniformly per ticker.
_RATIO_RANGES: dict[str, tuple[float, float]] = {
    "gross_profit": (0.30, 0.60),
    "sga": (0.12, 0.22),
    "net_income": (0.06, 0.14),
    "receivables": (0.10, 0.15),
    "inventory": (0.08, 0.12),
    "cash": (0.10, 0.20),
    "short_term_investments": (0.03, 0.08),
    "ppe": (0.40, 0.60),
    "long_term_investments": (0.05, 0.12),
    "other_assets": (0.15, 0.25),
    "operating_liabilities": (0.10, 0.14),
    "short_term_debt": (0.02, 0.04),
    "long_term_debt": (0.20, 0.40),
}
_DEPRECIATION_RATE = 0.10
_JITTER = 0.02


@dataclass(frozen=True)
class _Accounts:
    growth: tuple[float, float] = (0.02, 0.08)
    current_multipliers: dict[str, float] = field(default_factory=dict)
    cash_conversion: float = 1.0
    drop_gross_profit: bool = False


_ACCOUNTS: dict[str, _Accounts] = {
    "beneish_red": _Accounts(
        growth=(0.35, 0.35),
        current_multipliers={"receivables": 1.8, "gross_profit": 0.85, "net_income": 1.3},
        cash_conversion=0.3,
    ),
    "beneish_gray": _Accounts(
        growth=(0.15, 0.15), current_multipliers={"receivables": 1.4}, cash_conversion=0.6,
    ),
    "high_accruals": _Accounts(
        current_multipliers={"inventory": 2.6, "receivables": 1.1}, cash_conversion=0.9,
    ),
    "bank": _Accounts(drop_gross_profit=True),
}


def profile_for(index: int) -> Profile:
    if index < len(PROFILES):
        return PROFILES[index]
    return Profile(SECTORS[index % len(SECTORS)])


def business_days(end: date, n: int) -> list[str]:
    days: list[str] = []
    d = end
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d.isoformat())
        d -= timedelta(days=1)
    return days[::-1]


class SyntheticProvider:
    def __init__(self, seed: int = 1, n_tickers: int = 20, *, n_days: int = DEFAULT_DAYS) -> None:
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not 1 <= n_tickers <= MAX_TICKERS:
            raise ValueError(f"n_tickers must be between 1 and {MAX_TICKERS}")
        if n_days < 120:
            raise ValueError("n_days must be at least 120")
        self.seed = seed
        self.dates = business_days(END_DATE, n_days)
        market = _market_factor(seed, n_days)
        sector_noise = {
            s: np.random.default_rng([seed, 1, k]).standard_normal(n_days)
            for k, s in enumerate(SECTORS)
        }
        self._universe: list[Ticker] = []
        self._prices: dict[str, list[tuple[str, float]]] = {}
        self._fundamentals: dict[str, dict] = {}
        self._audit: dict[str, dict] = {}
        for i in range(n_tickers):
            self._add_ticker(i, market, sector_noise)

    def _add_ticker(self, i: int, market: np.ndarray, sector_noise: dict[str, np.ndarray]) -> None:
        profile = profile_for(i)
        symbol = f"SYN{i + 1:02d}"
        rng = np.random.default_rng([self.seed, 2, i])
        closes = _price_path(profile, market, sector_noise[profile.sector], rng)
        self._universe.append(Ticker(symbol, profile.sector))
        self._prices[symbol] = list(zip(self.dates, closes))
        self._fundamentals[symbol] = _fundamentals(symbol, profile, rng, END_DATE)
        self._audit[symbol] = _audit_signals(profile.kind, rng)

    def universe(self) -> list[Ticker]:
        return list(self._universe)

    def prices(self, symbol: str) -> list[tuple[str, float]]:
        return list(self._prices[symbol])

    def fundamentals(self, symbol: str) -> dict:
        return self._fundamentals[symbol]

    def audit_signals(self, symbol: str) -> dict | None:
        return dict(self._audit[symbol])


def _market_factor(seed: int, n: int) -> np.ndarray:
    factor = np.random.default_rng([seed, 0]).normal(MARKET_DRIFT, MARKET_VOL, n)
    stress_start, stress_len = int(n * 0.60), int(n * 0.08)
    factor[stress_start:stress_start + stress_len] -= 0.003
    rebound_start = stress_start + stress_len
    factor[rebound_start:rebound_start + int(n * 0.15)] += 0.0012
    return factor


def _price_path(
    profile: Profile, market: np.ndarray, sector: np.ndarray, rng: np.random.Generator,
) -> list[float]:
    n = len(market)
    beta_m, beta_s, vol_s, vol_i = SECTOR_MODEL[profile.sector]
    vol_i *= rng.uniform(0.8, 1.25)
    log_moves = beta_m * market + beta_s * vol_s * sector + rng.normal(0.0, vol_i, n)
    if profile.crash:
        crash_start, crash_len = int(n * 0.30), int(n * 0.15)
        log_moves[crash_start:crash_start + crash_len] -= 0.008
        recovery_start = crash_start + crash_len
        log_moves[recovery_start:recovery_start + int(n * 0.25)] += 0.002
    start = rng.uniform(20.0, 250.0)
    return [round(float(c), 4) for c in start * np.exp(np.cumsum(log_moves))]


def _fundamentals(symbol: str, profile: Profile, rng: np.random.Generator, as_of: date) -> dict:
    spec = _ACCOUNTS.get(profile.kind, _Accounts())
    ratios = {k: rng.uniform(lo, hi) for k, (lo, hi) in _RATIO_RANGES.items()}
    revenue_prior = rng.uniform(1.0, 30.0) * 1e9
    growth = rng.uniform(*spec.growth)
    prior = _year(revenue_prior, ratios, {}, 1.0, rng)
    current = _year(revenue_prior * (1 + growth), ratios, spec.current_multipliers, spec.cash_conversion, rng)
    if spec.drop_gross_profit:
        prior["grossProfit"] = current["grossProfit"] = None

    age_days = 200 if profile.kind == "stale_report" else int(rng.integers(25, 100))
    filing = as_of - timedelta(days=age_days)
    fy_end = filing - timedelta(days=60)
    years = [(fy_end, filing, current), (fy_end - timedelta(days=365), filing - timedelta(days=365), prior)]
    return {
        "General": {"Code": symbol, "Name": f"Synthetic Company {symbol[3:]}", "GicSector": profile.sector},
        "Highlights": {
            "MarketCapitalization": round(float(current["totalRevenue"]) * rng.uniform(1.0, 6.0)),
            "MostRecentQuarter": fy_end.isoformat(),
        },
        "Financials": {
            branch: {"yearly": {fy.isoformat(): _statement(branch, fy, filed, fig) for fy, filed, fig in years}}
            for branch in ("Income_Statement", "Balance_Sheet", "Cash_Flow")
        },
    }


def _year(
    revenue: float, ratios: dict[str, float], multipliers: dict[str, float],
    cash_conversion: float, rng: np.random.Generator,
) -> dict[str, float | None]:
    v = {
        k: revenue * r * multipliers.get(k, 1.0) * rng.uniform(1 - _JITTER, 1 + _JITTER)
        for k, r in ratios.items()
    }
    depreciation = v["ppe"] * _DEPRECIATION_RATE
    current_assets = v["cash"] + v["short_term_investments"] + v["receivables"] + v["inventory"]
    return {
        "totalRevenue": revenue,
        "grossProfit": v["gross_profit"],
        "sellingGeneralAdministrative": v["sga"],
        "depreciationAndAmortization": depreciation,
        "netIncome": v["net_income"],
        "netReceivables": v["receivables"],
        "inventory": v["inventory"],
        "cash": v["cash"],
        "shortTermInvestments": v["short_term_investments"],
        "totalCurrentAssets": current_assets,
        "propertyPlantAndEquipmentNet": v["ppe"],
        "longTermInvestments": v["long_term_investments"],
        "totalAssets": current_assets + v["ppe"] + v["long_term_investments"] + v["other_assets"],
        "shortTermDebt": v["short_term_debt"],
        "totalCurrentLiabilities": v["operating_liabilities"] + v["short_term_debt"],
        "longTermDebt": v["long_term_debt"],
        "totalCashFromOperatingActivities": cash_conversion * v["net_income"] + 0.95 * depreciation,
    }


_STATEMENT_KEYS = {
    "Income_Statement": (
        "totalRevenue", "grossProfit", "sellingGeneralAdministrative",
        "depreciationAndAmortization", "netIncome",
    ),
    "Balance_Sheet": (
        "netReceivables", "inventory", "cash", "shortTermInvestments", "totalCurrentAssets",
        "propertyPlantAndEquipmentNet", "longTermInvestments", "totalAssets",
        "shortTermDebt", "totalCurrentLiabilities", "longTermDebt",
    ),
    "Cash_Flow": ("totalCashFromOperatingActivities",),
}


def _statement(branch: str, fy_end: date, filed: date, figures: dict[str, float | None]) -> dict:
    out: dict[str, str | None] = {"date": fy_end.isoformat(), "filing_date": filed.isoformat()}
    for key in _STATEMENT_KEYS[branch]:
        value = figures[key]
        out[key] = None if value is None else f"{value:.2f}"
    return out


def _audit_signals(kind: str, rng: np.random.Generator) -> dict:
    audit = {
        "auditor_tier": "A" if rng.random() < 0.6 else "B",
        "going_concern": False,
        "restatements_36m": 0,
        "revenue_recognition_restatement": False,
    }
    if kind == "missing_audit":
        del audit["restatements_36m"]
    elif kind == "unverified_audit":
        audit["going_concern"] = None
    elif kind == "restatement":
        audit["restatements_36m"] = 1
        audit["revenue_recognition_restatement"] = True
    elif kind == "going_concern":
        audit["going_concern"] = True
    elif kind == "stale_report":
        audit["auditor_tier"] = "B"
    return audit

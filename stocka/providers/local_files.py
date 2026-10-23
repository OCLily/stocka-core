"""Read prices and fundamentals from local files.

Directory layout:

    DATA_DIR/
      prices.csv               header `date,ticker,close`, ISO dates, one row per close
      fundamentals/
        <TICKER>.json          {"fundamentals": {...}, "audit_signals": {...} or null}

`fundamentals` must have the EODHD shape described in
`stocka.providers.base`; the sector is read from
`fundamentals.General.GicSector`. The universe is the set of JSON files;
every one of them needs prices. `write_local_files` writes any provider in
this layout, which is also the easiest way to see a complete example.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

from stocka.providers.base import DataProvider, Ticker

PRICE_COLUMNS = ("date", "ticker", "close")


class LocalFilesProvider:
    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        prices_path = self.data_dir / "prices.csv"
        fundamentals_dir = self.data_dir / "fundamentals"
        if not prices_path.is_file():
            raise FileNotFoundError(f"missing {prices_path}")
        if not fundamentals_dir.is_dir():
            raise FileNotFoundError(f"missing directory {fundamentals_dir}")
        self._prices = _read_prices(prices_path)
        self._records = {p.stem: _read_record(p) for p in sorted(fundamentals_dir.glob("*.json"))}
        without_prices = sorted(set(self._records) - set(self._prices))
        if without_prices:
            raise ValueError(f"no prices in prices.csv for: {', '.join(without_prices)}")

    def universe(self) -> list[Ticker]:
        return [
            Ticker(symbol, (record["fundamentals"].get("General") or {}).get("GicSector") or "Unknown")
            for symbol, record in self._records.items()
        ]

    def prices(self, symbol: str) -> list[tuple[str, float]]:
        if symbol not in self._prices:
            raise KeyError(f"no prices for {symbol!r}")
        return list(self._prices[symbol])

    def fundamentals(self, symbol: str) -> dict:
        return self._record(symbol)["fundamentals"]

    def audit_signals(self, symbol: str) -> dict | None:
        return self._record(symbol).get("audit_signals")

    def _record(self, symbol: str) -> dict:
        if symbol not in self._records:
            raise KeyError(f"no fundamentals file for {symbol!r}")
        return self._records[symbol]


def _read_prices(path: Path) -> dict[str, list[tuple[str, float]]]:
    rows: dict[str, dict[str, float]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if tuple(reader.fieldnames or ()) != PRICE_COLUMNS:
            raise ValueError(f"{path.name}: header must be {','.join(PRICE_COLUMNS)}")
        for line_no, row in enumerate(reader, start=2):
            day, ticker = row["date"], row["ticker"]
            try:
                date.fromisoformat(day)
                close = float(row["close"])
            except (TypeError, ValueError):
                raise ValueError(f"{path.name}:{line_no}: bad date or close") from None
            if day in rows[ticker]:
                raise ValueError(f"{path.name}:{line_no}: duplicate row for {ticker} on {day}")
            rows[ticker][day] = close
    return {ticker: sorted(by_day.items()) for ticker, by_day in rows.items()}


def _read_record(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        record = json.load(fh)
    if not isinstance(record, dict) or not isinstance(record.get("fundamentals"), dict):
        raise ValueError(f"{path.name}: expected an object with a 'fundamentals' object")
    audit = record.get("audit_signals")
    if audit is not None and not isinstance(audit, dict):
        raise ValueError(f"{path.name}: 'audit_signals' must be an object or null")
    return record


def write_local_files(provider: DataProvider, data_dir: str | Path) -> Path:
    data_dir = Path(data_dir)
    fundamentals_dir = data_dir / "fundamentals"
    fundamentals_dir.mkdir(parents=True, exist_ok=True)
    universe = provider.universe()
    with (data_dir / "prices.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(PRICE_COLUMNS)
        for ticker in universe:
            for day, close in provider.prices(ticker.symbol):
                writer.writerow((day, ticker.symbol, repr(float(close))))
    for ticker in universe:
        record = {
            "fundamentals": provider.fundamentals(ticker.symbol),
            "audit_signals": provider.audit_signals(ticker.symbol),
        }
        path = fundamentals_dir / f"{ticker.symbol}.json"
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data_dir

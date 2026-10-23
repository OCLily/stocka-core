"""SyntheticProvider and LocalFilesProvider: determinism, shape, guard profiles, round trip."""

import json
import math
import statistics
import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np

from brain.risk.drawdown import max_drawdown_from_closes
from brain.scoring.beneish import BENEISH_THRESHOLD_GRAY_LOW, BENEISH_THRESHOLD_RED, compute_beneish_m_score
from brain.scoring.dq_score import SLOAN_Q5_ABSOLUTE_VETO_THRESHOLD
from brain.scoring.sloan import compute_sloan_accruals
from stocka.providers import DataProvider, LocalFilesProvider, SyntheticProvider, write_local_files
from stocka.providers.synthetic import (
    DEFAULT_DAYS,
    END_DATE,
    SECTORS,
    Profile,
    _market_factor,
    _price_path,
    business_days,
)


class SyntheticProviderTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.provider = SyntheticProvider(seed=3, n_tickers=20)

    def test_satisfies_the_data_provider_protocol(self):
        self.assertIsInstance(self.provider, DataProvider)

    def test_same_seed_gives_identical_data(self):
        other = SyntheticProvider(seed=3, n_tickers=20)
        for t in self.provider.universe():
            self.assertEqual(self.provider.prices(t.symbol), other.prices(t.symbol))
            self.assertEqual(self.provider.fundamentals(t.symbol), other.fundamentals(t.symbol))
            self.assertEqual(self.provider.audit_signals(t.symbol), other.audit_signals(t.symbol))

    def test_different_seed_gives_different_prices(self):
        other = SyntheticProvider(seed=4, n_tickers=20)
        self.assertNotEqual(self.provider.prices("SYN01"), other.prices("SYN01"))

    def test_universe_has_requested_size_names_and_gics_sectors(self):
        universe = self.provider.universe()
        self.assertEqual([t.symbol for t in universe], [f"SYN{i:02d}" for i in range(1, 21)])
        self.assertTrue(all(t.sector in SECTORS for t in universe))

    def test_prices_are_positive_closes_on_ascending_weekdays(self):
        for t in self.provider.universe():
            series = self.provider.prices(t.symbol)
            days = [d for d, _ in series]
            self.assertEqual(days, sorted(days))
            self.assertEqual(len(set(days)), len(days))
            self.assertTrue(all(date.fromisoformat(d).weekday() < 5 for d in days))
            self.assertTrue(all(c > 0 for _, c in series))

    def test_smaller_universe_is_a_prefix_of_a_larger_one(self):
        small = SyntheticProvider(seed=3, n_tickers=4)
        for t in small.universe():
            self.assertEqual(small.prices(t.symbol), self.provider.prices(t.symbol))
            self.assertEqual(small.fundamentals(t.symbol), self.provider.fundamentals(t.symbol))

    def test_tickers_beyond_the_profile_list_are_generated(self):
        self.assertEqual(len(SyntheticProvider(seed=0, n_tickers=30).universe()), 30)

    def test_fundamentals_feed_the_engine_calculators(self):
        for t in self.provider.universe():
            fundamentals = self.provider.fundamentals(t.symbol)
            self.assertEqual(fundamentals["General"]["GicSector"], t.sector)
            self.assertTrue(compute_sloan_accruals(fundamentals)["complete"], t.symbol)
            if t.symbol != "SYN09":
                self.assertTrue(compute_beneish_m_score(fundamentals)["complete"], t.symbol)

    def test_bank_profile_has_no_gross_profit_so_beneish_is_incomplete(self):
        beneish = compute_beneish_m_score(self.provider.fundamentals("SYN09"))
        self.assertFalse(beneish["complete"])
        self.assertIn("Income.grossProfit[t]", beneish["missing_inputs"])

    def test_guard_profiles_trip_their_guards_for_many_seeds(self):
        for seed in range(10):
            p = SyntheticProvider(seed=seed, n_tickers=20)
            with self.subTest(seed=seed):
                self.assertGreater(compute_beneish_m_score(p.fundamentals("SYN03"))["m_score"], BENEISH_THRESHOLD_RED)
                gray = compute_beneish_m_score(p.fundamentals("SYN12"))["m_score"]
                self.assertTrue(BENEISH_THRESHOLD_GRAY_LOW <= gray <= BENEISH_THRESHOLD_RED)
                accruals = compute_sloan_accruals(p.fundamentals("SYN05"))["accruals_ratio"]
                self.assertGreater(accruals, SLOAN_Q5_ABSOLUTE_VETO_THRESHOLD)
                self.assertLess(compute_beneish_m_score(p.fundamentals("SYN05"))["m_score"], BENEISH_THRESHOLD_GRAY_LOW)
                self.assertNotIn("restatements_36m", p.audit_signals("SYN10"))
                self.assertIsNone(p.audit_signals("SYN13")["going_concern"])
                self.assertEqual(p.audit_signals("SYN14")["restatements_36m"], 1)
                self.assertTrue(p.audit_signals("SYN18")["going_concern"])

    def test_plain_tickers_stay_in_the_clear_zone(self):
        for seed in range(10):
            p = SyntheticProvider(seed=seed, n_tickers=20)
            for symbol in ("SYN01", "SYN02", "SYN04", "SYN06", "SYN07", "SYN08", "SYN11", "SYN17", "SYN19", "SYN20"):
                with self.subTest(seed=seed, symbol=symbol):
                    fundamentals = p.fundamentals(symbol)
                    m_score = compute_beneish_m_score(fundamentals)["m_score"]
                    ratio = compute_sloan_accruals(fundamentals)["accruals_ratio"]
                    self.assertLess(m_score, BENEISH_THRESHOLD_GRAY_LOW)
                    self.assertLess(ratio, SLOAN_Q5_ABSOLUTE_VETO_THRESHOLD)

    def test_crash_ticker_has_a_deep_drawdown(self):
        for seed in range(5):
            with self.subTest(seed=seed):
                dd = max_drawdown_from_closes(SyntheticProvider(seed=seed, n_tickers=8).prices("SYN08"))
                self.assertLess(dd["max_dd_pct"], -35.0)

    def test_crash_injection_lowers_the_crash_window_and_deepens_the_drawdown(self):
        n = DEFAULT_DAYS
        market = _market_factor(3, n)
        sector = np.random.default_rng([3, 1, SECTORS.index("Energy")]).standard_normal(n)
        crash = _price_path(Profile("Energy", crash=True), market, sector, np.random.default_rng(11))
        plain = _price_path(Profile("Energy"), market, sector, np.random.default_rng(11))
        start, length = int(n * 0.30), int(n * 0.15)
        self.assertEqual(crash[:start], plain[:start])
        end = start + length - 1
        window_gap = math.log(crash[end] / crash[start - 1]) - math.log(plain[end] / plain[start - 1])
        self.assertAlmostEqual(window_gap, -0.008 * length, places=3)
        days = business_days(END_DATE, n)
        dd_crash = max_drawdown_from_closes(list(zip(days, crash)))["max_dd_pct"]
        dd_plain = max_drawdown_from_closes(list(zip(days, plain)))["max_dd_pct"]
        self.assertLess(dd_crash, dd_plain)

    def test_volatility_differs_across_tickers(self):
        vols = []
        for symbol in ("SYN06", "SYN18"):
            closes = [c for _, c in self.provider.prices(symbol)]
            vols.append(statistics.stdev(b / a - 1 for a, b in zip(closes, closes[1:])))
        self.assertGreater(vols[1], 1.5 * vols[0])

    def test_audit_signals_are_returned_as_copies(self):
        self.provider.audit_signals("SYN01")["going_concern"] = True
        self.assertFalse(self.provider.audit_signals("SYN01")["going_concern"])

    def test_invalid_arguments_raise(self):
        for kwargs in ({"seed": -1}, {"n_tickers": 0}, {"n_tickers": 100}, {"n_days": 50}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    SyntheticProvider(**kwargs)


class LocalFilesProviderTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write_minimal(self, prices="date,ticker,close\n2025-01-02,AAA,10.5\n2025-01-03,AAA,10.7\n", record=None):
        (self.data_dir / "prices.csv").write_text(prices, encoding="utf-8")
        (self.data_dir / "fundamentals").mkdir(exist_ok=True)
        if record is None:
            record = {"fundamentals": {"General": {"GicSector": "Energy"}}, "audit_signals": None}
        (self.data_dir / "fundamentals" / "AAA.json").write_text(json.dumps(record), encoding="utf-8")

    def test_round_trip_reproduces_the_synthetic_provider(self):
        source = SyntheticProvider(seed=5, n_tickers=6)
        write_local_files(source, self.data_dir)
        loaded = LocalFilesProvider(self.data_dir)
        self.assertIsInstance(loaded, DataProvider)
        self.assertEqual(loaded.universe(), source.universe())
        for t in source.universe():
            self.assertEqual(loaded.prices(t.symbol), source.prices(t.symbol))
            self.assertEqual(loaded.fundamentals(t.symbol), source.fundamentals(t.symbol))
            self.assertEqual(loaded.audit_signals(t.symbol), source.audit_signals(t.symbol))

    def test_minimal_files_are_read(self):
        self._write_minimal()
        provider = LocalFilesProvider(self.data_dir)
        self.assertEqual([(t.symbol, t.sector) for t in provider.universe()], [("AAA", "Energy")])
        self.assertEqual(provider.prices("AAA"), [("2025-01-02", 10.5), ("2025-01-03", 10.7)])
        self.assertIsNone(provider.audit_signals("AAA"))

    def test_price_rows_are_sorted_by_date(self):
        self._write_minimal(prices="date,ticker,close\n2025-01-03,AAA,2\n2025-01-02,AAA,1\n")
        self.assertEqual(LocalFilesProvider(self.data_dir).prices("AAA"), [("2025-01-02", 1.0), ("2025-01-03", 2.0)])

    def test_missing_sector_becomes_unknown(self):
        self._write_minimal(record={"fundamentals": {}})
        self.assertEqual(LocalFilesProvider(self.data_dir).universe()[0].sector, "Unknown")

    def test_missing_prices_file_raises(self):
        (self.data_dir / "fundamentals").mkdir()
        with self.assertRaises(FileNotFoundError):
            LocalFilesProvider(self.data_dir)

    def test_missing_fundamentals_directory_raises(self):
        (self.data_dir / "prices.csv").write_text("date,ticker,close\n", encoding="utf-8")
        with self.assertRaises(FileNotFoundError):
            LocalFilesProvider(self.data_dir)

    def test_wrong_csv_header_raises(self):
        self._write_minimal(prices="day,symbol,price\n2025-01-02,AAA,1\n")
        with self.assertRaises(ValueError):
            LocalFilesProvider(self.data_dir)

    def test_bad_close_value_raises(self):
        self._write_minimal(prices="date,ticker,close\n2025-01-02,AAA,n.a.\n")
        with self.assertRaises(ValueError):
            LocalFilesProvider(self.data_dir)

    def test_bad_date_raises(self):
        self._write_minimal(prices="date,ticker,close\n02.01.2025,AAA,1\n")
        with self.assertRaises(ValueError):
            LocalFilesProvider(self.data_dir)

    def test_duplicate_price_rows_raise(self):
        self._write_minimal(prices="date,ticker,close\n2025-01-02,AAA,1\n2025-01-02,AAA,2\n")
        with self.assertRaises(ValueError):
            LocalFilesProvider(self.data_dir)

    def test_ticker_without_prices_raises(self):
        self._write_minimal(prices="date,ticker,close\n2025-01-02,BBB,1\n")
        with self.assertRaises(ValueError):
            LocalFilesProvider(self.data_dir)

    def test_record_without_fundamentals_object_raises(self):
        self._write_minimal(record={"audit_signals": {}})
        with self.assertRaises(ValueError):
            LocalFilesProvider(self.data_dir)

    def test_unknown_ticker_raises_key_error(self):
        self._write_minimal()
        provider = LocalFilesProvider(self.data_dir)
        with self.assertRaises(KeyError):
            provider.prices("ZZZ")
        with self.assertRaises(KeyError):
            provider.fundamentals("ZZZ")


if __name__ == "__main__":
    unittest.main()

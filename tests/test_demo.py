"""Demo end to end: report content, the no-performance-figures rule, CLI."""

import contextlib
import io
import re
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from stocka import __version__
from stocka.cli import main
from stocka.config import Config, DemoSettings
from stocka.demo import run_demo
from stocka.providers import SyntheticProvider, write_local_files

FORBIDDEN = re.compile(r"return|cagr|sharpe|sortino|win.?rate|hit.?rate", re.IGNORECASE)
NO_BOOTSTRAP = Config(demo=DemoSettings(bootstrap_sims=0))


def demo_report(tickers, *, seed=1, config=NO_BOOTSTRAP):
    config = config.with_overrides(seed=seed, tickers=tickers)
    return run_demo(SyntheticProvider(seed, tickers), config)


def table_row(report, ticker):
    return next(line for line in report.splitlines() if line.startswith(f"| {ticker} |"))


class DemoReportTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.report = demo_report(20)

    def test_first_line_says_synthetic_not_market_data(self):
        self.assertIn("synthetic data, not market data", self.report.splitlines()[0])

    def test_report_contains_no_performance_vocabulary(self):
        self.assertIsNone(FORBIDDEN.search(self.report))

    def test_report_has_the_three_sections_and_every_ticker(self):
        for heading in ("## 1. Data-quality screen", "## 2. Basket risk", "## 3. Forecast calibration"):
            self.assertIn(heading, self.report)
        for i in range(1, 21):
            table_row(self.report, f"SYN{i:02d}")

    def test_basket_risk_lists_the_required_measures(self):
        for label in ("Annualised volatility", "Maximum drawdown", "Expected shortfall 97.5 %",
                      "### Correlated pairs", "### Concentration"):
            self.assertIn(label, self.report)

    def test_guard_tickers_fail_for_their_reasons(self):
        expected = {
            "SYN03": "engine veto: beneish_red_flag",
            "SYN05": "engine veto: sloan_q5_anomaly",
            "SYN10": "audit input rejected: missing restatements_36m",
            "SYN13": "unverified input: going_concern_unverified",
            "SYN14": "engine veto: restatement_36m",
            "SYN18": "engine veto: going_concern",
        }
        for ticker, reason in expected.items():
            with self.subTest(ticker=ticker):
                row = table_row(self.report, ticker)
                self.assertIn("| fail |", row)
                self.assertIn(reason, row)

    def test_plain_ticker_passes(self):
        self.assertIn("| pass |", table_row(self.report, "SYN01"))

    def test_same_seed_gives_identical_report(self):
        self.assertEqual(demo_report(20), self.report)

    def test_tiny_universe_runs_without_sloan_quintiles(self):
        report = demo_report(2)
        self.assertIn("synthetic data, not market data", report.splitlines()[0])
        self.assertIsNone(FORBIDDEN.search(report))
        self.assertIn("Sloan quintile unavailable", report)

    def test_single_ticker_skips_correlation(self):
        report = demo_report(1)
        self.assertIn("1 of 1 tickers pass the screen.", report)
        self.assertIn("Not computed: needs two tickers", report)

    def test_no_passing_ticker_means_no_basket(self):
        config = replace(NO_BOOTSTRAP, screen=replace(NO_BOOTSTRAP.screen, max_beneish_m=-10.0))
        report = demo_report(3, config=config)
        self.assertIn("No ticker passed the screen", report)
        self.assertIsNone(FORBIDDEN.search(report))

    def test_bootstrap_drawdown_appears_when_enabled(self):
        report = demo_report(4, config=Config(demo=DemoSettings(bootstrap_sims=200)))
        self.assertIn("Bootstrap max drawdown, P95 / P99", report)
        self.assertIsNone(FORBIDDEN.search(report))


class CliTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.config = self.dir / "fast.toml"
        self.config.write_text("[demo]\nbootstrap_sims = 0\n", encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def test_version_flag_prints_the_version(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as exit_info:
            main(["--version"])
        self.assertEqual(exit_info.exception.code, 0)
        self.assertIn(__version__, out.getvalue())

    def test_demo_prints_the_report_to_stdout(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["demo", "--tickers", "3", "--seed", "2", "--config", str(self.config)])
        self.assertEqual(code, 0)
        self.assertIn("synthetic data, not market data", out.getvalue().splitlines()[0])

    def test_demo_writes_the_report_to_the_out_file(self):
        target = self.dir / "report.md"
        with contextlib.redirect_stderr(io.StringIO()):
            code = main(["demo", "--tickers", "3", "--config", str(self.config), "--out", str(target)])
        self.assertEqual(code, 0)
        self.assertIn("synthetic data, not market data", target.read_text(encoding="utf-8").splitlines()[0])

    def test_data_dir_reads_local_files(self):
        data_dir = write_local_files(SyntheticProvider(seed=2, n_tickers=4), self.dir / "data")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["demo", "--data-dir", str(data_dir), "--config", str(self.config)])
        self.assertEqual(code, 0)
        report = out.getvalue()
        self.assertIn("user-supplied data", report.splitlines()[0])
        self.assertIn("read from local files", report)
        self.assertIsNone(FORBIDDEN.search(report))

    def test_invalid_settings_exit_with_code_2(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = main(["demo", "--tickers", "0", "--config", str(self.config)])
        self.assertEqual(code, 2)
        self.assertIn("demo.tickers", err.getvalue())

    def test_missing_data_dir_exits_with_code_2(self):
        with contextlib.redirect_stderr(io.StringIO()):
            code = main(["demo", "--data-dir", str(self.dir / "absent"), "--config", str(self.config)])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()

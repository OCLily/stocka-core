"""Config loading and validation."""

import tempfile
import unittest
from pathlib import Path

from stocka.config import Config, ConfigError, DemoSettings, ScreenSettings, config_from_dict, load_config

ROOT = Path(__file__).resolve().parents[1]


class ConfigLoadingTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, text):
        path = self.dir / "config.toml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_no_path_gives_the_defaults(self):
        self.assertEqual(load_config(None), Config())

    def test_example_file_equals_the_defaults(self):
        self.assertEqual(load_config(ROOT / "config" / "example.toml"), Config())

    def test_partial_file_keeps_other_defaults(self):
        config = load_config(self._write("[screen]\nmin_dq_score = 80\n"))
        self.assertEqual(config.screen.min_dq_score, 80)
        self.assertEqual(config.screen.max_beneish_m, ScreenSettings().max_beneish_m)
        self.assertEqual(config.demo, DemoSettings())

    def test_integer_is_accepted_for_a_float_field(self):
        config = load_config(self._write("[screen]\nmax_beneish_m = -2\n"))
        self.assertEqual(config.screen.max_beneish_m, -2.0)
        self.assertIsInstance(config.screen.max_beneish_m, float)

    def test_invalid_toml_raises_config_error(self):
        with self.assertRaises(ConfigError):
            load_config(self._write("[screen\nmin_dq_score = 80\n"))

    def test_missing_file_raises_config_error(self):
        with self.assertRaises(ConfigError):
            load_config(self.dir / "absent.toml")


class ConfigValidationTests(unittest.TestCase):

    def test_unknown_section_is_rejected(self):
        with self.assertRaisesRegex(ConfigError, "unknown section"):
            config_from_dict({"risk": {}})

    def test_unknown_key_is_rejected(self):
        with self.assertRaisesRegex(ConfigError, "unknown key"):
            config_from_dict({"screen": {"max_drawdown": 0.2}})

    def test_expected_shortfall_level_is_not_configurable(self):
        with self.assertRaisesRegex(ConfigError, "unknown key"):
            config_from_dict({"screen": {"es_level": 0.99}})

    def test_section_must_be_a_table(self):
        with self.assertRaises(ConfigError):
            config_from_dict({"demo": 3})

    def test_wrong_types_are_rejected(self):
        cases = [
            {"demo": {"seed": True}},
            {"demo": {"tickers": 5.0}},
            {"screen": {"min_dq_score": "70"}},
            {"screen": {"max_beneish_m": "low"}},
            {"screen": {"max_beneish_m": float("nan")}},
            {"screen": {"exclude_sloan_q5": 1}},
            {"screen": {"require_verified_audit": "yes"}},
        ]
        for data in cases:
            with self.subTest(data=data):
                with self.assertRaises(ConfigError):
                    config_from_dict(data)

    def test_out_of_range_values_are_rejected(self):
        cases = [
            {"demo": {"seed": -1}},
            {"demo": {"tickers": 0}},
            {"demo": {"tickers": 100}},
            {"demo": {"bootstrap_sims": 50}},
            {"demo": {"bootstrap_sims": 20_001}},
            {"screen": {"min_dq_score": 101}},
            {"screen": {"min_dq_score": -1}},
        ]
        for data in cases:
            with self.subTest(data=data):
                with self.assertRaises(ConfigError):
                    config_from_dict(data)

    def test_bootstrap_can_be_switched_off(self):
        self.assertEqual(config_from_dict({"demo": {"bootstrap_sims": 0}}).demo.bootstrap_sims, 0)

    def test_overrides_replace_only_the_given_values(self):
        config = Config().with_overrides(seed=9)
        self.assertEqual(config.demo.seed, 9)
        self.assertEqual(config.demo.tickers, DemoSettings().tickers)
        self.assertEqual(Config().with_overrides(), Config())

    def test_overrides_are_validated(self):
        with self.assertRaises(ConfigError):
            Config().with_overrides(tickers=0)


if __name__ == "__main__":
    unittest.main()

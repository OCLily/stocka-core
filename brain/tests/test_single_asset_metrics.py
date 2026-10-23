import unittest
from datetime import date, timedelta


def _dates(n):
    start = date(2020, 1, 1)
    out = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def _synthetic_dd_series():
    """oldest-first [(date_iso, adj_close)] — rise, -25% drawdown, partial recovery.

    Peak=100 -> trough=75 (exactly -25% MaxDD) -> partial recovery to 90, plus
    enough bars (>=60) so vol/downside-dev are computable (n >= _MIN_OBS).
    """
    closes = [100.0]
    # rise phase
    for i in range(20):
        closes.append(closes[-1] * 1.01)
    peak = closes[-1]
    # drawdown phase: descend to exactly -25% off peak
    trough = peak * 0.75
    steps_down = 15
    for i in range(1, steps_down + 1):
        closes.append(peak - (peak - trough) * i / steps_down)
    # partial recovery, staying underwater (never reclaims peak)
    for i in range(30):
        closes.append(closes[-1] * 1.003)
    dates = _dates(len(closes))
    return list(zip(dates, closes))


def _short_series(n=5):
    closes = [100.0, 101.0, 99.0, 102.0, 98.0][:n]
    dates = _dates(len(closes))
    return list(zip(dates, closes))


class Tier1MetricsTest(unittest.TestCase):
    def test_recovery_and_maxdd_from_series(self):
        from brain.risk.single_asset_metrics import single_asset_metrics
        series = _synthetic_dd_series()
        m = single_asset_metrics(series)                 # include_reward default False
        # Fixture descends in 15 equal steps to exactly peak*0.75, so 0.25 is
        # analytically exact; only float noise (~5e-17) is admissible tolerance.
        self.assertAlmostEqual(m["max_drawdown"], 0.25, delta=1e-12)
        self.assertIsInstance(m["recovery_underwater_bars_longest"], int)
        self.assertIsNone(m.get("sharpe"))               # reward off by default
        self.assertIsNone(m.get("sortino"))
        self.assertIsNone(m.get("calmar"))

    def test_rf_zero_is_declared(self):
        from brain.risk.single_asset_metrics import RF, single_asset_metrics
        self.assertEqual(RF, 0.0)                          # rf=0 explicit

    def test_short_history_is_none_not_zero(self):
        from brain.risk.single_asset_metrics import single_asset_metrics
        m = single_asset_metrics(_short_series(n=5))       # < _MIN_OBS (=60, risk_profile.py)
        self.assertIsNone(m["vol_annual"])                 # n<60: Vol nicht berechenbar -> None, nie 0
        self.assertIsNotNone(m["notes"])                   # honest note ("nicht berechenbar (n=5)")
        self.assertIsInstance(m["max_drawdown"], float)    # MaxDD/Recovery sind auch bei kurzer Serie definiert
        self.assertIsInstance(m["recovery_underwater_bars_longest"], int)

    def test_reward_axes_only_with_include_reward(self):
        from brain.risk.single_asset_metrics import single_asset_metrics
        series = _synthetic_dd_series()
        m_off = single_asset_metrics(series)                       # default
        m_on = single_asset_metrics(series, include_reward=True)   # explicit opt-in
        self.assertIsNone(m_off["sharpe"])
        self.assertIsNotNone(m_on["sharpe"])                       # rf=0 sharpe computable on this series

    def test_maxdd_n_lt_2_is_fail_open_zero_not_none(self):
        # single_asset_metrics wires compute_max_drawdown (brain/backtest/metrics.py),
        # which is fail-open for n<2: (0.0, None, None) — "no drawdown data" and "zero
        # drawdown" are indistinguishable, unlike brain/risk/drawdown.py's fail-closed
        # n<2 -> error. Documented here so the gap is visible; whether single_asset_metrics
        # should be rewired onto drawdown.py is a separate decision, not made here.
        from brain.risk.single_asset_metrics import single_asset_metrics
        m_empty = single_asset_metrics([])
        self.assertEqual(m_empty["max_drawdown"], 0.0)
        self.assertIsNone(m_empty["max_drawdown_peak"])
        m_one = single_asset_metrics(_short_series(n=1))
        self.assertEqual(m_one["max_drawdown"], 0.0)
        self.assertIsNone(m_one["max_drawdown_peak"])

    def test_maxdd_silently_ignores_nan_bar(self):
        # A NaN close mid-series is silently excluded from max-drawdown (every NaN
        # comparison in compute_max_drawdown's peak/dd tracking is False) rather than
        # raising or propagating NaN — fail-open, unlike bounded_returns (vol path),
        # which explicitly excludes non-finite bars via math.isfinite.
        import math
        from brain.risk.single_asset_metrics import single_asset_metrics
        series = _synthetic_dd_series()
        contaminated = list(series)
        d, _ = contaminated[10]
        contaminated[10] = (d, math.nan)
        m = single_asset_metrics(contaminated)
        self.assertAlmostEqual(m["max_drawdown"], 0.25, places=2)


if __name__ == "__main__":
    unittest.main()

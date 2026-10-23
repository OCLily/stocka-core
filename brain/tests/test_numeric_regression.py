"""Referenzwert-Regression fuer den Drawdown-/CVaR-Pfad.

ZWECK: rot werden, wenn sich die Numerik des Drawdown-/CVaR-Pfads still
veraendert, etwa durch ein numpy-Upgrade. Die Drawdown-Stufen der persoenlichen
Kapitalregeln (-22/-32/-42 %) haengen an genau diesen Zahlen; eine Aenderung,
die sie um 0,3 Punkte verschiebt, darf nicht unbemerkt durchlaufen.

Wichtig fuer den Leser: der live gerenderte Drawdown-/CVaR-Pfad ist numpy-frei.
`max_drawdown_from_closes` (brain/risk/drawdown.py) und `compute_es_historical`
(brain/risk/es_historical.py) importieren beide nur `math` aus der Stdlib. Ein
numpy-2->3-Sprung kann sie per Konstruktion nicht verschieben. Die numpy-tragende
DD-Rechnung ist `bootstrap_sim` (np.random.default_rng / np.cumprod /
np.maximum.accumulate / np.percentile) -- im privaten System ohne Produktiv-Caller
(in diesem Auszug ruft sie nur die Demo auf), aber der einzige Ort im Risk-Paket,
an dem eine geaenderte numpy-Semantik ueberhaupt beobachtbar waere.

Der Test pinnt deshalb BEIDE Haelften:
  A) die live gerenderten Zahlen (pure Python) -- pinnt den Kapitalpfad selbst,
  B) den numpy-Pfad (bootstrap_sim) -- pinnt die Bibliotheks-Semantik.

Ohne (B) waere der Test numerik-blind; ohne (A) wuerde er den Kapitalpfad nicht
abdecken. Eines der beiden allein wuerde den Zweck verfehlen.

Was (B) garantiert und was nicht: die Referenzwerte sind unter numpy 2.5.x
gemessen. Innerhalb einer numpy-Version ist der Pfad mit festem Seed
deterministisch, der Test also nicht flaky. Versionsuebergreifend garantiert
numpy (NEP 19) jedoch keinen stabilen Strom fuer Generator-Methoden wie
`geometric` oder `integers` und kein unveraendertes `np.percentile`: ein
numpy-Feature-Release darf diese Werte legitim verschieben. Dann gilt der
Absatz unten -- pruefen, ob die Verschiebung erklaerbar ist, und neu pinnen.
Fuer `arch` gibt es keine fixierte Referenz -- der arch-Pfad wird nur auf
Selbst-Reproduzierbarkeit und darauf geprueft, dass er ueberhaupt genommen wird
(also arch importierbar ist).

Eingabe ist eine hartkodierte, in reinem Python erzeugte LCG-Reihe (kein
numpy-RNG in der Eingabe -- sonst pruefte der Test numpy gegen sich selbst) mit
vier eingebauten -11-%-Schocktagen, damit Tail-Schaetzer und Drawdown wirklich
etwas zu tun haben (max-DD ~-40 %, ES-97.5 % ~3,4 % gegen VaR-97.5 % ~1,2 %).

Bei einem BEWUSSTEN Upgrade, das eine dieser Zahlen legitim verschiebt: NICHT
die Toleranz aufweichen. Die neue Zahl mit der ausloesenden Paketversion hier
eintragen und die Verschiebung im PR benennen -- das ist der Review-Moment, fuer
den dieser Test existiert.
"""
from __future__ import annotations

import unittest

from brain.risk.bootstrap_sim import bootstrap_dd_sim
from brain.risk.drawdown import max_drawdown_from_closes
from brain.risk.es_historical import compute_es_historical

# Referenzwerte gemessen unter numpy 2.5.2 und arch 8.0.0, Python 3.14.7.
_REF_MAX_DD_PCT = -39.9751
_REF_CURRENT_DD_PCT = -37.0549
_REF_N_BARS = 801

# es_historical.py uses the floor+1 k-convention of the shared
# brain/risk/_tail_k.py helper (Acerbi/Tasche's plain floor would give
# worst_k 18 / 7 instead of 19 / 8). VaR/ES measured against the same
# fixed_returns() series and package versions as above.
_REF_VAR_975_PCT = 1.2285
_REF_ES_975_PCT = 3.2698
_REF_WORST_K_975 = 19
_REF_VAR_990_PCT = 1.2524
_REF_ES_990_PCT = 6.0617
_REF_WORST_K_990 = 8

_REF_P95_DD = -0.6857986979964198
_REF_P99_DD = -0.7623183190933727
_REF_CDAR_95PLUS = -0.7308680686036233

# places=10 auf Werte der Groessenordnung 0.7 => zulaessige Abweichung 5e-11,
# also wenige ULP. Eng genug, dass jede Semantik-Aenderung (Promotion-Regeln,
# geaenderte Reduktions-Reihenfolge, anderer Percentile-Interpolator) rot wird,
# und weit genug, dass reine Fused-Multiply-Add-Unterschiede zwischen x86_64 und
# arm64 nicht falsch-rot faerben.
_PLACES = 10


def fixed_returns(n: int = 800) -> list:
    """Deterministische Tages-Return-Reihe, reines Python (kein numpy-RNG).

    LCG (glibc-Parameter) + vier -11-%-Schocktage. Bewusst hier definiert und
    nicht als Fixture-Datei: die Reihe IST Teil der Referenz.
    """
    out = []
    state = 20260827
    for i in range(n):
        state = (1103515245 * state + 12345) % (2 ** 31)
        u = state / (2 ** 31)
        shock = -0.11 if i in (120, 121, 315, 640) else 0.0
        out.append(round(0.0004 + 0.013 * (u - 0.5) * 2.0 + shock, 12))
    return out


def fixed_closes(returns) -> list:
    """[(date_iso, adj_close)] aus der Return-Reihe, Startkurs 100.0."""
    px = 100.0
    out = [("2023-01-03", round(px, 10))]
    d = 3
    for r in returns:
        px *= (1.0 + r)
        d += 1
        out.append((f"2023-{1 + (d // 28) % 12:02d}-{1 + d % 28:02d}", round(px, 10)))
    return out


class TestLiveDrawdownReference(unittest.TestCase):
    """(A) Live-Pfad: brain/risk/drawdown.py — pure Python, exakt reproduzierbar."""

    def test_max_drawdown_reference_value(self):
        r = max_drawdown_from_closes(fixed_closes(fixed_returns()))
        self.assertIsNone(r["error"])
        self.assertEqual(r["n_bars"], _REF_N_BARS)
        self.assertEqual(r["max_dd_pct"], _REF_MAX_DD_PCT)
        self.assertEqual(r["current_dd_pct"], _REF_CURRENT_DD_PCT)

    def test_max_drawdown_is_in_drawdown_tier_territory(self):
        """Die Referenz ist bewusst so gewaehlt, dass sie zwischen der -32-%- und
        der -42-%-Drawdown-Stufe liegt: eine Numerik-Verschiebung ist hier keine
        Kosmetik, sie kann eine Stufe kippen."""
        r = max_drawdown_from_closes(fixed_closes(fixed_returns()))
        self.assertLess(r["max_dd_pct"], -32.0)
        self.assertGreater(r["max_dd_pct"], -42.0)


class TestLiveEsReference(unittest.TestCase):
    """(A) Live-Pfad: brain/risk/es_historical.py — pure Python, exakt reproduzierbar."""

    def test_es_historical_reference_values(self):
        es = compute_es_historical(fixed_returns())
        self.assertEqual(es["n_obs"], 756)
        self.assertEqual(es["var_975_pct"], _REF_VAR_975_PCT)
        self.assertEqual(es["es_975_pct"], _REF_ES_975_PCT)
        self.assertEqual(es["worst_k_975"], _REF_WORST_K_975)
        self.assertEqual(es["var_990_pct"], _REF_VAR_990_PCT)
        self.assertEqual(es["es_990_pct"], _REF_ES_990_PCT)
        self.assertEqual(es["worst_k_990"], _REF_WORST_K_990)

    def test_es_tail_is_non_degenerate(self):
        """ES muss deutlich ueber VaR liegen — sonst waere die Referenz so flach,
        dass jede Implementierung sie traefe (keine triviale Eingabe)."""
        es = compute_es_historical(fixed_returns())
        self.assertGreater(es["es_975_pct"], es["var_975_pct"] * 2.0)


class TestNumpyDrawdownReference(unittest.TestCase):
    """(B) numpy-Pfad: bootstrap_sim — hier waere ein Semantik-Wechsel sichtbar."""

    def _sim(self, **kw):
        params = dict(n_sims=2000, block_len=5, rng_seed=20260827, use_arch=False)
        params.update(kw)
        return bootstrap_dd_sim(fixed_returns(), **params)

    def test_numpy_bootstrap_reference_values(self):
        sim = self._sim()
        self.assertEqual(sim["method"], "numpy_fallback")
        self.assertAlmostEqual(sim["p95_dd"], _REF_P95_DD, places=_PLACES)
        self.assertAlmostEqual(sim["p99_dd"], _REF_P99_DD, places=_PLACES)
        self.assertAlmostEqual(sim["cdar_95plus"], _REF_CDAR_95PLUS, places=_PLACES)

    def test_numpy_bootstrap_is_seed_deterministic(self):
        self.assertEqual(self._sim()["p95_dd"], self._sim()["p95_dd"])

    def test_arch_path_is_taken_and_self_reproducible(self):
        """Fuer arch gibt es keine fixierte Referenzzahl, aber der
        Pfad muss genommen werden (arch importierbar) und mit gleichem Seed
        denselben Wert liefern."""
        a = bootstrap_dd_sim(fixed_returns(), n_sims=500, block_len=5,
                             rng_seed=7, use_arch=True)
        b = bootstrap_dd_sim(fixed_returns(), n_sims=500, block_len=5,
                             rng_seed=7, use_arch=True)
        self.assertEqual(a["method"], "arch")
        self.assertEqual(a["p95_dd"], b["p95_dd"])


if __name__ == "__main__":
    unittest.main()

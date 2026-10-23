"""Correlation — Pearson-Rolling (default 60d) auf paarweisen Return-Series.

Scope: dieses Modul implementiert NUR die Pearson-Rolling-Korrelation, den
bewusst einfachen Fallback statt eines DCC-GARCH-Modells (DCC-GARCH via `arch`
ist nicht gebaut). Ein Cluster-Alarm ist NICHT implementiert — weder hier
noch sonstwo in diesem Auszug. Es fehlen alle Bestandteile ausser der
Paar-Kante: (a) die Graph-Komponenten-Bildung, die aus Kanten erst einen
Cluster macht; (b) eine kombinierte Sektor-plus-Korrelations-Bedingung aus
den privaten Regeln des Autors (das Modul kennt weder Sektoren noch eine
Mindest-Clustergroesse); (c) ein Konzentrations-Alert aus denselben Regeln
(das Modul kennt keine Positionsgewichte).

pairwise_correlation_matrix, _pearson und CORRELATION_WINDOW_DEFAULT dienen
einer reinen Messung; find_correlation_clusters und pearson_rolling_correlation
haben im privaten System keinen Produktions-Aufrufer. In diesem Auszug ruft die
Demo pairwise_correlation_matrix und find_correlation_clusters auf. Eine
Sektor-Obergrenze ersetzt den fehlenden Cluster-Alarm nicht: sie misst
GICS-Etiketten, nicht Kovarianz.

Pure-function — caller liefert pre-fetched returns. Kein Direct-Provider-Import.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence


# Implementierungswahl, KEIN aus einer Spezifikation abgeleiteter Wert: die
# zugrunde liegende Regel nennt keine Fensterlaenge.
CORRELATION_WINDOW_DEFAULT = 60
CLUSTER_THRESHOLD_DEFAULT = 0.75  # Paare darueber gelten als effektiv eine Wette


class CorrelationError(Exception):
    """Raised on invalid correlation input (alignment / window-vs-N)."""


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson product-moment correlation. Returns 0.0 if either series is
    flat (zero variance) — degenerate case, caller should treat as "no signal"."""
    n = len(xs)
    if n != len(ys):
        raise CorrelationError(f"xs/ys length-mismatch: {n} vs {len(ys)}")
    if n < 2:
        raise CorrelationError(f"need at least 2 obs for Pearson, got {n}")
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = 0.0
    sx2 = 0.0
    sy2 = 0.0
    for x, y in zip(xs, ys):
        dx = x - mean_x
        dy = y - mean_y
        num += dx * dy
        sx2 += dx * dx
        sy2 += dy * dy
    denom = math.sqrt(sx2 * sy2)
    if denom == 0.0:
        # Bekannte Schwaeche: 0.0 macht "nicht messbar" (flache Serie,
        # ausgesetzter Handel) ununterscheidbar von "nachweislich
        # unkorreliert" — die fail-open-Richtung in einem Cluster-Detektor:
        # der Ticker faellt unter jede Schwelle und sieht perfekt
        # diversifizierend aus, obwohl ueber ihn nichts bekannt ist.
        # Der saubere Fix (None statt 0.0, in Matrix und Cluster-Suche
        # explizit als "nicht messbar" gefuehrt) aendert Rueckgabewert und
        # Cluster-Ergebnis, also Risiko-Verhalten, und braucht eine bewusste
        # Entscheidung statt eines stillen Fixes. Bis dahin bleibt 0.0.
        return 0.0
    return num / denom


def pearson_rolling_correlation(
    returns_a: Sequence[float],
    returns_b: Sequence[float],
    *,
    window: int = CORRELATION_WINDOW_DEFAULT,
) -> list:
    """Compute the rolling Pearson correlation between two return series.

    Returns a list of length max(0, N-window+1). Series must be index-aligned
    (caller's responsibility — typically by trading-date).

    Parameters
    ----------
    returns_a, returns_b : Sequence[float]
        Daily-return series; same length, aligned by trading-day.
    window : int, optional
        Rolling window in days. Default 60 -- Implementierungswahl, KEIN
        Spezifikationswert (s. CORRELATION_WINDOW_DEFAULT oben).

    Returns
    -------
    list of float
        Rolling correlation values, one per window-aligned slice.
    """
    n = len(returns_a)
    if n != len(returns_b):
        raise CorrelationError(
            f"returns_a/returns_b length-mismatch: {n} vs {len(returns_b)}"
        )
    if window < 2:
        raise CorrelationError(f"window must be >=2, got {window}")
    if n < window:
        raise CorrelationError(
            f"need at least {window} observations for window={window}, got {n}"
        )
    out: list = []
    for i in range(window - 1, n):
        a_slice = returns_a[i - window + 1 : i + 1]
        b_slice = returns_b[i - window + 1 : i + 1]
        out.append(_pearson(a_slice, b_slice))
    return out


def pairwise_correlation_matrix(
    returns_by_ticker: Mapping[str, Sequence[float]],
    *,
    window: int = CORRELATION_WINDOW_DEFAULT,
) -> dict:
    """Compute the latest-window pairwise Pearson correlation matrix.

    Uses the LAST `window` observations per ticker (so each ticker series can
    be longer than `window`; trailing edge alignment is assumed). All tickers
    must have at least `window` observations.

    Returns nested dict {ticker_a: {ticker_b: corr}}.
    Diagonal is 1.0, matrix is symmetric.
    """
    tickers = sorted(returns_by_ticker.keys())
    series: dict = {}
    for t in tickers:
        s = returns_by_ticker[t]
        if len(s) < window:
            raise CorrelationError(
                f"ticker {t!r} has {len(s)} obs, need {window} for window={window}"
            )
        series[t] = list(s)[-window:]

    matrix: dict = {t: {} for t in tickers}
    for i, t_a in enumerate(tickers):
        matrix[t_a][t_a] = 1.0
        for t_b in tickers[i + 1 :]:
            rho = _pearson(series[t_a], series[t_b])
            matrix[t_a][t_b] = rho
            matrix[t_b][t_a] = rho
    return matrix


def find_correlation_clusters(
    matrix: Mapping[str, Mapping[str, float]],
    *,
    threshold: float = CLUSTER_THRESHOLD_DEFAULT,
) -> list:
    """Find ticker-pairs with abs(correlation) > threshold.

    Returns raw pairwise correlation edges above `threshold` only. This is
    NOT a cluster-alarm signal (several positions forming one effective
    bet) — turning these pair-edges into clusters requires graph-component
    composition, and that is NOT implemented here or anywhere else in this
    excerpt; no caller performs it. Ebenfalls nicht implementiert: die
    Sektor-plus-Korrelations-Bedingung und der Konzentrations-Alert aus den
    privaten Regeln des Autors — diese Funktion bekommt weder Sektoren noch
    eine Mindest-Clustergroesse noch Positionsgewichte. Do not treat this
    function's output as a concentration signal.

    Returns a list of (ticker_a, ticker_b, correlation) tuples, sorted
    descending by |correlation|.
    """
    tickers = sorted(matrix.keys())
    pairs: list = []
    for i, t_a in enumerate(tickers):
        for t_b in tickers[i + 1 :]:
            rho = matrix.get(t_a, {}).get(t_b)
            if rho is None:
                continue
            # Bekannte Abweichung: abs() meldet auch stark NEGATIVE
            # Korrelationen als Kante, und die Sortierung unten hebt sie
            # zusaetzlich nach oben. Die zugrunde liegende Regel ist
            # vorzeichenbehaftet (nur hohe POSITIVE Korrelation macht aus zwei
            # Positionen effektiv eine Wette);
            # ein Hedge-Paar bei rho = -0.90 SENKT die Konzentration.
            # Der Wechsel auf `rho > threshold` verschiebt eine
            # Risiko-Entscheidungsgrenze und bricht die festgeschriebene
            # abs()-Semantik (ein bestehender Test erwartet ein
            # anti-korreliertes Paar im Ergebnis). Verhalten hier bewusst
            # unveraendert.
            if abs(rho) > threshold:
                pairs.append((t_a, t_b, rho))
    pairs.sort(key=lambda r: abs(r[2]), reverse=True)
    return pairs

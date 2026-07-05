"""Historical anchor price paths for the SVB crisis window.

The simulation is anchored to what actually happened on 8-10 March 2023: daily
closes for all 10 cohort members (P0 dataset `100DaysMarketData.csv`, Yahoo
Finance provenance), interpolated to the simulator's 30-minute tick (13 bars
per trading day, 39 anchor ticks total), exactly as v1's `historicalPrices.ts`
interpolated daily OHLC to 30-minute bars.

The anchor is *not* the simulated price: it seeds valuations for the
zero-intelligence flow and the scenario controller's crisis clock. Strategic
agents trade through the order book and can push prices away from the anchor.

For runs longer than 39 ticks the anchor is extended by a seeded, mean-zero
GARCH(1,1)-like log-return process per symbol, calibrated so that each symbol's
extension volatility matches its P0 DCC-GARCH forecast scaled to tick frequency.
"""
from __future__ import annotations

import math
import random
import zlib

from .cohort import SYMBOLS, DCC_GARCH_VOLATILITIES

TICKS_PER_DAY = 13  # 09:30-16:00 in 30-minute bars

#: Daily closes, 7-10 March 2023 (P0 `100DaysMarketData.csv`). SIVB has no
#: 10 March close (halted, FDIC receivership); we carry the 106.04 last trade
#: used throughout P0/P1.
DAILY_CLOSES: dict[str, list[float]] = {
    #        Mar-7      Mar-8      Mar-9     Mar-10
    "SIVB": [275.075, 267.505, 138.87495, 106.04],
    "SBNY": [108.0, 103.64995, 95.315, 76.02],
    "JPM":  [140.06, 137.7375, 133.2825, 131.54],
    "BAC":  [33.35, 32.705, 31.35, 29.98],
    "GS":   [350.9125, 347.43, 347.67, 332.98],
    "MS":   [97.115, 95.9275, 93.86, 91.12],
    "FRC":  [118.37, 114.85, 101.05, 70.22],
    "WAL":  [73.5, 71.57, 65.7975, 43.005],
    "ABNB": [128.175, 126.2, 122.455, 118.645],
    "ETSY": [118.6648, 110.64, 107.775, 105.6025],
}

CRISIS_TICK = 14  # 09 March 10:00 (tick index within the 39-bar window), v1 CRISIS_START


def anchor_path(symbol: str, n_ticks: int, seed: int = 0) -> list[float]:
    """Anchor price for `symbol` over `n_ticks` ticks.

    Ticks 0..38 interpolate the historical daily closes geometrically
    (log-linear within each day, matching v1's piecewise interpolation intent).
    Ticks >= 39 extend with a seeded volatility-clustered log-return process.
    """
    closes = DAILY_CLOSES[symbol]
    path: list[float] = []
    for d in range(3):  # trading days 8, 9, 10 March
        p0, p1 = closes[d], closes[d + 1]
        for k in range(TICKS_PER_DAY):
            w = (k + 1) / TICKS_PER_DAY
            path.append(p0 * math.exp(w * math.log(p1 / p0)))
    if n_ticks <= len(path):
        return path[:n_ticks]

    # ── extension: GARCH(1,1)-like process, seeded ─────────────────────
    rng = random.Random((seed << 8) ^ zlib.crc32(symbol.encode()))  # stable across processes
    daily_vol = max(1e-4, DCC_GARCH_VOLATILITIES[symbol] / 10.0)  # temper P0 crisis forecast
    tick_vol = daily_vol / math.sqrt(TICKS_PER_DAY)
    omega, alpha, beta = tick_vol**2 * 0.05, 0.10, 0.85
    var = tick_vol**2
    price = path[-1]
    last_r = 0.0
    while len(path) < n_ticks:
        var = omega + alpha * last_r**2 + beta * var
        last_r = rng.gauss(0.0, math.sqrt(var))
        price *= math.exp(last_r)
        path.append(max(price, 0.01))
    return path


def all_anchor_paths(n_ticks: int, seed: int = 0) -> dict[str, list[float]]:
    return {s: anchor_path(s, n_ticks, seed) for s in SYMBOLS}

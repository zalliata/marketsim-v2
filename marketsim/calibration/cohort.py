"""The 10-asset SVB-crisis cohort and its empirical network calibration.

Data provenance (Paper P0, "Modeling Market Volatility Through Graph Theory and
Generative AI"): DCC-GARCH(1,1) volatility forecasts for 10 March 2023, the
crisis-period correlation matrix, and the LSTM-amplified contagion probability
matrix. Ported verbatim from v1 `src/data/contagionGraph.ts`.

Graph helpers (eigenvector centrality, hubs, neighbours) are used by the
graph-exploiter (G1-G3) and graph-defender (D1-D3) agents and by scenario
analytics. Centrality is normalised to sum to 1 across the cohort, matching
P0/P4 conventions.
"""
from __future__ import annotations

import math

SYMBOLS: list[str] = ["SIVB", "SBNY", "JPM", "BAC", "GS", "MS", "FRC", "WAL", "ABNB", "ETSY"]

#: DCC-GARCH(1,1) volatility forecasts for 2023-03-10 (per-day return std, decimal)
DCC_GARCH_VOLATILITIES: dict[str, float] = {
    "SIVB": 1.4521, "SBNY": 0.4750, "JPM": 0.2514, "BAC": 0.0285, "GS": 0.3806,
    "MS": 0.0632, "FRC": 0.1755, "WAL": 0.0370, "ABNB": 0.0907, "ETSY": 0.1250,
}

#: Crisis-period correlation matrix (P0 Table 1), order = SYMBOLS
CORRELATION_MATRIX: list[list[float]] = [
    [1.000000, 0.452882, -0.247280, -0.113950, -0.361720, 0.071484, 0.765112, 0.553646, 0.466366, 0.122655],
    [0.452882, 1.000000, -0.602270, 0.301132, -0.282780, -0.486100, 0.309382, 0.271419, 0.150187, -0.450880],
    [-0.247280, -0.602270, 1.000000, 0.483196, 0.819216, 0.863465, 0.100538, 0.378927, 0.079114, 0.658270],
    [-0.113950, 0.301132, 0.483196, 1.000000, 0.722436, 0.318076, 0.140483, 0.558767, 0.090097, 0.075580],
    [-0.361720, -0.282780, 0.819216, 0.722436, 1.000000, 0.682024, 0.083084, 0.402140, -0.071420, 0.545587],
    [0.071484, -0.486100, 0.863465, 0.318076, 0.682024, 1.000000, 0.361075, 0.588593, 0.365916, 0.761898],
    [0.765112, 0.309382, 0.100538, 0.140483, 0.083084, 0.361075, 1.000000, 0.592186, 0.142337, 0.541994],
    [0.553646, 0.271419, 0.378927, 0.558767, 0.402140, 0.588593, 0.592186, 1.000000, 0.636260, 0.265899],
    [0.466366, 0.150187, 0.079114, 0.090097, -0.071420, 0.365916, 0.142337, 0.636260, 1.000000, -0.025110],
    [0.122655, -0.450880, 0.658270, 0.075580, 0.545587, 0.761898, 0.541994, 0.265899, -0.025110, 1.000000],
]

#: LSTM contagion probability matrix (P0 Table 2). Values can exceed 1.0 where
#: sentiment amplification applies to SVB-linked edges (e.g. SIVB→FRC = 1.167).
CONTAGION_MATRIX: list[list[float]] = [
    [0.000000, 0.634485, 0.244209, 0.075888, 0.434649, 0.196749, 1.166975, 0.881243, 0.745106, 0.262571],
    [0.634485, 0.000000, 0.602273, 0.301132, 0.282778, 0.486104, 0.309382, 0.271419, 0.150187, 0.450877],
    [0.244209, 0.602273, 0.000000, 0.483196, 0.819216, 0.863465, 0.100538, 0.378927, 0.079114, 0.658270],
    [0.075888, 0.301132, 0.483196, 0.000000, 0.722436, 0.318076, 0.140483, 0.558767, 0.090097, 0.075580],
    [0.434649, 0.282778, 0.819216, 0.722436, 0.000000, 0.682024, 0.083084, 0.402140, 0.071417, 0.545587],
    [0.196749, 0.486104, 0.863465, 0.318076, 0.682024, 0.000000, 0.361075, 0.588593, 0.365916, 0.761898],
    [1.166975, 0.309382, 0.100538, 0.140483, 0.083084, 0.361075, 0.000000, 0.592186, 0.142337, 0.541994],
    [0.881243, 0.271419, 0.378927, 0.558767, 0.402140, 0.588593, 0.592186, 0.000000, 0.636260, 0.265899],
    [0.745106, 0.150187, 0.079114, 0.090097, 0.071417, 0.365916, 0.142337, 0.636260, 0.000000, 0.025110],
    [0.262571, 0.450877, 0.658270, 0.075580, 0.545587, 0.761898, 0.541994, 0.265899, 0.025110, 0.000000],
]

_IDX = {s: i for i, s in enumerate(SYMBOLS)}


def _eigenvector_centrality(matrix: list[list[float]], iters: int = 200) -> dict[str, float]:
    """Power-iteration eigenvector centrality on |matrix| with zero diagonal,
    normalised to sum to 1 (P0/P4 convention)."""
    n = len(SYMBOLS)
    a = [[abs(matrix[i][j]) if i != j else 0.0 for j in range(n)] for i in range(n)]
    x = [1.0 / n] * n
    for _ in range(iters):
        y = [sum(a[i][j] * x[j] for j in range(n)) for i in range(n)]
        norm = math.sqrt(sum(v * v for v in y)) or 1.0
        x = [v / norm for v in y]
    total = sum(x) or 1.0
    return {s: x[i] / total for s, i in _IDX.items()}


#: Eigenvector centrality of the contagion network (sum-normalised)
CENTRALITY: dict[str, float] = _eigenvector_centrality(CONTAGION_MATRIX)


def get_centrality(symbol: str) -> float:
    return CENTRALITY[symbol]


def get_contagion_hubs(top_n: int = 3) -> list[str]:
    """Symbols with the highest eigenvector centrality (attack/defence targets)."""
    return sorted(SYMBOLS, key=lambda s: -CENTRALITY[s])[:top_n]


def get_contagion_probability(src: str, dst: str) -> float:
    return CONTAGION_MATRIX[_IDX[src]][_IDX[dst]]


def get_top_contagion_neighbors(symbol: str, top_n: int = 3) -> list[str]:
    """Strongest outgoing contagion edges from `symbol` (cascade pathways)."""
    i = _IDX[symbol]
    others = [(SYMBOLS[j], CONTAGION_MATRIX[i][j]) for j in range(len(SYMBOLS)) if j != i]
    return [s for s, _ in sorted(others, key=lambda kv: -kv[1])[:top_n]]


def get_total_contagion_exposure(symbol: str) -> float:
    """Row sum of contagion probabilities: how much stress `symbol` transmits."""
    i = _IDX[symbol]
    return sum(CONTAGION_MATRIX[i][j] for j in range(len(SYMBOLS)) if j != i)


def get_garch_volatility(symbol: str) -> float:
    return DCC_GARCH_VOLATILITIES[symbol]

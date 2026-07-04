"""Microstructure feature extractors for the P5 detection classifier.

These operate on the per-tick `TickMetrics` log (which the engine now produces
natively) and compute the aggregate-observable features the P5 classifier is
trained on: signed order-flow imbalance, spread percentile, depth asymmetry,
cancellation rate, price impact, and cross-asset correlation spikes. All use
*only* publicly observable quantities (no agent attribution), matching RQ-F6's
"aggregate market observables alone" constraint.
"""
from __future__ import annotations

import statistics
from ..types import TickMetrics


def window_features(ticks: list[TickMetrics], symbol: str) -> dict[str, float]:
    """Feature vector for one symbol over a window of ticks."""
    xs = [t for t in ticks if t.symbol == symbol]
    if not xs:
        return {}
    spreads = [t.spread for t in xs if t.spread is not None]
    ofi = [t.signed_ofi for t in xs]
    vols = [t.volume for t in xs]
    depth_asym = [(t.bid_depth - t.ask_depth) / (t.bid_depth + t.ask_depth)
                  for t in xs if (t.bid_depth + t.ask_depth) > 0]
    cancels = [t.cancellations for t in xs]
    impacts = [t.price_impact for t in xs]
    mean = lambda a: sum(a) / len(a) if a else 0.0
    return {
        "ofi_mean": mean(ofi),
        "ofi_abs_mean": mean([abs(x) for x in ofi]),
        "ofi_std": statistics.pstdev(ofi) if len(ofi) > 1 else 0.0,
        "spread_mean": mean(spreads),
        "spread_max": max(spreads) if spreads else 0.0,
        "depth_asymmetry_mean": mean(depth_asym),
        "depth_asymmetry_absmax": max([abs(x) for x in depth_asym], default=0.0),
        "cancel_rate": mean(cancels),
        "price_impact_mean": mean(impacts),
        "volume_mean": mean(vols),
        "realized_vol_max": max((t.realized_vol for t in xs), default=0.0),
    }


def cross_asset_corr_spike(ticks: list[TickMetrics], symbols: list[str]) -> float:
    """Mean pairwise correlation of tick log-returns across symbols in the window
    (a contagion/coordination indicator). 0 if insufficient data."""
    series = {s: [t.log_return for t in ticks if t.symbol == s] for s in symbols}
    series = {s: v for s, v in series.items() if len(v) > 2}
    keys = list(series)
    if len(keys) < 2:
        return 0.0
    n = min(len(series[k]) for k in keys)
    corrs = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = series[keys[i]][:n], series[keys[j]][:n]
            try:
                corrs.append(statistics.correlation(a, b))
            except (statistics.StatisticsError, ValueError):
                continue
    return sum(corrs) / len(corrs) if corrs else 0.0

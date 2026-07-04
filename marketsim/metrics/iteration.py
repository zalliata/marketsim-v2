"""Per-iteration summary metrics (the v1 `scenario_iterations` row schema).

Given the full tick-metrics log and final accounts of one simulation run, this
produces the single-row summary used by the P2-P6 analysis scripts, plus the
systemic-risk fields (cascade onset, stabilisation effectiveness) needed by P6.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..types import TickMetrics, AgentAccount


@dataclass
class IterationSummary:
    realized_volatility: float
    max_volatility: float
    time_in_high_vol_regime: int
    final_price: float
    price_drawdown: float
    price_return: float
    adversary_pnl: float
    defense_pnl: float
    total_trades: int
    total_volume: int
    bid_ask_spread_mean: float
    sentiment_volatility: float
    min_sentiment: float
    max_sentiment: float
    # systemic-risk extensions (P6)
    cascade_onset_tick: int          # first tick RV crosses high-vol threshold, -1 if none
    cascade_frequency: float         # fraction of ticks in high-vol regime
    stabilisation_effectiveness: float  # 1 - peak_RV / baseline_peak_RV, if provided
    halts: int
    llm_mode: str = "scripted"
    adversary_share: float = 0.0


HIGH_VOL = 0.03  # realised-vol threshold for "high-vol regime" (v1 convention)


def summarize(ticks: list[TickMetrics], accounts: dict[str, AgentAccount],
              agent_types: dict[str, str], primary: str = "SIVB",
              baseline_peak_vol: float | None = None,
              llm_mode: str = "scripted", adversary_share: float = 0.0,
              final_marks: dict[str, float] | None = None) -> IterationSummary:
    prim = [t for t in ticks if t.symbol == primary]
    rv = [t.realized_vol for t in prim]
    prices = [t.price for t in prim]
    sents = [t.sentiment for t in prim]
    peak_price = max(prices) if prices else 0.0
    final_price = prices[-1] if prices else 0.0
    high_vol_ticks = [i for i, t in enumerate(prim) if t.realized_vol > HIGH_VOL]
    marks = final_marks or {}

    # PnL = mark-to-market equity - actual starting cash (robust to capital scaling)
    def pnl(pred):
        tot = 0.0
        for aid, a in accounts.items():
            if pred(agent_types.get(aid, "")):
                tot += a.equity(marks) - a.initial_cash
        return tot

    sent_vol = _std(sents)
    spreads = [t.spread for t in prim if t.spread is not None]

    return IterationSummary(
        realized_volatility=rv[-1] if rv else 0.0,
        max_volatility=max(rv) if rv else 0.0,
        time_in_high_vol_regime=len(high_vol_ticks),
        final_price=final_price,
        price_drawdown=(1 - final_price / peak_price) if peak_price > 0 else 0.0,
        price_return=(math.log(final_price / prices[0]) if prices and prices[0] > 0 else 0.0),
        adversary_pnl=pnl(lambda t: t.startswith("adversarial")),
        defense_pnl=pnl(lambda t: t.startswith("pricing")),
        total_trades=sum(t.trade_count for t in ticks),
        total_volume=sum(t.volume for t in ticks),
        bid_ask_spread_mean=(sum(spreads) / len(spreads)) if spreads else 0.0,
        sentiment_volatility=sent_vol,
        min_sentiment=min(sents) if sents else 0.0,
        max_sentiment=max(sents) if sents else 0.0,
        cascade_onset_tick=high_vol_ticks[0] if high_vol_ticks else -1,
        cascade_frequency=(len(high_vol_ticks) / len(prim)) if prim else 0.0,
        stabilisation_effectiveness=(
            1 - (max(rv) / baseline_peak_vol) if baseline_peak_vol and baseline_peak_vol > 0
            else 0.0),
        halts=sum(1 for t in prim if t.halted),
        llm_mode=llm_mode,
        adversary_share=adversary_share,
    )


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

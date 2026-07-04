"""Results-backend abstraction and the row schemas.

Schemas are additive supersets of the v1 exports: existing columns keep their
names so P2-P4 analysis scripts keep working; new columns (microstructure,
llm_mode, adversary_share, label) are appended. See docs/DATA_SCHEMA.md.
"""
from __future__ import annotations

from typing import Protocol

from ..runner.simulation import RunResult

# Column order for each table (documented in DATA_SCHEMA.md)
TIME_STEPS_COLS = [
    "run_id", "scenario_id", "iteration", "seed", "tick", "symbol", "price",
    "log_return", "realized_volatility", "sentiment", "best_bid", "best_ask",
    "spread", "bid_depth", "ask_depth", "signed_ofi", "cancellations",
    "trade_count", "volume", "price_impact", "halted", "llm_mode",
]
SCENARIO_ITERATIONS_COLS = [
    "run_id", "scenario_id", "iteration", "seed", "realized_volatility",
    "max_volatility", "time_in_high_vol_regime", "final_price", "price_drawdown",
    "price_return", "adversary_pnl", "defense_pnl", "total_trades", "total_volume",
    "bid_ask_spread_mean", "sentiment_volatility", "min_sentiment", "max_sentiment",
    "cascade_onset_tick", "cascade_frequency", "stabilisation_effectiveness",
    "halts", "llm_mode", "adversary_share", "label",
]
AGENT_SNAPSHOTS_COLS = [
    "run_id", "scenario_id", "iteration", "seed", "agent_id", "agent_type",
    "final_cash", "equity", "fees_paid", "trades_count", "positions_json",
]


class ResultsBackend(Protocol):
    def write_run(self, run_id: str, iteration: int, result: RunResult,
                  agent_types: dict[str, str], label: str = "") -> None: ...
    def close(self) -> None: ...

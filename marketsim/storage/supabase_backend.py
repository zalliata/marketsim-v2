"""Supabase backend — writes the same tables to a Postgres/Supabase project.

Credentials from SUPABASE_URL and SUPABASE_SERVICE_KEY. If the client library
is missing, credentials are absent, or a write fails, the caller falls back to
LocalCSVBackend (see storage/__init__.py::make_backend) so runs never block on
network availability. Rows are buffered and flushed in batches for throughput.
"""
from __future__ import annotations

import json
import os
from typing import Any

from .base import RunResult


class SupabaseBackend:
    def __init__(self, batch_size: int = 500):
        from supabase import create_client  # type: ignore
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_SERVICE_KEY"]
        self.client = create_client(url, key)
        self.batch_size = batch_size
        self._buf: dict[str, list[dict]] = {"time_steps": [], "scenario_iterations": [],
                                            "agent_snapshots": []}

    @staticmethod
    def available() -> bool:
        if not (os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY")):
            return False
        try:
            import supabase  # noqa: F401
            return True
        except ImportError:
            return False

    def _push(self, table: str, row: dict) -> None:
        self._buf[table].append(row)
        if len(self._buf[table]) >= self.batch_size:
            self._flush(table)

    def _flush(self, table: str) -> None:
        if self._buf[table]:
            self.client.table(table).insert(self._buf[table]).execute()
            self._buf[table] = []

    def write_run(self, run_id: str, iteration: int, result: RunResult,
                  agent_types: dict[str, str], label: str = "") -> None:
        for t in result.ticks:
            self._push("time_steps", {
                "run_id": run_id, "scenario_id": result.scenario_id, "iteration": iteration,
                "seed": result.seed, "tick": t.tick, "symbol": t.symbol, "price": t.price,
                "log_return": t.log_return, "realized_volatility": t.realized_vol,
                "sentiment": t.sentiment, "best_bid": t.best_bid, "best_ask": t.best_ask,
                "spread": t.spread, "bid_depth": t.bid_depth, "ask_depth": t.ask_depth,
                "signed_ofi": t.signed_ofi, "cancellations": t.cancellations,
                "trade_count": t.trade_count, "volume": t.volume,
                "price_impact": t.price_impact, "halted": t.halted, "llm_mode": result.llm_mode})
        s = result.summary
        self._push("scenario_iterations", {
            "run_id": run_id, "scenario_id": result.scenario_id, "iteration": iteration,
            "seed": result.seed, "realized_volatility": s.realized_volatility,
            "max_volatility": s.max_volatility, "time_in_high_vol_regime": s.time_in_high_vol_regime,
            "final_price": s.final_price, "price_drawdown": s.price_drawdown,
            "price_return": s.price_return, "adversary_pnl": s.adversary_pnl,
            "defense_pnl": s.defense_pnl, "total_trades": s.total_trades,
            "total_volume": s.total_volume, "bid_ask_spread_mean": s.bid_ask_spread_mean,
            "sentiment_volatility": s.sentiment_volatility, "min_sentiment": s.min_sentiment,
            "max_sentiment": s.max_sentiment, "cascade_onset_tick": s.cascade_onset_tick,
            "cascade_frequency": s.cascade_frequency,
            "stabilisation_effectiveness": s.stabilisation_effectiveness, "halts": s.halts,
            "llm_mode": s.llm_mode, "adversary_share": s.adversary_share, "label": label})
        for aid, fin in result.agent_final.items():
            self._push("agent_snapshots", {
                "run_id": run_id, "scenario_id": result.scenario_id, "iteration": iteration,
                "seed": result.seed, "agent_id": aid, "agent_type": agent_types.get(aid, ""),
                "final_cash": fin["cash"], "equity": fin["equity"], "fees_paid": fin["fees_paid"],
                "trades_count": fin["trades_count"], "positions_json": json.dumps(fin["positions"])})

    def close(self) -> None:
        for table in list(self._buf):
            self._flush(table)

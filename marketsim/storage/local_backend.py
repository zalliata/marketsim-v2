"""Local CSV backend — writes v1-compatible tables under results/<run_id>/.

Used as the default and as the automatic fallback when Supabase credentials
are absent or a write fails, so a run never dies for want of a database.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .base import (TIME_STEPS_COLS, SCENARIO_ITERATIONS_COLS, AGENT_SNAPSHOTS_COLS)
from ..runner.simulation import RunResult


class LocalCSVBackend:
    def __init__(self, root: str = "results"):
        self.root = Path(root)
        self._writers: dict[str, tuple] = {}

    def _writer(self, run_id: str, table: str, cols: list[str]):
        key = f"{run_id}/{table}"
        if key not in self._writers:
            d = self.root / run_id
            d.mkdir(parents=True, exist_ok=True)
            f = open(d / f"{table}.csv", "w", newline="", encoding="utf-8")
            w = csv.writer(f)
            w.writerow(cols)
            self._writers[key] = (f, w)
        return self._writers[key][1]

    def write_run(self, run_id: str, iteration: int, result: RunResult,
                  agent_types: dict[str, str], label: str = "") -> None:
        ts = self._writer(run_id, "time_steps", TIME_STEPS_COLS)
        for t in result.ticks:
            ts.writerow([run_id, result.scenario_id, iteration, result.seed, t.tick,
                         t.symbol, t.price, t.log_return, t.realized_vol, t.sentiment,
                         t.best_bid, t.best_ask, t.spread, t.bid_depth, t.ask_depth,
                         t.signed_ofi, t.cancellations, t.trade_count, t.volume,
                         t.price_impact, int(t.halted), result.llm_mode])

        it = self._writer(run_id, "scenario_iterations", SCENARIO_ITERATIONS_COLS)
        s = result.summary
        it.writerow([run_id, result.scenario_id, iteration, result.seed,
                     s.realized_volatility, s.max_volatility, s.time_in_high_vol_regime,
                     s.final_price, s.price_drawdown, s.price_return, s.adversary_pnl,
                     s.defense_pnl, s.total_trades, s.total_volume, s.bid_ask_spread_mean,
                     s.sentiment_volatility, s.min_sentiment, s.max_sentiment,
                     s.cascade_onset_tick, s.cascade_frequency,
                     s.stabilisation_effectiveness, s.halts, s.llm_mode,
                     s.adversary_share, label])

        ag = self._writer(run_id, "agent_snapshots", AGENT_SNAPSHOTS_COLS)
        for aid, fin in result.agent_final.items():
            ag.writerow([run_id, result.scenario_id, iteration, result.seed, aid,
                         agent_types.get(aid, ""), fin["cash"], fin["equity"],
                         fin["fees_paid"], fin["trades_count"],
                         json.dumps(fin["positions"])])

    def close(self) -> None:
        for f, _ in self._writers.values():
            f.close()
        self._writers.clear()

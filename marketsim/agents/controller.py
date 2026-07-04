"""Scenario controller (S1): exogenous shocks, halts, cost changes.

Unlike v1 (where controller logic was scattered across edge functions), this is
a single agent with a schedule of timed events. It never trades; it acts on the
market directly (sentiment shocks, transaction-cost changes, circuit-breaker
halts) and implements the P3 circuit-breaker trigger rule.
"""
from __future__ import annotations

from ..engine.market import Market
from ..types import Order
from .base import Agent, MarketView


class ScenarioController(Agent):
    """config:
    - sentiment_shocks: list of {tick, delta}
    - tc_changes: list of {tick, bps}
    - halts: list of {tick, n_ticks}
    - breaker: {vol_trigger, drawdown_trigger, halt_ticks, symbol} — dynamic halt
    """

    def bind(self, market: Market) -> None:
        self.market = market
        for sh in self.config.get("sentiment_shocks", []):
            market.sentiment_process.schedule_shock(int(sh["tick"]), float(sh["delta"]))
        self._peak: dict[str, float] = {}

    def decide(self, view: MarketView) -> list[Order]:
        m = self.market
        t = view.tick
        for tc in self.config.get("tc_changes", []):
            if int(tc["tick"]) == t:
                m.set_tc_bps(float(tc["bps"]))
        for h in self.config.get("halts", []):
            if int(h["tick"]) == t:
                m.halt(int(h["n_ticks"]))
        br = self.config.get("breaker")
        if br and not m.is_halted:
            sym = br.get("symbol", "SIVB")
            price = view.prices[sym]
            self._peak[sym] = max(self._peak.get(sym, price), price)
            drawdown = 1 - price / self._peak[sym] if self._peak[sym] > 0 else 0.0
            if (view.realized_vol[sym] > float(br.get("vol_trigger", 1e9))
                    or drawdown > float(br.get("drawdown_trigger", 1e9))):
                m.halt(int(br.get("halt_ticks", 13)))
        return []

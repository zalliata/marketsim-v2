"""Adversarial LLM agents (A1-A3).

This is the module that fixes v1's central defect: the batch runner drove
"adversarial_llm" agents with hard-coded heuristics and never called a model.
Here every decision flows through an `LLMClient` — either a real provider
(genuine-LLM runs) or the deterministic `ScriptedClient` (Monte Carlo
batteries) — through the *same* prompt/state/decision interface, so the two
modes are directly comparable and each output row records which produced it.

Registry variants (P1 §3):
- A1 Pure Volatility Maximiser  — objective=maximize_volatility, full info
- A2 Volatility-Profit Hybrid   — objective=volatility_profit_hybrid
- A3 Limited-Info / Comm-Constrained — sentiment/graph access off and/or
  max_messages_per_tick=1 (enforced in the view layer, not by convention)

Target selection: highest-centrality symbol if graph access is on (hub attack
premium documented in P1), else the primary symbol (SIVB).
"""
from __future__ import annotations

from ..calibration.cohort import get_contagion_hubs
from ..llm.prompts import adversarial_system_prompt
from ..types import Order, OrderType, Side
from .base import Agent, MarketView


class AdversarialLLMAgent(Agent):
    def __init__(self, *args, llm_client=None, **kwargs):
        super().__init__(*args, **kwargs)
        if llm_client is None:
            from ..llm.client import ScriptedClient
            llm_client = ScriptedClient(seed=self.rng.randrange(1 << 30))
        self.llm = llm_client
        cfg = self.config
        self.objective = str(cfg.get("objective", "maximize_volatility"))
        self.vw = float(cfg.get("volatility_weight", 1.0))
        self.pw = float(cfg.get("profit_weight", 0.0))
        self.system_prompt = adversarial_system_prompt(self.objective, self.vw, self.pw)

    def _target(self, view: MarketView) -> str:
        if view.graph_enabled:
            return get_contagion_hubs(1)[0]
        return str(self.config.get("target_symbol", "SIVB"))

    def _state(self, view: MarketView) -> dict:
        sym = self._target(view)
        book = view.books[sym]
        state: dict = {
            "tick": view.tick,
            "is_crisis": view.is_crisis,
            "target_symbol": sym,
            "price": round(view.prices[sym], 2),
            "recent_returns": [round(r, 5) for r in view.return_history[sym][-5:]],
            "realized_vol": round(view.realized_vol[sym], 5),
            "position": view.account.position(sym),
            "cash": round(view.account.cash, 2),
            "base_size": int(self.config.get("base_size", 100)),
            "volatility_weight": self.vw,
            "profit_weight": self.pw,
        }
        if self.info.sentiment:
            state["sentiment"] = round(view.sentiment, 3)
        if view.graph_enabled:
            state["network_hubs"] = get_contagion_hubs(3)
        if book.best_bid is not None:
            state["best_bid"] = book.best_bid
        if book.best_ask is not None:
            state["best_ask"] = book.best_ask
        if self.info.peer_messages and view.messages:
            state["peer_messages"] = [m["content"] for m in view.messages[-3:]]
        return state

    def decide(self, view: MarketView) -> list[Order]:
        decision = self.llm.decide(self.system_prompt, self._state(view))
        action = decision.get("action", "hold")
        if action == "hold":
            return []
        sym = decision.get("symbol") or self._target(view)
        if sym not in view.prices:
            sym = self._target(view)
        qty = max(0, int(decision.get("quantity", 0)))
        if qty == 0:
            return []
        max_pos = int(self.config.get("max_position_per_stock", 10_000))
        pos = view.account.position(sym)
        rationale = str(decision.get("rationale", ""))[:180]
        slip = float(self.config.get("max_slippage", 0.01))  # adversaries pay up
        book = view.books[sym]

        if self.info.peer_messages:
            self.post_message(f"{action} {sym} x{qty}")  # coordination channel (budgeted)

        if action == "buy":
            qty = min(qty, max(0, max_pos - pos))
            qty = self.affordable_qty(view, sym, qty)
            if qty <= 0:
                return []
            ref = book.best_ask or view.prices[sym]
            return [Order(self.agent_id, sym, Side.BUY, qty, OrderType.LIMIT,
                          round(ref * (1 + slip), 4), reason=f"[{self.llm.mode}] {rationale}")]
        qty = min(qty, max_pos + pos)  # short cap
        if qty <= 0:
            return []
        ref = book.best_bid or view.prices[sym]
        return [Order(self.agent_id, sym, Side.SELL, qty, OrderType.LIMIT,
                      round(ref * (1 - slip), 4), reason=f"[{self.llm.mode}] {rationale}")]

"""Agent framework.

Every agent implements `on_tick(view) -> list[Order]`. `MarketView` is a
read-only snapshot filtered by the agent's information access; v1's ablation
switches (`access_to_sentiment`, `access_to_graph_features`,
`access_to_peer_messages`) are enforced *by construction*: an agent without
sentiment access receives `sentiment = 0.0` and cannot cheat.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from ..engine.market import Market
from ..types import AgentAccount, BookSnapshot, Order


@dataclass
class InfoAccess:
    sentiment: bool = True
    graph_features: bool = False
    peer_messages: bool = False
    max_messages_per_tick: int = 10   # comm-constrained adversary (A3) sets 1


@dataclass
class MarketView:
    """What one agent is allowed to see at one tick."""
    tick: int
    is_crisis: bool
    halted: bool
    sentiment: float                          # 0.0 if no sentiment access
    prices: dict[str, float]
    anchors: dict[str, float]
    books: dict[str, BookSnapshot]
    realized_vol: dict[str, float]
    return_history: dict[str, list[float]]    # per-symbol log returns (copy)
    account: AgentAccount
    graph_enabled: bool
    messages: list[dict] = field(default_factory=list)  # peer messages, if permitted


class Agent:
    """Base class. Subclasses override `decide`."""

    def __init__(self, agent_id: str, name: str, agent_type: str,
                 initial_cash: float, info: Optional[InfoAccess] = None,
                 seed: int = 0, config: Optional[dict] = None):
        self.agent_id = agent_id
        self.name = name
        self.agent_type = agent_type
        self.initial_cash = initial_cash
        self.info = info or InfoAccess()
        self.rng = random.Random(seed ^ hash(agent_id) & 0xFFFFFFFF)
        self.config = config or {}
        self.outbox: list[dict] = []          # messages posted this tick

    # ── lifecycle ────────────────────────────────────────────────────────
    def build_view(self, market: Market, is_crisis: bool,
                   messages: list[dict]) -> MarketView:
        return MarketView(
            tick=market.tick,
            is_crisis=is_crisis,
            halted=market.is_halted,
            sentiment=market.sentiment if self.info.sentiment else 0.0,
            prices=dict(market.last_price),
            anchors={s: market.anchor(s) for s in market.books},
            books={s: b.snapshot() for s, b in market.books.items()},
            realized_vol={s: market.realized_vol(s) for s in market.books},
            return_history={s: list(h) for s, h in market.return_history.items()},
            account=market.accounts[self.agent_id],
            graph_enabled=self.info.graph_features,
            messages=messages if self.info.peer_messages else [],
        )

    def on_tick(self, view: MarketView) -> list[Order]:
        self.outbox = []
        return self.decide(view)

    def decide(self, view: MarketView) -> list[Order]:  # pragma: no cover
        raise NotImplementedError

    def post_message(self, content: str) -> None:
        """Broadcast to peer-message-enabled agents (budgeted per tick)."""
        if len(self.outbox) < self.info.max_messages_per_tick:
            self.outbox.append({"from": self.agent_id, "content": content})

    # ── helpers shared by subclasses ─────────────────────────────────────
    def affordable_qty(self, view: MarketView, symbol: str, qty: int) -> int:
        """Clamp a buy to available cash at the ask (or anchor) price."""
        px = view.books[symbol].best_ask or view.prices[symbol]
        if px <= 0:
            return 0
        return max(0, min(qty, int(view.account.cash / px)))

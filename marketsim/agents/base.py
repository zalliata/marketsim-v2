"""Agent framework.

Every agent implements `on_tick(view) -> list[Order]`. `MarketView` is a
read-only snapshot filtered by the agent's information access; v1's ablation
switches (`access_to_sentiment`, `access_to_graph_features`,
`access_to_peer_messages`) are enforced *by construction*: an agent without
sentiment access receives `sentiment = 0.0` and cannot cheat.
"""
from __future__ import annotations

import random
import zlib
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
    tc_bps: float = 0.0                       # per-side transaction cost (P3 sweep)


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
        # zlib.crc32, NOT hash(): str hash is salted per process, which silently
        # broke cross-session reproducibility of seeded runs.
        self.rng = random.Random(seed ^ zlib.crc32(agent_id.encode()))
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
            tc_bps=market.config.tc_bps,
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

    # ── transaction-cost awareness (P3 / RQ-F4) ───────────────────────────
    def gross_edge_bps(self, view: MarketView, symbol: str) -> float:
        """Expected per-share capture (bps) from trading `symbol` this tick.

        Proxy: current realized volatility of the symbol scaled by
        `edge_capture` (config; fraction/multiple of the per-tick vol the
        strategy expects to monetise). Calibrate `edge_capture` so that the
        implied deterrence threshold tc* is a *result*, not an assumption —
        report sensitivity to it in the paper.
        """
        capture = float(self.config.get("edge_capture", 8.0))
        return view.realized_vol.get(symbol, 0.0) * 10_000.0 * capture

    def fee_gated(self, view: MarketView, symbol: str) -> bool:
        """True when the round-trip fee exceeds the expected edge.

        Strict inequality: with tc_bps == 0 nothing is gated, so tc=0 runs
        reproduce the fee-blind baseline exactly (same seeds, same trades).
        """
        return 2.0 * view.tc_bps > self.gross_edge_bps(view, symbol)

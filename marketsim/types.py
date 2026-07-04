"""Core value types shared across the simulator.

Units convention (used everywhere in marketsim):
- prices: USD per share
- quantities: whole shares (int)
- returns: log-returns per tick
- transaction costs: basis points per side (1 bp = 0.0001 of notional)
- time: integer tick index t = 0, 1, 2, ... (one tick = one 30-minute bar by calibration)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"
    CANCEL = "cancel"  # cancels a resting order by id


@dataclass
class Order:
    """An instruction submitted by an agent to the market for one symbol."""
    agent_id: str
    symbol: str
    side: Side
    quantity: int
    type: OrderType = OrderType.LIMIT
    limit_price: Optional[float] = None      # required for LIMIT
    cancel_order_id: Optional[int] = None    # required for CANCEL
    reason: str = ""                         # free-text rationale, logged to agent_trades
    order_id: Optional[int] = None           # assigned by the book on acceptance


@dataclass
class Trade:
    """A fill produced by the matching engine."""
    tick: int
    symbol: str
    price: float
    quantity: int
    buyer_id: str
    seller_id: str
    aggressor_side: Side           # which side crossed the spread (taker)
    maker_order_id: int
    buyer_reason: str = ""
    seller_reason: str = ""


@dataclass
class BookSnapshot:
    """Top-of-book + depth aggregates for one symbol at one tick (post-matching)."""
    symbol: str
    best_bid: Optional[float]
    best_ask: Optional[float]
    bid_depth: int                 # shares within top 5 price levels, bid side
    ask_depth: int
    mid: Optional[float]
    spread: Optional[float]

    @property
    def depth_ratio(self) -> float:
        """Bid-side depth share in [0,1]; 0.5 = balanced book. NaN-safe."""
        tot = self.bid_depth + self.ask_depth
        return self.bid_depth / tot if tot > 0 else 0.5


@dataclass
class AgentAccount:
    """Cash and per-symbol positions for one agent. Managed by the market."""
    agent_id: str
    cash: float
    positions: dict[str, int] = field(default_factory=dict)
    fees_paid: float = 0.0
    trades_count: int = 0
    initial_cash: float = 0.0      # set at registration; PnL baseline

    def position(self, symbol: str) -> int:
        return self.positions.get(symbol, 0)

    def equity(self, marks: dict[str, float]) -> float:
        """Mark-to-market equity: cash + sum(position * last price)."""
        return self.cash + sum(q * marks.get(s, 0.0) for s, q in self.positions.items())


@dataclass
class TickMetrics:
    """Per-tick, per-symbol microstructure record (feeds the time_steps table)."""
    tick: int
    symbol: str
    price: float                   # last trade price, else mid, else anchor
    log_return: float
    realized_vol: float            # rolling window realised volatility of log-returns
    sentiment: float
    best_bid: Optional[float]
    best_ask: Optional[float]
    spread: Optional[float]
    bid_depth: int
    ask_depth: int
    signed_ofi: int                # buy-initiated volume minus sell-initiated volume this tick
    cancellations: int
    trade_count: int
    volume: int
    price_impact: float            # abs(delta mid) per unit executed volume (0 if no volume)
    halted: bool = False

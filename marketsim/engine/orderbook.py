"""Continuous double auction limit order book for one symbol.

Price-time priority; supports limit orders, market orders, cancels, partial fills.
This replaces v1's price-curve-with-additive-impact model so that order-flow
imbalance, depth, spreads, and cancellation rates are real, measurable objects
(required by Papers 2, 3, 5, 6).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..types import Order, OrderType, Side, Trade, BookSnapshot


@dataclass
class RestingOrder:
    order_id: int
    agent_id: str
    side: Side
    price: float
    quantity: int          # remaining
    entry_seq: int         # time priority
    reason: str = ""


@dataclass
class OrderBook:
    symbol: str
    _bids: list[RestingOrder] = field(default_factory=list)   # sorted desc by (price, -entry_seq)
    _asks: list[RestingOrder] = field(default_factory=list)   # sorted asc by (price, entry_seq)
    _next_id: int = 1
    _seq: int = 0
    cancellations_this_tick: int = 0

    # ── queries ─────────────────────────────────────────────────────────
    def best_bid(self) -> Optional[float]:
        return self._bids[0].price if self._bids else None

    def best_ask(self) -> Optional[float]:
        return self._asks[0].price if self._asks else None

    def mid(self) -> Optional[float]:
        b, a = self.best_bid(), self.best_ask()
        if b is not None and a is not None:
            return (b + a) / 2.0
        return b if b is not None else a

    def depth(self, side: Side, levels: int = 5) -> int:
        book = self._bids if side == Side.BUY else self._asks
        prices: list[float] = []
        total = 0
        for o in book:
            if o.price not in prices:
                if len(prices) == levels:
                    break
                prices.append(o.price)
            total += o.quantity
        return total

    def snapshot(self) -> BookSnapshot:
        b, a = self.best_bid(), self.best_ask()
        return BookSnapshot(
            symbol=self.symbol, best_bid=b, best_ask=a,
            bid_depth=self.depth(Side.BUY), ask_depth=self.depth(Side.SELL),
            mid=self.mid(), spread=(a - b) if (a is not None and b is not None) else None,
        )

    def orders_of(self, agent_id: str) -> list[RestingOrder]:
        return [o for o in self._bids + self._asks if o.agent_id == agent_id]

    # ── mutation ────────────────────────────────────────────────────────
    def cancel(self, order_id: int) -> bool:
        for book in (self._bids, self._asks):
            for i, o in enumerate(book):
                if o.order_id == order_id:
                    book.pop(i)
                    self.cancellations_this_tick += 1
                    return True
        return False

    def cancel_all(self, agent_id: str) -> int:
        n = 0
        for book in (self._bids, self._asks):
            keep = []
            for o in book:
                if o.agent_id == agent_id:
                    n += 1
                else:
                    keep.append(o)
            book[:] = keep
        self.cancellations_this_tick += n
        return n

    def submit(self, order: Order, tick: int) -> list[Trade]:
        """Match an incoming order; rest any unfilled limit remainder.

        Market orders fill against available depth and any residual is dropped
        (no resting market orders). Self-trades are skipped (order behind own quote).
        """
        if order.type == OrderType.CANCEL:
            if order.cancel_order_id is not None:
                self.cancel(order.cancel_order_id)
            return []

        trades: list[Trade] = []
        qty = int(order.quantity)
        if qty <= 0:
            return []

        opposite = self._asks if order.side == Side.BUY else self._bids

        def crosses(px_resting: float) -> bool:
            if order.type == OrderType.MARKET:
                return True
            assert order.limit_price is not None
            return (px_resting <= order.limit_price) if order.side == Side.BUY \
                else (px_resting >= order.limit_price)

        i = 0
        while qty > 0 and i < len(opposite):
            resting = opposite[i]
            if not crosses(resting.price):
                break
            if resting.agent_id == order.agent_id:   # never self-trade
                i += 1
                continue
            fill = min(qty, resting.quantity)
            buyer = order.agent_id if order.side == Side.BUY else resting.agent_id
            seller = resting.agent_id if order.side == Side.BUY else order.agent_id
            trades.append(Trade(
                tick=tick, symbol=self.symbol, price=resting.price, quantity=fill,
                buyer_id=buyer, seller_id=seller, aggressor_side=order.side,
                maker_order_id=resting.order_id,
                buyer_reason=order.reason if order.side == Side.BUY else resting.reason,
                seller_reason=order.reason if order.side == Side.SELL else resting.reason,
            ))
            qty -= fill
            resting.quantity -= fill
            if resting.quantity == 0:
                opposite.pop(i)
            # do not advance i on removal; next order shifts into slot i

        if qty > 0 and order.type == OrderType.LIMIT and order.limit_price is not None:
            self._seq += 1
            ro = RestingOrder(self._next_id, order.agent_id, order.side,
                              float(order.limit_price), qty, self._seq, order.reason)
            order.order_id = self._next_id
            self._next_id += 1
            book = self._bids if order.side == Side.BUY else self._asks
            book.append(ro)
            book.sort(key=(lambda o: (-o.price, o.entry_seq)) if order.side == Side.BUY
                      else (lambda o: (o.price, o.entry_seq)))
        return trades

    def reset_tick_counters(self) -> None:
        self.cancellations_this_tick = 0

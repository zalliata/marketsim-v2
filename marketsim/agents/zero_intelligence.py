"""Zero-intelligence traders (Z1 uniform, Z2 side-biased).

Gode & Sunder (1993)-style budget-constrained random traders. They provide the
baseline liquidity and price-discovery flow. Unlike v1 (random buy/sell against
a price curve), they post *limit orders around the anchor value*, which is what
keeps the book two-sided and prices tethered to the historical crisis path
without dictating them.

Config fields (v1-compatible names):
- side_bias: probability an active order is a buy (0.5 = Z1, e.g. 0.4 = Z2 bearish)
- order_size_distribution: {min, max} shares, uniform
- trade_probability: chance of acting on a symbol each tick
- quote_offset_std: lognormal-ish dispersion of limit prices around anchor (fraction)
"""
from __future__ import annotations

from ..types import Order, OrderType, Side
from .base import Agent, MarketView


class ZeroIntelligenceAgent(Agent):
    def decide(self, view: MarketView) -> list[Order]:
        cfg = self.config
        side_bias = float(cfg.get("side_bias", 0.5))
        omin = int(cfg.get("order_size_distribution", {}).get("min", 10))
        omax = int(cfg.get("order_size_distribution", {}).get("max", 100))
        p_act = float(cfg.get("trade_probability", 0.6))
        offset_std = float(cfg.get("quote_offset_std", 0.004))
        max_pos = int(cfg.get("max_position_per_stock", 1000))

        # sentiment tilts the side bias during crisis (panic selling), as in v1
        bias = side_bias + 0.2 * view.sentiment
        orders: list[Order] = []
        for symbol in view.prices:
            if self.rng.random() > p_act:
                continue
            anchor = view.anchors[symbol]
            qty = self.rng.randint(omin, omax)
            buy = self.rng.random() < bias
            pos = view.account.position(symbol)
            if buy and pos + qty > max_pos:
                continue
            if not buy and pos - qty < -max_pos:
                continue
            offset = self.rng.gauss(0.0, offset_std)
            # buyers shade below anchor, sellers above → resting liquidity;
            # occasional aggressive orders cross the spread (price discovery)
            aggressive = self.rng.random() < 0.25
            if buy:
                px = anchor * (1 + (offset if aggressive else -abs(offset)))
                qty = self.affordable_qty(view, symbol, qty)
                if qty <= 0:
                    continue
                orders.append(Order(self.agent_id, symbol, Side.BUY, qty,
                                    OrderType.LIMIT, round(px, 4), reason="ZI buy"))
            else:
                px = anchor * (1 + (offset if aggressive else abs(offset)))
                orders.append(Order(self.agent_id, symbol, Side.SELL, qty,
                                    OrderType.LIMIT, round(px, 4), reason="ZI sell"))
        return orders

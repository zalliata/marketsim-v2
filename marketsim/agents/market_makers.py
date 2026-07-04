"""Rule-based market makers.

P1 — FixedSpreadMM: symmetric quotes around the reference price at a constant
half-spread; inventory capped. The v1 'Fixed-Spread Baseline MM'.

P2 — VolInventoryMM: half-spread widens with realised volatility and quote
centre skews against inventory — a discretised Avellaneda-Stoikov (2008)
policy, which is what v1's 'Vol-Inventory Aware MM' approximated.

P4 — SignalAwareMM: renamed from v1's 'LLM-Signal-Aware RL Defense' (which
neither used an LLM nor RL in batch mode). Widens spreads when manipulation
indicators fire: sentiment shock, order-flow imbalance spike, cross-asset
depth asymmetry. Decisions are rule-based and documented; optional LLM
commentary can be attached via the llm module but never drives quotes.

Common quoting mechanics: each tick the MM cancels its stale quotes (counted
in the cancellation-rate metric, as in real markets) and reposts two-sided
limit orders of `quote_size` shares.
"""
from __future__ import annotations

from ..types import Order, OrderType, Side
from .base import Agent, MarketView


class _QuotingMM(Agent):
    def half_spread(self, view: MarketView, symbol: str) -> float:
        raise NotImplementedError

    def center(self, view: MarketView, symbol: str) -> float:
        return view.prices[symbol]

    def decide(self, view: MarketView) -> list[Order]:
        cfg = self.config
        size = int(cfg.get("quote_size", 50))
        max_inv = int(cfg.get("max_inventory", 2000))
        orders: list[Order] = []
        for symbol in view.prices:
            c = self.center(view, symbol)
            h = self.half_spread(view, symbol)
            inv = view.account.position(symbol)
            bid_px = round(c - h, 4)
            ask_px = round(c + h, 4)
            # inventory guard: stop adding on the side that grows exposure
            if inv < max_inv:
                bqty = self.affordable_qty(view, symbol, size)
                if bqty > 0:
                    orders.append(Order(self.agent_id, symbol, Side.BUY, bqty,
                                        OrderType.LIMIT, bid_px, reason=f"{self.agent_type} bid"))
            if inv > -max_inv:
                orders.append(Order(self.agent_id, symbol, Side.SELL, size,
                                    OrderType.LIMIT, ask_px, reason=f"{self.agent_type} ask"))
        return orders


class FixedSpreadMM(_QuotingMM):
    """P1. half_spread = spread_bps/2 of price (config spread_bps, default 20)."""

    def half_spread(self, view: MarketView, symbol: str) -> float:
        bps = float(self.config.get("spread_bps", 20.0))
        return view.prices[symbol] * bps / 10_000.0 / 2.0


class VolInventoryMM(_QuotingMM):
    """P2. Spread scales with realised vol; centre skews against inventory."""

    def half_spread(self, view: MarketView, symbol: str) -> float:
        base_bps = float(self.config.get("base_spread_bps", 15.0))
        vol_mult = float(self.config.get("vol_multiplier", 40.0))
        rv = view.realized_vol[symbol]
        return view.prices[symbol] * (base_bps / 10_000.0 + vol_mult * rv * 0.01) / 2.0

    def center(self, view: MarketView, symbol: str) -> float:
        inv = view.account.position(symbol)
        max_inv = int(self.config.get("max_inventory", 2000))
        skew = float(self.config.get("inventory_skew", 0.001))
        # long inventory → shade quotes down to offload; short → up
        return view.prices[symbol] * (1 - skew * (inv / max(max_inv, 1)))


class SignalAwareMM(VolInventoryMM):
    """P4. VolInventoryMM plus manipulation-indicator spread widening.

    Indicators (all from public observables — no privileged data):
    - sentiment shock: |sentiment| > sentiment_shock_level (default 0.5)
    - OFI spike: |signed OFI| of the last tick > ofi_spike shares (default 300)
    - depth asymmetry: bid/ask depth ratio outside [0.25, 0.75]
    Each active indicator multiplies the half-spread by defense_multiplier.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_ofi: dict[str, int] = {}

    def observe_tick_metrics(self, per_symbol_ofi: dict[str, int]) -> None:
        self._last_ofi = per_symbol_ofi

    def half_spread(self, view: MarketView, symbol: str) -> float:
        h = super().half_spread(view, symbol)
        cfg = self.config
        mult = float(cfg.get("defense_multiplier", 1.6))
        active = 0
        if abs(view.sentiment) > float(cfg.get("sentiment_shock_level", 0.5)):
            active += 1
        if abs(self._last_ofi.get(symbol, 0)) > int(cfg.get("ofi_spike", 300)):
            active += 1
        book = view.books[symbol]
        if not (0.25 <= book.depth_ratio <= 0.75):
            active += 1
        return h * (mult ** active)

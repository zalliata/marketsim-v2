"""Multi-asset market: tick loop, fee charging, halts, accounts, metrics.

One `Market` owns one `OrderBook` per cohort symbol, all agent accounts, the
sentiment process, and the anchor paths. `step(orders)` runs one tick:

1. if halted, only cancels are processed;
2. orders are matched in a seeded-random agent order (fairness across agents);
3. fills settle cash/positions and charge transaction costs (bps per side);
4. per-symbol `TickMetrics` are computed (microstructure logging for P2/P3/P5/P6).

Transaction cost (`tc_bps`) is a market parameter — the sweep variable of
Paper P3. Circuit-breaker halts are controlled by the scenario controller
through `halt(n_ticks)`.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from ..calibration.anchors import all_anchor_paths
from ..calibration.cohort import SYMBOLS
from ..calibration.sentiment import SentimentProcess
from ..types import AgentAccount, Order, OrderType, Side, TickMetrics, Trade
from .orderbook import OrderBook

RV_WINDOW = 20  # ticks used for realised volatility (v1 convention)


@dataclass
class MarketConfig:
    n_ticks: int = 39
    tc_bps: float = 10.0            # transaction cost per side, basis points
    seed: int = 0
    crisis_enabled: bool = True
    symbols: list[str] = field(default_factory=lambda: list(SYMBOLS))


class Market:
    def __init__(self, config: MarketConfig):
        self.config = config
        self.tick = 0
        self.rng = random.Random(config.seed)
        self.books: dict[str, OrderBook] = {s: OrderBook(s) for s in config.symbols}
        self.accounts: dict[str, AgentAccount] = {}
        self.anchors = all_anchor_paths(config.n_ticks, config.seed)
        self.sentiment_process = SentimentProcess(seed=config.seed,
                                                  crisis_enabled=config.crisis_enabled)
        self.sentiment: float = 0.0
        self.last_price: dict[str, float] = {s: self.anchors[s][0] for s in config.symbols}
        self._prev_price: dict[str, float] = dict(self.last_price)
        self.return_history: dict[str, list[float]] = {s: [] for s in config.symbols}
        self.trades: list[Trade] = []
        self.halt_remaining = 0
        self.halt_count = 0

    # ── registration ────────────────────────────────────────────────────
    def register(self, agent_id: str, cash: float) -> AgentAccount:
        acct = AgentAccount(agent_id=agent_id, cash=cash, initial_cash=cash)
        self.accounts[agent_id] = acct
        return acct

    # ── controls (scenario controller) ──────────────────────────────────
    def halt(self, n_ticks: int) -> None:
        """Trigger a circuit-breaker halt for `n_ticks` ticks (P3 arm B)."""
        self.halt_remaining = max(self.halt_remaining, n_ticks)
        self.halt_count += 1

    def set_tc_bps(self, bps: float) -> None:
        self.config.tc_bps = bps

    # ── views ────────────────────────────────────────────────────────────
    def anchor(self, symbol: str) -> float:
        return self.anchors[symbol][min(self.tick, len(self.anchors[symbol]) - 1)]

    def price(self, symbol: str) -> float:
        return self.last_price[symbol]

    def realized_vol(self, symbol: str) -> float:
        h = self.return_history[symbol][-RV_WINDOW:]
        if len(h) < 2:
            return 0.0
        m = sum(h) / len(h)
        return math.sqrt(sum((r - m) ** 2 for r in h) / len(h))

    @property
    def is_halted(self) -> bool:
        return self.halt_remaining > 0

    # ── tick ─────────────────────────────────────────────────────────────
    def step(self, orders_by_agent: dict[str, list[Order]]) -> list[TickMetrics]:
        for book in self.books.values():
            book.reset_tick_counters()

        halted = self.is_halted
        pre_mid = {s: (self.books[s].mid() or self.last_price[s]) for s in self.books}
        ofi: dict[str, int] = {s: 0 for s in self.books}
        volume: dict[str, int] = {s: 0 for s in self.books}
        trade_count: dict[str, int] = {s: 0 for s in self.books}

        agent_ids = list(orders_by_agent.keys())
        self.rng.shuffle(agent_ids)
        for aid in agent_ids:
            for order in orders_by_agent[aid]:
                if order.symbol not in self.books:
                    continue
                if halted and order.type != OrderType.CANCEL:
                    continue  # halt: matching suspended, cancels allowed
                fills = self.books[order.symbol].submit(order, self.tick)
                for tr in fills:
                    self._settle(tr)
                    sgn = tr.quantity if tr.aggressor_side == Side.BUY else -tr.quantity
                    ofi[tr.symbol] += sgn
                    volume[tr.symbol] += tr.quantity
                    trade_count[tr.symbol] += 1
                    self.last_price[tr.symbol] = tr.price

        self.sentiment = self.sentiment_process.step(self.tick)

        metrics: list[TickMetrics] = []
        for s, book in self.books.items():
            snap = book.snapshot()
            # tick price: last trade if any this tick, else mid, else carry
            price = self.last_price[s] if volume[s] > 0 else (snap.mid or self.last_price[s])
            prev = self._prev_price[s]
            log_ret = math.log(price / prev) if prev > 0 and price > 0 else 0.0
            self.return_history[s].append(log_ret)
            mid_now = snap.mid or price
            impact = (abs(mid_now - pre_mid[s]) / volume[s]) if volume[s] > 0 else 0.0
            metrics.append(TickMetrics(
                tick=self.tick, symbol=s, price=price, log_return=log_ret,
                realized_vol=self.realized_vol(s), sentiment=self.sentiment,
                best_bid=snap.best_bid, best_ask=snap.best_ask, spread=snap.spread,
                bid_depth=snap.bid_depth, ask_depth=snap.ask_depth,
                signed_ofi=ofi[s], cancellations=book.cancellations_this_tick,
                trade_count=trade_count[s], volume=volume[s],
                price_impact=impact, halted=halted,
            ))
            self.last_price[s] = price
            self._prev_price[s] = price

        if self.halt_remaining > 0:
            self.halt_remaining -= 1
        self.tick += 1
        return metrics

    # ── settlement ────────────────────────────────────────────────────────
    def _settle(self, tr: Trade) -> None:
        fee = tr.price * tr.quantity * self.config.tc_bps / 10_000.0
        buyer, seller = self.accounts.get(tr.buyer_id), self.accounts.get(tr.seller_id)
        if buyer:
            buyer.cash -= tr.price * tr.quantity + fee
            buyer.fees_paid += fee
            buyer.positions[tr.symbol] = buyer.position(tr.symbol) + tr.quantity
            buyer.trades_count += 1
        if seller:
            seller.cash += tr.price * tr.quantity - fee
            seller.fees_paid += fee
            seller.positions[tr.symbol] = seller.position(tr.symbol) - tr.quantity
            seller.trades_count += 1
        self.trades.append(tr)

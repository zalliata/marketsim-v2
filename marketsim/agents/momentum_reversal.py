"""Momentum (MOM) and Short-Term Reversal (STREV) agents.

Signal math ported verbatim from v1 `src/lib/simulation/signals.ts`
(the one part of v1 with tests):

- momentum:  mom_t = Σ log-returns over [t−L−skip, t−skip)   (L=12, skip=1)
- reversal:  strev_t = −Σ log-returns over last L steps      (L=1)
- z-score normalisation over a rolling window of past signal values (20)
- target position = clamp(z * position_scale, ±max_inventory)
- order = target − current (skip if |Δ| < 1 share)

Execution differs from v1: orders go to the book as marketable limit orders
(cross the spread up to a slippage cap) instead of trading against a curve.

Economic provenance: Jegadeesh & Titman (1993) momentum; Jegadeesh (1990)
short-term reversal. The 'Hub-Focused Reversal' variant (v1 M-agents) trades
only the top-centrality symbols when graph access is enabled.
"""
from __future__ import annotations

from ..calibration.cohort import get_contagion_hubs
from ..types import Order, OrderType, Side
from .base import Agent, MarketView


def momentum_signal(returns: list[float], lookback: int = 12, skip: int = 1) -> float:
    req = lookback + skip
    if len(returns) < req:
        return 0.0
    return sum(returns[len(returns) - req: len(returns) - skip])


def reversal_signal(returns: list[float], lookback: int = 1) -> float:
    if len(returns) < lookback:
        return 0.0
    return -sum(returns[-lookback:])


def z_score(current: float, history: list[float], window: int = 20) -> float:
    w = history[-window:]
    if len(w) < 3:
        return current
    mean = sum(w) / len(w)
    var = sum((v - mean) ** 2 for v in w) / len(w)
    std = var ** 0.5
    if std < 1e-10:
        return 0.0
    return (current - mean) / std


class SignalAgent(Agent):
    """Shared machinery for MOM and STREV. config['signal'] ∈ {momentum, reversal}."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._signal_history: dict[str, list[float]] = {}

    def _raw_signal(self, returns: list[float]) -> float:
        cfg = self.config
        if cfg.get("signal", "momentum") == "momentum":
            return momentum_signal(returns, int(cfg.get("lookback", 12)),
                                   int(cfg.get("skip_recent", 1)))
        return reversal_signal(returns, int(cfg.get("lookback", 1)))

    def decide(self, view: MarketView) -> list[Order]:
        cfg = self.config
        scale = float(cfg.get("position_scale", 300))
        max_inv = int(cfg.get("max_inventory", 1000))
        slippage = float(cfg.get("max_slippage", 0.004))
        hub_only = bool(cfg.get("hub_focused", False)) and view.graph_enabled
        symbols = get_contagion_hubs(3) if hub_only else list(view.prices)

        orders: list[Order] = []
        for symbol in symbols:
            raw = self._raw_signal(view.return_history[symbol])
            hist = self._signal_history.setdefault(symbol, [])
            z = z_score(raw, hist, int(cfg.get("z_window", 20)))
            hist.append(raw)

            target = max(-max_inv, min(max_inv, round(z * scale)))
            current = view.account.position(symbol)
            delta = target - current
            if abs(delta) < 1:
                continue
            side = Side.BUY if delta > 0 else Side.SELL
            qty = abs(int(delta))
            if side == Side.BUY:
                qty = self.affordable_qty(view, symbol, qty)
                if qty <= 0:
                    continue
            book = view.books[symbol]
            ref = (book.best_ask if side == Side.BUY else book.best_bid) or view.prices[symbol]
            px = ref * (1 + slippage) if side == Side.BUY else ref * (1 - slippage)
            orders.append(Order(self.agent_id, symbol, side, qty, OrderType.LIMIT,
                                round(px, 4),
                                reason=f"{cfg.get('signal', 'momentum')} z={z:.2f} target={target}"))
        return orders

"""Graph-aware agents: exploiters (G1-G3) and defenders (D1-D3).

These operate on the contagion network from `calibration.cohort`. Exploiters
attack systemically central nodes to maximise cascade propagation; defenders
provide targeted liquidity at those same nodes. Ported from v1's graph-agent
decision branches, with order-book execution and explicit use of the centrality
/ contagion helpers.

Exploiters (adversarial, graph access required):
- G1 HubAttacker        — sells the top-centrality hub to trigger contagion
- G2 PathwayExploiter    — attacks the strongest outgoing-contagion neighbour
                           of a distressed hub (cascade pathway)
- G3 CrossSectorAmplifier— trades the peripheral (low-centrality) nodes to
                           spread stress across sectors

Defenders (pricing, graph access required):
- D1 CentralityLiquidityDefender — tight two-sided quotes at top-centrality hubs
- D2 ContagionFirewallDefender   — absorbs sell pressure on high-exposure edges
- D3 AdaptiveGraphRebalancer     — rebalances liquidity toward whichever hub is
                                    currently most stressed (highest realised vol)
"""
from __future__ import annotations

from ..calibration.cohort import (get_contagion_hubs, get_top_contagion_neighbors,
                                   get_total_contagion_exposure, SYMBOLS, get_centrality)
from ..types import Order, OrderType, Side
from .base import Agent, MarketView


# ── exploiters ──────────────────────────────────────────────────────────
class HubAttacker(Agent):
    def decide(self, view: MarketView) -> list[Order]:
        if not view.graph_enabled:
            return []
        hub = get_contagion_hubs(1)[0]
        size = int(self.config.get("attack_size", 200))
        # attack hardest when crisis + negative sentiment (max cascade leverage)
        if not (view.is_crisis or view.sentiment < -0.2):
            return []
        book = view.books[hub]
        ref = book.best_bid or view.prices[hub]
        slip = float(self.config.get("max_slippage", 0.01))
        return [Order(self.agent_id, hub, Side.SELL, size, OrderType.LIMIT,
                      round(ref * (1 - slip), 4),
                      reason=f"G1 hub attack on {hub} (centrality {get_centrality(hub):.3f})")]


class PathwayExploiter(Agent):
    def decide(self, view: MarketView) -> list[Order]:
        if not view.graph_enabled:
            return []
        hub = get_contagion_hubs(1)[0]
        targets = get_top_contagion_neighbors(hub, int(self.config.get("n_pathways", 2)))
        size = int(self.config.get("attack_size", 120))
        slip = float(self.config.get("max_slippage", 0.01))
        orders = []
        for sym in targets:
            if view.sentiment >= 0 and not view.is_crisis:
                continue
            ref = view.books[sym].best_bid or view.prices[sym]
            orders.append(Order(self.agent_id, sym, Side.SELL, size, OrderType.LIMIT,
                                round(ref * (1 - slip), 4),
                                reason=f"G2 pathway {hub}->{sym}"))
        return orders


class CrossSectorAmplifier(Agent):
    def decide(self, view: MarketView) -> list[Order]:
        if not view.graph_enabled:
            return []
        periphery = sorted(SYMBOLS, key=get_centrality)[:int(self.config.get("n_periphery", 3))]
        size = int(self.config.get("attack_size", 100))
        slip = float(self.config.get("max_slippage", 0.01))
        orders = []
        for sym in periphery:
            if self.rng.random() > 0.5:
                continue
            side = Side.SELL if view.sentiment < 0 else Side.BUY
            ref = ((view.books[sym].best_bid if side == Side.SELL else view.books[sym].best_ask)
                   or view.prices[sym])
            px = ref * (1 - slip) if side == Side.SELL else ref * (1 + slip)
            if side == Side.BUY:
                size2 = self.affordable_qty(view, sym, size)
                if size2 <= 0:
                    continue
                size = size2
            orders.append(Order(self.agent_id, sym, side, size, OrderType.LIMIT,
                                round(px, 4), reason=f"G3 cross-sector {sym}"))
        return orders


# ── defenders ─────────────────────────────────────────────────────────────
class _GraphDefender(Agent):
    def _quote(self, view: MarketView, symbol: str, half_bps: float, size: int) -> list[Order]:
        c = view.prices[symbol]
        h = c * half_bps / 10_000.0
        out = []
        bqty = self.affordable_qty(view, symbol, size)
        if bqty > 0:
            out.append(Order(self.agent_id, symbol, Side.BUY, bqty, OrderType.LIMIT,
                             round(c - h, 4), reason=f"{self.agent_type} bid {symbol}"))
        out.append(Order(self.agent_id, symbol, Side.SELL, size, OrderType.LIMIT,
                         round(c + h, 4), reason=f"{self.agent_type} ask {symbol}"))
        return out


class CentralityLiquidityDefender(_GraphDefender):
    def decide(self, view: MarketView) -> list[Order]:
        if not view.graph_enabled:
            return []
        hubs = get_contagion_hubs(int(self.config.get("n_hubs", 3)))
        half = float(self.config.get("half_spread_bps", 10.0))
        size = int(self.config.get("quote_size", 80))
        orders = []
        for sym in hubs:
            orders += self._quote(view, sym, half, size)
        return orders


class ContagionFirewallDefender(_GraphDefender):
    def decide(self, view: MarketView) -> list[Order]:
        if not view.graph_enabled:
            return []
        # defend the highest total-exposure transmitters
        targets = sorted(SYMBOLS, key=get_total_contagion_exposure, reverse=True)
        targets = targets[:int(self.config.get("n_targets", 2))]
        size = int(self.config.get("quote_size", 120))
        orders = []
        for sym in targets:
            # bias toward absorbing sell pressure during stress: post a firm bid
            c = view.prices[sym]
            bqty = self.affordable_qty(view, sym, size)
            if bqty > 0:
                orders.append(Order(self.agent_id, sym, Side.BUY, bqty, OrderType.LIMIT,
                                    round(c * (1 - float(self.config.get("bid_bps", 8)) / 10_000), 4),
                                    reason=f"D2 firewall bid {sym}"))
        return orders


class AdaptiveGraphRebalancer(_GraphDefender):
    def decide(self, view: MarketView) -> list[Order]:
        if not view.graph_enabled:
            return []
        hubs = get_contagion_hubs(3)
        # concentrate liquidity at the currently most-stressed hub
        target = max(hubs, key=lambda s: view.realized_vol[s])
        half = float(self.config.get("half_spread_bps", 8.0))
        size = int(self.config.get("quote_size", 150))
        return self._quote(view, target, half, size)

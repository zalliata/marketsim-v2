"""Tabular Q-learning market maker (P3 in the agent registry).

Faithful port of v1 `src/lib/qlearning.ts`: identical state discretisation
(7 price-change × 5 sentiment × 3 position × 3 volatility buckets), the same
default hyperparameters (α=0.1, γ=0.95, ε=0.3 decaying ×0.995 to 0.05), and
epsilon-greedy action selection over {buy, sell, hold}. All randomness comes
from the agent's seeded generator (determinism guarantee).

Differences vs v1, documented for the papers:
- reward = Δ mark-to-market equity per tick (v1 used per-trade PnL proxies),
  with optional reward shaping presets from the p4-rl-reward-* scenarios:
  'standard' (pure PnL), 'aggressive' (PnL − 0.5·|inventory| penalty weight),
  'social' (PnL − vol_penalty·realised vol) — the social planner variant.
- Q-tables can be saved/loaded as JSON → pretraining and transfer scenarios
  (p4-pretrain-rl, p4-transfer-test) are first-class.
- trades execute through the order book (marketable limit, slippage-capped).
"""
from __future__ import annotations

import json
from pathlib import Path

from ..types import Order, OrderType, Side
from .base import Agent, MarketView

ACTIONS = ("buy", "sell", "hold")


def discretize(price_change_pct: float, sentiment: float,
               position_ratio: float, volatility: float) -> str:
    """v1 bucket boundaries, verbatim."""
    if price_change_pct < -5: p = "crash"
    elif price_change_pct < -2: p = "down"
    elif price_change_pct < -0.5: p = "slight_down"
    elif price_change_pct < 0.5: p = "flat"
    elif price_change_pct < 2: p = "slight_up"
    elif price_change_pct < 5: p = "up"
    else: p = "surge"
    if sentiment < -0.6: s = "very_bearish"
    elif sentiment < -0.2: s = "bearish"
    elif sentiment < 0.2: s = "neutral"
    elif sentiment < 0.6: s = "bullish"
    else: s = "very_bullish"
    if position_ratio < -0.1: pos = "short"
    elif position_ratio > 0.1: pos = "long"
    else: pos = "neutral"
    if volatility < 0.01: v = "low"
    elif volatility < 0.03: v = "medium"
    else: v = "high"
    return f"{p}|{s}|{pos}|{v}"


class QLearningMM(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.config
        self.alpha = float(cfg.get("learning_rate", 0.1))
        self.gamma = float(cfg.get("discount_factor", 0.95))
        self.epsilon = float(cfg.get("exploration_rate", 0.3))
        self.eps_decay = float(cfg.get("exploration_decay", 0.995))
        self.eps_min = float(cfg.get("min_exploration", 0.05))
        self.reward_mode = str(cfg.get("reward_mode", "standard"))
        self.training = bool(cfg.get("training", True))
        self.q: dict[str, dict[str, float]] = {}
        qpath = cfg.get("q_table_path")
        if qpath and Path(qpath).exists():
            self.q = json.loads(Path(qpath).read_text())
        self.symbol = str(cfg.get("symbol", "SIVB"))   # v1 RL agent is single-symbol
        self._last: tuple[str, str] | None = None       # (state, action)
        self._last_equity: float | None = None

    # ── Q machinery ─────────────────────────────────────────────────────
    def _qrow(self, state: str) -> dict[str, float]:
        return self.q.setdefault(state, {a: 0.0 for a in ACTIONS})

    def _choose(self, state: str) -> str:
        if self.training and self.rng.random() < self.epsilon:
            return self.rng.choice(ACTIONS)
        row = self._qrow(state)
        return max(ACTIONS, key=lambda a: row[a])

    def _reward(self, view: MarketView) -> float:
        equity = view.account.equity(view.prices)
        if self._last_equity is None:
            self._last_equity = equity
            return 0.0
        pnl = equity - self._last_equity
        self._last_equity = equity
        if self.reward_mode == "aggressive":
            return pnl - 0.5 * abs(view.account.position(self.symbol)) * 0.01
        if self.reward_mode == "social":
            vol_penalty = float(self.config.get("vol_penalty", 5000.0))
            return pnl - vol_penalty * view.realized_vol[self.symbol]
        return pnl

    def _update(self, reward: float, new_state: str) -> None:
        if self._last is None or not self.training:
            return
        state, action = self._last
        row = self._qrow(state)
        best_next = max(self._qrow(new_state).values())
        row[action] += self.alpha * (reward + self.gamma * best_next - row[action])
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    def save_q_table(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.q))

    # ── decision ─────────────────────────────────────────────────────────
    def decide(self, view: MarketView) -> list[Order]:
        sym = self.symbol
        rets = view.return_history.get(sym, [])
        price_change = (100.0 * (2.718281828 ** rets[-1] - 1)) if rets else 0.0
        equity = view.account.equity(view.prices) or 1.0
        pos_ratio = view.account.position(sym) * view.prices[sym] / equity
        state = discretize(price_change, view.sentiment, pos_ratio, view.realized_vol[sym])

        self._update(self._reward(view), state)
        action = self._choose(state)
        self._last = (state, action)

        if action == "hold":
            return []
        qty = int(self.config.get("trade_size", 50))
        slip = float(self.config.get("max_slippage", 0.004))
        book = view.books[sym]
        if action == "buy":
            qty = self.affordable_qty(view, sym, qty)
            if qty <= 0:
                return []
            ref = book.best_ask or view.prices[sym]
            return [Order(self.agent_id, sym, Side.BUY, qty, OrderType.LIMIT,
                          round(ref * (1 + slip), 4), reason=f"QL {state}")]
        ref = book.best_bid or view.prices[sym]
        return [Order(self.agent_id, sym, Side.SELL, qty, OrderType.LIMIT,
                      round(ref * (1 - slip), 4), reason=f"QL {state}")]

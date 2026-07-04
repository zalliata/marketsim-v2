# Agent reference

Every agent implements `on_tick(view) -> list[Order]`. Information access
(sentiment, graph features, peer messages) is filtered into the `MarketView` by
construction, so an ablated agent physically cannot see what it was denied.
Registry IDs and default configs are in `marketsim/agents/registry.py`.

## Zero-intelligence — Z1, Z2 (`zero_intelligence.py`)
Budget-constrained random traders (Gode & Sunder, 1993). Post limit orders
around the anchor value; ~25% of orders cross the spread for price discovery.
Z2 carries a bearish `side_bias`. Sentiment tilts the buy/sell split during
crisis. **Config:** side_bias, order_size_distribution{min,max},
trade_probability, quote_offset_std, max_position_per_stock.

## Momentum / Reversal — M1, M2 (`momentum_reversal.py`)
Signal math ported verbatim from v1 (Jegadeesh & Titman 1993; Jegadeesh 1990):
- momentum = Σ log-returns over [t−L−skip, t−skip), L=12, skip=1
- reversal = −Σ log-returns over last L steps, L=1
- z-score over a rolling window → target position = clamp(z·scale, ±max_inv)
- order = target − current
`hub_focused` restricts trading to the top-centrality symbols (graph access).
**Config:** signal, lookback, skip_recent, z_window, position_scale,
max_inventory, max_slippage, hub_focused.

## Market makers — P1, P2, P4 (`market_makers.py`)
- **P1 FixedSpreadMM** — constant half-spread (`spread_bps`), inventory capped.
- **P2 VolInventoryMM** — half-spread scales with realised vol; quote centre
  skews against inventory (Avellaneda–Stoikov 2008, discretised).
- **P4 SignalAwareMM** — P2 plus manipulation-indicator spread widening
  (sentiment shock, OFI spike, depth asymmetry). Rule-based by design; renamed
  from v1's misleading "LLM-Signal-Aware RL Defense". Optional LLM commentary
  never drives quotes.
Each MM cancels and reposts quotes every tick (feeds the cancellation-rate
metric). **Config:** quote_size, max_inventory, spread_bps / base_spread_bps,
vol_multiplier, inventory_skew, defense_multiplier, sentiment_shock_level,
ofi_spike.

## RL market maker — P3 (`rl_market_maker.py`)
Tabular Q-learning, ported from v1: 7×5×3×3 state discretisation, α=0.1, γ=0.95,
ε=0.3 decay ×0.995 → 0.05, ε-greedy over {buy, sell, hold}. Reward = Δ
mark-to-market equity, with presets:
- `standard` — pure PnL
- `aggressive` — PnL − inventory penalty
- `social` — PnL − vol_penalty·realised vol (social-planner variant)
Q-tables save/load as JSON → pretraining and transfer scenarios. **Config:**
learning_rate, discount_factor, exploration_rate/decay/min, reward_mode,
training, q_table_path, save_q_table, symbol, trade_size, vol_penalty.

## Adversarial LLM — A1, A2, A3 (`adversarial_llm.py`)
Volatility-maximising adversaries. Every decision flows through an `LLMClient`
(scripted or a real provider) given a permitted-fields market-state JSON;
response is `{action, symbol, quantity, rationale}`. A1 pure vol-max, A2
vol-profit hybrid, A3 limited-info + comm-constrained (1 message/tick). Targets
the top-centrality hub when graph access is on. **Config:** objective,
volatility_weight, profit_weight, base_size, max_position_per_stock,
max_slippage, target_symbol.

## Graph exploiters — G1, G2, G3 (`graph_agents.py`)
- **G1 HubAttacker** — sells the top-centrality hub during stress.
- **G2 PathwayExploiter** — attacks the strongest outgoing-contagion neighbours
  of the hub (cascade pathways).
- **G3 CrossSectorAmplifier** — trades peripheral nodes to spread stress.
Require graph access. **Config:** attack_size, n_pathways, n_periphery,
max_slippage.

## Graph defenders — D1, D2, D3 (`graph_agents.py`)
- **D1 CentralityLiquidityDefender** — tight two-sided quotes at top hubs.
- **D2 ContagionFirewallDefender** — firm bids on the highest-exposure
  transmitters to absorb sell pressure.
- **D3 AdaptiveGraphRebalancer** — concentrates liquidity at the currently
  most-stressed hub. **Config:** n_hubs, half_spread_bps, quote_size, bid_bps.

## Scenario controller — S1 (`controller.py`)
Never trades. Executes a timed schedule: sentiment shocks (incl. the 60%-negative
crisis injection), transaction-cost changes, and circuit-breaker halts, plus a
dynamic breaker that halts on a volatility or drawdown trigger (P3 arm B).
**Config:** sentiment_shocks[], tc_changes[], halts[], breaker{vol_trigger,
drawdown_trigger, halt_ticks, symbol}.

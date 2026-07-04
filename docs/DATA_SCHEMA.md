# Output data schema

Every backend (local CSV, Supabase) writes the same three tables. Columns match
the v1 exports where they existed; **new columns are appended**, so existing
P2–P4 analysis scripts continue to read the files unchanged.

## `time_steps` — per tick, per symbol
| column | type | notes |
|--------|------|-------|
| run_id | str | batch identifier |
| scenario_id | str | |
| iteration | int | Monte Carlo index |
| seed | int | `base_seed + iteration` |
| tick | int | 0-based; one tick = one 30-min bar |
| symbol | str | cohort member |
| price | float | last trade, else mid, else anchor |
| log_return | float | tick log-return |
| realized_volatility | float | rolling 20-tick RV |
| sentiment | float | [-1, 1] |
| best_bid, best_ask, spread | float | top of book (may be blank if one-sided) |
| bid_depth, ask_depth | int | shares within top 5 levels |
| **signed_ofi** | int | buy-initiated − sell-initiated volume (NEW) |
| **cancellations** | int | cancels this tick (NEW) |
| trade_count, volume | int | executed this tick |
| **price_impact** | float | \|Δmid\| per unit volume (NEW) |
| **halted** | 0/1 | circuit-breaker state (NEW) |
| **llm_mode** | str | scripted / anthropic / openai / gemini (NEW) |

The five NEW microstructure columns are what Papers 2/3/5/6 consume and what v1
could not produce.

## `scenario_iterations` — one row per iteration
Original columns: realized_volatility, max_volatility, time_in_high_vol_regime,
final_price, price_drawdown, price_return, adversary_pnl, defense_pnl,
total_trades, total_volume, bid_ask_spread_mean, sentiment_volatility,
min_sentiment, max_sentiment.

New columns: **cascade_onset_tick**, **cascade_frequency**,
**stabilisation_effectiveness** (P6 phase-transition inputs), **halts**,
**llm_mode**, **adversary_share** (P5/P6 sweep key), **label** (P5
manipulated/clean).

PnL note: `adversary_pnl` / `defense_pnl` are mark-to-market equity minus each
agent's *actual starting cash*, so they are correct even when the P5/P6 share
sweep scales adversary capital.

## `agent_snapshots` — one row per agent per iteration
run_id, scenario_id, iteration, seed, agent_id, agent_type, final_cash, equity,
fees_paid, trades_count, positions_json.

## Supabase note
The Supabase backend inserts the same fields; new columns are additive, so v1
dashboards keep working. Add the new columns to the Supabase tables (nullable)
before first use, or let the local-CSV fallback handle offline runs.

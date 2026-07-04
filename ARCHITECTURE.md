# marketsim-v2 — Architecture and Redesign Plan

**Crisis-calibrated multi-agent market simulator for the dissertation programme
"Algorithmic Trading, Financial Contagion, and the Architecture of Regulatory Resilience"**
Zorina Alliata · ASE Bucharest · July 2026

---

## 1. Why a redesign

The original codebase (`Original Version/p3-ai-agents-market-main`) is a Lovable-generated
React/Vite/Supabase application. It served Papers 0–2 but has five structural problems:

1. **The "LLM" agents never call an LLM.** The batch scenario runner
   (`src/lib/simulation/decisions.ts`) drives `adversarial_llm` agents with hard-coded
   if/else heuristics on sentiment thresholds. The only real LLM call (Gemini via the
   Lovable gateway in `supabase/functions/agent-reasoning`) is reachable only from the
   interactive UI — never from the Monte Carlo batches used in the papers.
2. **Simulation logic is split across three runtimes** — frontend TypeScript
   (`src/lib/simulation`), Deno edge functions (`run-simulation/index.ts`, 1,368 lines;
   `sim-iterate`, 694 lines), and UI hooks (`useScenarioRunner.ts`) — with duplicated,
   diverging decision logic (two different `adversarial_llm` implementations exist).
3. **No order book in the batch path.** Batch agents trade against a deterministic price
   curve with additive impact, so order-flow imbalance, depth, spreads, and cancellations —
   the observables Papers 3, 5, and 6 need — cannot be measured.
4. **Scenarios are data inside a UI file** and cannot express parameter sweeps
   (transaction-cost grids, market-share grids), which are the core designs of P3/P5/P6.
5. **No tests beyond one signals file, no docs of record.**

marketsim-v2 is a single-runtime Python package with one engine, one agent framework,
one scenario system, and one storage layer.

## 2. Design decisions (agreed 4 July 2026)

| Decision | Choice |
|---|---|
| Language | Python ≥ 3.10, standard-library + numpy/pandas core |
| LLM agents | Real API via pluggable `LLMClient` + deterministic `ScriptedClient` fallback |
| Interface | CLI (`marketsim …`) + YAML/JSON configs; no UI dependency |
| Storage | Supabase (same tables as v1 exports) with automatic local-CSV fallback |
| Original code | Untouched; kept side-by-side for comparison |

## 3. Package layout

```
marketsim-v2/
├── README.md                  Quick start, install, examples
├── ARCHITECTURE.md            This file
├── docs/
│   ├── AGENTS.md              Every agent: economics, algorithm, config, provenance vs v1
│   ├── SCENARIOS.md           Every scenario incl. paper mapping (P1–P6)
│   ├── DATA_SCHEMA.md         Output tables & columns (v1-compatible + extensions)
│   └── MIGRATION_NOTES.md     What changed vs Original Version, and why
├── pyproject.toml
├── configs/                   Example run configs and sweep grids (YAML)
├── marketsim/
│   ├── types.py               Order, Trade, Quote, MarketView, AgentAccount (dataclasses)
│   ├── engine/
│   │   ├── orderbook.py       Price–time-priority continuous double auction, per symbol
│   │   └── market.py          Multi-asset market: tick loop, fees, halts, settlement
│   ├── calibration/
│   │   ├── cohort.py          10-asset SVB cohort: correlation, contagion, centrality, vols
│   │   ├── anchors.py         Historical 30-min price path (Mar 8–10 2023) + extension
│   │   └── sentiment.py       Sentiment process with 60%-negative crisis injection
│   ├── agents/
│   │   ├── base.py            Agent ABC: on_tick(view) -> list[Order]; account mgmt
│   │   ├── registry.py        Canonical agent IDs (Z1..Z2, A1..A3, G1..G3, D1..D3, P1..P4, M1..M2, S1)
│   │   ├── zero_intelligence.py
│   │   ├── momentum_reversal.py   MOM & STREV (ported signal math, verbatim semantics)
│   │   ├── market_makers.py       Fixed-spread & vol/inventory-aware quoting
│   │   ├── rl_market_maker.py     Tabular Q-learning MM (ported; pretrain/transfer support)
│   │   ├── adversarial_llm.py     Volatility-maximising adversaries via LLMClient
│   │   ├── graph_exploiters.py    Hub attacker, pathway exploiter, cross-sector amplifier
│   │   ├── graph_defenders.py     Centrality liquidity, contagion firewall, adaptive rebalancer
│   │   └── controller.py          Scenario controller: shocks, halts, cost changes
│   ├── llm/
│   │   ├── client.py          LLMClient ABC + ScriptedClient (deterministic, seeded)
│   │   ├── providers.py       AnthropicClient / OpenAIClient / GeminiClient (env-key based)
│   │   └── prompts.py         System + state prompts (from v1 agent-reasoning, upgraded)
│   ├── scenarios/
│   │   ├── registry.py        Scenario dataclass, lookup, YAML loader
│   │   └── definitions.py     All 28 v1 scenarios + P3/P5/P6 batteries (sweep-aware)
│   ├── runner/
│   │   ├── simulation.py      One seeded run: wiring, tick loop, metric collection
│   │   ├── batch.py           N-iteration Monte Carlo (multiprocessing), aggregates
│   │   └── sweep.py           Parameter grids: tc bps, adversary share, composition
│   ├── metrics/
│   │   ├── microstructure.py  OFI, spread percentile, depth ratio, cancel rate, price impact
│   │   ├── systemic.py        Realised vol, cascade onset/frequency, stabilisation effectiveness
│   │   └── iteration.py       Per-iteration summary rows (v1 scenario_iterations schema)
│   ├── storage/
│   │   ├── base.py            ResultsBackend ABC; table schemas
│   │   ├── local_backend.py   CSV per table per run (v1 export-compatible)
│   │   └── supabase_backend.py  supabase-py, same tables; falls back to local if no creds
│   └── cli.py                 marketsim list | run | batch | sweep | train-rl
└── tests/                     pytest: engine, signals, agents, scenarios, metrics
```

## 4. The engine (what changes most vs v1)

A true continuous double auction per symbol replaces the price-curve-plus-impact model:

- **Limit order book** with price–time priority, partial fills, cancellations, market and
  limit orders. Every fill produces a `Trade` with maker/taker attribution.
- **Fundamental anchor.** The historical 30-minute SVB-crisis path (39 bars/symbol, ported
  from `historicalPrices.ts`) anchors valuations; runs longer than 39 steps extend the
  anchor with a seeded GARCH-like process calibrated to the cohort vols. Zero-intelligence
  flow quotes around the anchor, which keeps prices tethered without dictating them —
  strategic agents can and do move prices away.
- **Transaction costs** are a market parameter in basis points per side (P3's sweep
  variable), charged at settlement and logged per agent.
- **Circuit breakers**: the market supports halt windows (no matching, cancels allowed) —
  triggered by the scenario controller on realised-vol or drawdown triggers (P3 arm B).
- **Per-tick microstructure logging**: best bid/ask, mid, spread, bid/ask depth (top 5),
  signed order-flow imbalance, cancellations, executed volume, trade count. These feed
  Papers 2/3/5/6 directly and fix the v1 gap where `time_steps` had only price/vol/sentiment.

## 5. Agent framework

Every agent implements `on_tick(view: MarketView) -> list[Order]` where `MarketView` is a
read-only snapshot filtered by the agent's information access (sentiment feed, graph
features, peer messages — the v1 ablation switches, now enforced by construction rather
than by convention). Configs are dataclasses with the same field names as the v1 JSON
configs wherever semantics survived, so old agent definitions can be imported.

| ID | Agent | v1 status → v2 treatment |
|----|-------|--------------------------|
| Z1–Z2 | Zero-intelligence (uniform / side-biased) | Correct in v1 → ported, now quotes limit orders around anchor |
| M1–M2 | Momentum / Short-term reversal | Signal math correct in v1 (`signals.ts`) → ported verbatim, executes via book |
| P1 | Fixed-spread MM | Correct → ported, real two-sided quoting with inventory cap |
| P2 | Vol/inventory-aware MM | Correct → ported (Avellaneda–Stoikov-style skewing documented) |
| P3 | RL market maker | Q-learning correct but trained on price-curve world → ported (same state buckets, α, γ, ε schedule) and retrained on the book; supports pretrain → save Q-table → transfer (p4-pretrain-rl / p4-transfer-test) |
| P4 | "LLM-signal-aware" defensive MM | Was heuristic → renamed honestly to signal-aware MM; optional LLM anomaly commentary, decisions remain rule-based (documented) |
| A1–A3 | Adversarial LLM (vol-max, hybrid, limited-info / comm-constrained) | **Was fake** → now real: prompt + market state → `LLMClient` → JSON decision. `ScriptedClient` reproduces the documented adversarial policy deterministically for large batches; provider clients (Anthropic/OpenAI/Gemini) for genuine-LLM runs. Mode is recorded in every output row. |
| G1–G3 | Graph exploiters (hub attacker, pathway, cross-sector) | Heuristics on contagion matrix, correct idea → ported with explicit centrality/contagion helpers and order-book execution |
| D1–D3 | Graph defenders (centrality liquidity, firewall, rebalancer) | Same → ported |
| S1 | Scenario controller | Split across runtimes → single class: sentiment shocks (60%-negative injection), tc changes, halts, composition changes |

`docs/AGENTS.md` documents, per agent: the economic rationale (with citations), the exact
algorithm, every config field, and the v1 → v2 provenance note.

## 6. LLM integration

```
LLMClient (ABC)
├── ScriptedClient      deterministic, seeded; encodes the adversarial policy the papers
│                        describe; zero cost; default for Monte Carlo batteries
├── AnthropicClient     ANTHROPIC_API_KEY
├── OpenAIClient        OPENAI_API_KEY
└── GeminiClient        GEMINI_API_KEY (parity with v1's gateway model)
```

- Adversaries receive a compact market-state JSON (prices, returns, sentiment, their book,
  graph features if permitted) and must return `{action, symbol, quantity, rationale}`.
- Malformed responses → safe `hold` + logged incident (never a crash mid-batch).
- Comm-constrained variant (A3) enforces the 1-message-per-tick budget in the view layer.
- Every run records `llm_mode` (`scripted` / provider name) so papers can state precisely
  which results are genuine-LLM and which are scripted-policy.
- Cost control: `--llm-budget` caps API calls per run; batch runner refuses provider mode
  for grids that would exceed it.

## 7. Scenario system (P3/P5/P6 ready)

A `Scenario` is: agents (with per-agent config overrides), market parameters (tc bps,
halt rule, timesteps), shock schedule, and optional **sweep axes**. All 28 v1 scenario IDs
are reproduced. New batteries:

| Battery | Paper | Definition |
|---|---|---|
| `p3-cost-sweep` | P3/RQ-F4 | tc ∈ 0…50 bps grid × adversary config (A- or G-family), 100 iters/level |
| `p3-circuit-breaker` | P3/RQ-F4 | {no intervention, FTT@tc*, halt N ∈ {6,13,26} ticks} at same stress trigger |
| `p5-labelled-battery` | P5/RQ-F6 | 6 adversary configs + 3 clean baselines, per-window manipulated/clean labels |
| `p5-share-sweep` | P5/RQ-F6 | adversary market share 1–50% × {A,G} × matched clean runs |
| `p6-composition-sweep` | P6/RQ-F7 | adversarial share 5–50% × scenario battery, cascade/stabilisation metrics |
| `p6-monoculture` | P6/RQ-F7 | single-strategy vs mixed-strategy adversary mass at each share |

Sweeps run as: `marketsim sweep p3-cost-sweep --grid "tc_bps=0:50:1" --iterations 100`.
The share axes are implemented by scaling agent capital/order-size mass, logged as
`adversary_share` in every row (P5's classifier and P6's phase-transition analysis both key on it).

## 8. Storage

Primary: **Supabase** via `supabase-py`, writing the exact v1 tables
(`simulations`, `time_steps`, `agent_trades`, `agent_snapshots`, `scenario_runs`,
`scenario_iterations`, `scenario_run_metrics`) plus new columns (microstructure fields,
`llm_mode`, `adversary_share`, `label`) — additive, so existing dashboards keep working.
Credentials from `SUPABASE_URL` / `SUPABASE_SERVICE_KEY` env vars.
If credentials are absent or writes fail, the run transparently lands in
`results/<run_id>/*.csv` with identical schemas (`docs/DATA_SCHEMA.md`); a `marketsim
upload results/<run_id>` command syncs later. Analysis scripts for P2–P4 read either.

## 9. Documentation & quality standards

- Every module has a header docstring stating purpose and paper linkage; every public
  function has a docstring with units (bps, shares, log-returns) spelled out.
- `docs/AGENTS.md` and `docs/SCENARIOS.md` are the citable technical appendix
  (replaces/updates thesis Appendix A).
- Determinism: every run takes a seed; batch iteration i uses `seed + i`; scripted-LLM
  and RL exploration draw from the seeded generator only. Same seed ⇒ identical output.
- Tests (`pytest`): order-book invariants (price-time priority, conservation of shares/cash),
  signal math against hand-computed cases (ported from v1's `signals.test.ts` and extended),
  Q-learning update correctness, scenario registry completeness (all 28 v1 IDs present),
  metric definitions, end-to-end smoke runs for one scenario per battery.
- CI-ready: `pip install -e . && pytest` is the whole loop; no network needed for tests.

## 10. Migration map (v1 → v2)

| v1 location | v2 location |
|---|---|
| `src/lib/simulation/signals.ts` | `marketsim/agents/momentum_reversal.py` (+ tests) |
| `src/lib/qlearning.ts` | `marketsim/agents/rl_market_maker.py` |
| `src/lib/simulation/decisions.ts` (heuristics) | split into per-agent modules; adversarial branch → `llm/client.py:ScriptedClient` |
| `src/data/contagionGraph.ts` | `marketsim/calibration/cohort.py` |
| `src/data/historicalPrices.ts` | `marketsim/calibration/anchors.py` |
| `src/data/scenarios.ts` | `marketsim/scenarios/definitions.py` |
| `supabase/functions/run-simulation`, `sim-iterate`, `sim-batch` | `marketsim/runner/*` |
| `supabase/functions/agent-reasoning` prompts | `marketsim/llm/prompts.py` |
| UI analytics panels | out of scope (CSV/Supabase consumed by Python analysis scripts) |

## 11. Build phases

1. **Core** — types, order book, market loop, calibration, ZI agents, local storage, CLI `run` ✅ this delivery
2. **Agents** — MOM/STREV, MMs, RL, adversarial (scripted + providers), graph agents, controller ✅ this delivery
3. **Scenarios & sweeps** — full registry, batch runner, P3/P5/P6 batteries ✅ this delivery
4. **Supabase backend + upload command** ✅ this delivery (needs env keys to activate)
5. **Validation** — re-run `rq2-adversary-enters` and one p4-* scenario, compare distributions against v1 exports (next session, needs longer batch time)
6. **RL retraining** — pretrain P3 agent on the new engine, save canonical Q-table artefact (next session)

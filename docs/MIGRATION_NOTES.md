# Migration notes: Original Version → marketsim-v2

## The five problems in the original, and their resolution

1. **"LLM" adversaries never called an LLM (the central defect).**
   In `Original Version/src/lib/simulation/decisions.ts`, the batch scenario
   runner drove `adversarial_llm` agents with hard-coded if/else rules on
   sentiment thresholds. The only real model call (`supabase/functions/
   agent-reasoning`, Gemini via the Lovable gateway) was reachable only from the
   interactive UI, never from the Monte Carlo batches that produced the paper
   results. **Resolution:** `marketsim/agents/adversarial_llm.py` routes every
   decision through an `LLMClient`. `ScriptedClient` is a deterministic policy
   for cheap reproducible batches; `AnthropicClient` / `OpenAIClient` /
   `GeminiClient` make real API calls. Both use the same prompt/state/JSON-
   decision interface, and the mode is recorded per row. Papers can now state
   exactly which results are genuine-LLM.

2. **Logic split across three runtimes with divergent copies.**
   Decision logic existed in frontend TypeScript, Deno edge functions, and a UI
   hook, and two different `adversarial_llm` implementations had drifted apart.
   **Resolution:** one Python package, one implementation per agent.

3. **No order book in the batch path.** Batch agents traded against a
   deterministic price curve with additive impact, so order-flow imbalance,
   depth, spreads, and cancellations could not be measured — exactly the
   observables Papers 3/5/6 require. **Resolution:** a real price-time-priority
   continuous double auction (`engine/orderbook.py`) with per-tick microstructure
   logging (`types.TickMetrics`).

4. **Scenarios could not express sweeps.** They were data literals in a UI file.
   **Resolution:** `scenarios/registry.py` adds a `SweepAxis`, and
   `definitions.py` defines the P3 cost sweep, P5 share sweep, and P6 composition
   sweep as first-class objects the CLI can run.

5. **Almost no tests, no documentation of record.** **Resolution:** `pytest`
   suite (order-book invariants, ported signal math, Q-learning updates, registry
   completeness, determinism, smoke runs) and the `docs/` set.

## Agent correctness audit (the user's specific concern)

| Agent | Original state | v2 |
|-------|---------------|-----|
| Zero-intelligence | Correct (random flow) | Ported; now posts limit orders around the anchor so the book stays two-sided |
| Momentum / Reversal | **Signal math correct** (`signals.ts`, the one tested file) | Ported verbatim (`momentum_reversal.py`) with the same tests re-implemented; now executes through the book |
| Fixed-spread MM | Correct | Ported; real two-sided quoting |
| Vol/Inventory MM | Correct (Avellaneda–Stoikov-style) | Ported with the skew documented |
| RL market maker | Q-learning correct, but trained against the price curve, and reward proxies were ad hoc | Ported with identical state buckets/α/γ/ε; reward = Δ mark-to-market equity with the p4-rl-reward-* presets; retrains on the book; Q-tables save/load for pretrain→transfer |
| "LLM-signal-aware RL defense" | Neither LLM nor RL in batch mode — a heuristic | Renamed **SignalAwareMM**; manipulation indicators documented; rule-based by design, optional LLM commentary only |
| Adversarial LLM | **Fake** (heuristic) | Real LLM or scripted policy through `LLMClient` |
| Graph exploiters/defenders | Heuristics on the contagion matrix; idea sound | Ported with explicit centrality/contagion helpers and book execution |
| Scenario controller | Scattered across edge functions | One class with a timed event schedule + P3 circuit-breaker trigger |

## Resolved refinements
- **P6 composition axis (done, July 2026).** Paper 5 empirically confirmed that
  the capital-based `adversary_share` knob is footprint-invariant: detection and
  order-flow footprint did not change from 1% to 50% capital, because adversary
  order sizing is fixed. P6 therefore now sweeps `composition_share` — the
  adversarial fraction of the *trading population*. `run_once` replicates noise
  and adversary agents to a fixed 20-trader population at the target fraction,
  holding market-maker / defender infrastructure fixed, so the adversarial share
  of order flow genuinely grows with the sweep (validated in
  `tests/test_composition.py`). The capital `adversary_share` path is retained
  for the P5 sweeps. Monoculture vs diversity is expressed by the roster's
  adversary set (one type vs several), which the replication cycles.

## Known refinements deferred to the next session
- **RL retraining artefact.** `p4-pretrain-rl` will produce the canonical
  Q-table; the transfer scenarios consume it. Needs a longer training batch.
- **Distribution validation.** Re-run `rq2-adversary-enters` and one p4-*
  scenario at 100 iterations and compare against the v1 Supabase exports.
- **P5 capital-share note.** The P5 `adversary_share` sweeps remain on the
  capital axis by design (that is what produced the footprint-invariance
  finding); no change needed.

# Scenario reference

Run `marketsim list` for the live catalogue (49 scenarios) and
`marketsim show <id>` for detail. Scenarios are defined in
`marketsim/scenarios/definitions.py`.

## Part A — Papers 0–2 / paper-4 coordination battery (v1 parity)
All 32 original scenario IDs are reproduced verbatim in roster and intent:

- **RQ1 baselines:** baseline-noise-only, baseline-with-mm, baseline-all-defenders
- **RQ2 adversary:** rq2-adversary-enters, rq2-adversary-vs-all-mm,
  rq2-two-adversaries, rq2-info-disadvantage
- **RQ3 defense / ablations:** rq3-defense-comparison, rq3-ablation-no-sentiment,
  rq3-ablation-no-graph
- **RQ4 graph arena:** rq4-full-arena, rq4-graph-exploiters-compete,
  rq4-graph-defenders-compete, rq4-graph-vs-llm-adversary,
  rq4-graph-vs-llm-defender, rq4-ablation-no-graph-exploiter
- **Crisis:** crisis-intensified, crisis-full-cast
- **Paper-4 RL/coordination battery:** p4-pretrain-rl, p4-solo-{rulebased,
  momstrev,rl}, p4-single-llm-{rulebased,momstrev,rl}, p4-coord-llm-{rulebased,
  momstrev,rl}, p4-rl-reward-{standard,social,aggressive}, p4-transfer-test.
  (v1's "MOM-STREV ML market maker" maps to M1; the RL reward variants set the
  RL agent's `reward_mode`; pretrain/transfer use Q-table save/load.)

## Part B — future-paper batteries (sweep-aware)

### Paper 3 / RQ-F4 — cost floors & circuit breakers
- **p3-cost-sweep-adversary** — tc 0–50 bps vs A1/A2/A3; find tc* where mean
  adversary PnL < 0. (Refine the grid near the crossing after the coarse pass.)
- **p3-cost-sweep-graph** — tc 0–50 bps vs G1/G2/G3; the P2 result at 10 bps is
  one grid point.
- **p3-circuit-breaker** — intervention arms {none, FTT@tc*, halt} at a common
  drawdown trigger, for the welfare comparison.

Run: `marketsim sweep p3-cost-sweep-adversary --iterations 100`.

### Paper 5 / RQ-F6 — detection
- **Labelled battery** (for the classifier): p5-clean-{noise,mm,defense} and
  p5-manip-{A1,A2,A3,G1,G2,G3}, each carrying a `regime` label. Clean runs span
  the same defensive configs as manipulated runs so the classifier learns
  manipulation, not defense presence.
- **p5-share-sweep-llm / p5-share-sweep-graph** — adversary share 1–50%; find the
  minimum detectable share, and test whether graph strategies are detectable at
  lower share than pure-vol (H2).

### Paper 6 / RQ-F7 — concentration & phase transition
- **p6-composition-sweep** — adversarial share 5–50% across the defended arena;
  locate the stabilisation-breakdown threshold.
- **p6-monoculture / p6-diversity** — single vs mixed adversary strategy at each
  share level; test whether diversity delays the transition (H2).

`metrics/systemic.py::locate_phase_transition` turns a share-sweep's
stabilisation and cascade curves into the threshold estimate.

## Sweep mechanics
A scenario's `SweepAxis` has `kind ∈ {tc_bps, adversary_share, breaker}`. The
sweep runner (`runner/batch.py::run_sweep`) runs a full Monte-Carlo batch per
grid point and tags every row with the grid value (`tc_bps` on the market,
`adversary_share` in the iteration summary), which is what the P3/P5/P6 analysis
scripts key on.

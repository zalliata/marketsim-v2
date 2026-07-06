"""All scenario definitions.

Part A reproduces every v1 scenario (32 IDs, verbatim agent rosters and intent;
MOM-STREV-ML market maker is mapped to M1 as the signal-driven MM, and the RL
reward variants set the RL agent's reward_mode). Part B adds the sweep-aware
batteries for Papers P3, P5, P6 that v1 could not express.

`SCENARIOS` is the public registry (id -> Scenario). `get(id)` looks one up.
"""
from __future__ import annotations

from .registry import Scenario, SweepAxis, frange

_S: dict[str, Scenario] = {}


def _add(sc: Scenario) -> None:
    _S[sc.id] = sc


# ══════════════════════════════════════════════════════════════════════
# Part A — v1 scenarios (Papers 1 & the paper-4 coordination battery)
# ══════════════════════════════════════════════════════════════════════

# RQ1 baselines
_add(Scenario("baseline-noise-only", "RQ1: Noise-Only Market",
              "Two zero-intelligence agents establish baseline price dynamics.",
              ["Z1", "Z2"], paper="P1/RQ-F1"))
_add(Scenario("baseline-with-mm", "RQ1: Market Maker Baseline",
              "Noise traders with a fixed-spread market maker.",
              ["Z1", "Z2", "P1"], paper="P1/RQ-F1"))
_add(Scenario("baseline-all-defenders", "RQ1: All Market Makers Compete",
              "Four market-making strategies compete for spread profit.",
              ["Z1", "P1", "P2", "P3", "P4"], paper="P1/RQ-F1"))

# RQ2 adversary
_add(Scenario("rq2-adversary-enters", "RQ2: Adversary vs Noise",
              "One LLM adversary enters a noise-only market.",
              ["Z1", "Z2", "A1"], paper="P1/RQ-F2"))
_add(Scenario("rq2-adversary-vs-all-mm", "RQ2: Adversary vs All MMs",
              "Adversary against the full market-maker stack.",
              ["Z1", "Z2", "A1", "P1", "P2", "P3", "P4"], paper="P1/RQ-F2"))
_add(Scenario("rq2-two-adversaries", "RQ2: Two Adversaries",
              "Pure-vol and hybrid adversaries together.",
              ["Z1", "Z2", "A1", "A2", "P2"], paper="P1/RQ-F2"))
_add(Scenario("rq2-info-disadvantage", "RQ2: Information-Disadvantaged Adversary",
              "Comm-constrained limited-info adversary (A3).",
              ["Z1", "Z2", "A3", "P2"], paper="P1/RQ-F2"))

# RQ3 defense comparison & ablations
_add(Scenario("rq3-defense-comparison", "RQ3: Defense Comparison",
              "All defenders vs coordinated adversaries.",
              ["Z1", "Z2", "A1", "A2", "P1", "P2", "P4"], paper="P1/RQ-F3"))
_add(Scenario("rq3-ablation-no-sentiment", "RQ3: Ablation — No Sentiment",
              "Adversary with sentiment access removed.",
              ["Z1", "Z2", "A1", "P2"], paper="P1/RQ-F3",
              agent_overrides={"A1": {"_ablate_sentiment": True}}))
_add(Scenario("rq3-ablation-no-graph", "RQ3: Ablation — No Graph",
              "Graph exploiter with graph access removed.",
              ["Z1", "Z2", "G1", "P2"], paper="P1/RQ-F3",
              agent_overrides={"G1": {"_ablate_graph": True}}))

# RQ4 graph arena
_add(Scenario("rq4-full-arena", "RQ4: Full Arena",
              "Graph exploiters and defenders plus LLM adversary.",
              ["Z1", "Z2", "A1", "G1", "G2", "D1", "D2", "P2"], paper="P1/RQ-F4"))
_add(Scenario("rq4-graph-exploiters-compete", "RQ4: Graph Exploiters Compete",
              "Three graph exploiters vs one defender.",
              ["Z1", "G1", "G2", "G3", "D1"], paper="P1/RQ-F4"))
_add(Scenario("rq4-graph-defenders-compete", "RQ4: Graph Defenders Compete",
              "One hub attacker vs three graph defenders.",
              ["Z1", "G1", "D1", "D2", "D3"], paper="P1/RQ-F4"))
_add(Scenario("rq4-graph-vs-llm-adversary", "RQ4: Graph Defenders vs LLM Adversary",
              "Graph defenders face an LLM adversary.",
              ["Z1", "Z2", "A1", "D1", "D2", "D3"], paper="P1/RQ-F4"))
_add(Scenario("rq4-graph-vs-llm-defender", "RQ4: Graph Exploiters vs Signal-Aware MM",
              "Graph exploiters vs the signal-aware defensive MM.",
              ["Z1", "Z2", "G1", "G2", "G3", "P4"], paper="P1/RQ-F4"))
_add(Scenario("rq4-ablation-no-graph-exploiter", "RQ4: Ablation — No Graph Exploiter",
              "Defender stack with the exploiter removed (control).",
              ["Z1", "Z2", "D1", "D2", "D3"], paper="P1/RQ-F4"))

# Crisis intensity
_add(Scenario("crisis-intensified", "Crisis: Intensified",
              "Stronger negative sentiment injection at crisis onset.",
              ["Z1", "Z2", "A1", "A2", "P2"], paper="P1",
              controller_config={"sentiment_shocks": [{"tick": 14, "delta": -0.4}]}))
_add(Scenario("crisis-full-cast", "Crisis: Full Cast",
              "Every agent family active through the crisis.",
              ["Z1", "Z2", "A1", "A2", "G1", "D1", "P2", "P4", "M1", "M2"], paper="P1"))

# ── Paper-4 coordination battery (the p4-* scenarios in the shipped data) ──
_add(Scenario("p4-pretrain-rl", "P4: RL Pre-Training (Noise Baseline)",
              "RL MM trains against noise traders; establishes Q-table baseline.",
              ["Z1", "Z2", "P3"], paper="P4",
              agent_overrides={"P3": {"training": True, "q_table_path": "artifacts/ql_pretrain.json",
                                      "save_q_table": True}}))
_add(Scenario("p4-solo-rulebased", "P4: Solo Rule-Based MM",
              "Vol-inventory MM alone with noise traders.",
              ["Z1", "Z2", "P2"], paper="P4"))
_add(Scenario("p4-solo-momstrev", "P4: Solo MOM-STREV MM",
              "Signal-driven MM alone with noise traders.",
              ["Z1", "Z2", "M1"], paper="P4"))
_add(Scenario("p4-solo-rl", "P4: Solo RL Market Maker",
              "Post-training RL MM alone with noise traders.",
              ["Z1", "Z2", "P3"], paper="P4",
              agent_overrides={"P3": {"training": False, "q_table_path": "artifacts/ql_pretrain.json"}}))
_add(Scenario("p4-single-llm-rulebased", "P4: Rule-Based vs Single LLM",
              "Rule-based MM faces one LLM adversary.",
              ["Z1", "Z2", "A1", "P2"], paper="P4"))
_add(Scenario("p4-single-llm-momstrev", "P4: MOM-STREV vs Single LLM",
              "Signal MM faces one LLM adversary.",
              ["Z1", "Z2", "A1", "M1"], paper="P4"))
_add(Scenario("p4-single-llm-rl", "P4: RL MM vs Single LLM",
              "RL MM faces one LLM adversary.",
              ["Z1", "Z2", "A1", "P3"], paper="P4",
              agent_overrides={"P3": {"training": True, "q_table_path": "artifacts/ql_single.json",
                                      "save_q_table": True}}))
_add(Scenario("p4-coord-llm-rulebased", "P4: Rule-Based vs Coordinated LLMs",
              "Rule-based MM faces two coordinating LLM adversaries.",
              ["Z1", "Z2", "A1", "A2", "P2"], paper="P4"))
_add(Scenario("p4-coord-llm-momstrev", "P4: MOM-STREV vs Coordinated LLMs",
              "Signal MM faces two coordinating LLM adversaries.",
              ["Z1", "Z2", "A1", "A2", "M1"], paper="P4"))
_add(Scenario("p4-coord-llm-rl", "P4: RL MM vs Coordinated LLMs",
              "RL MM faces two coordinating LLM adversaries (online learning).",
              ["Z1", "Z2", "A1", "A2", "P3"], paper="P4",
              agent_overrides={"P3": {"training": True}}))
_add(Scenario("p4-rl-reward-standard", "P4: RL Standard Reward vs Coord LLMs",
              "RL MM, standard PnL reward.",
              ["Z1", "Z2", "A1", "A2", "P3"], paper="P4",
              agent_overrides={"P3": {"reward_mode": "standard", "training": True}}))
_add(Scenario("p4-rl-reward-social", "P4: RL Social Reward vs Coord LLMs",
              "RL MM, socially-robust reward (volatility penalty).",
              ["Z1", "Z2", "A1", "A2", "P3"], paper="P4",
              agent_overrides={"P3": {"reward_mode": "social", "training": True}}))
_add(Scenario("p4-rl-reward-aggressive", "P4: RL Aggressive Reward vs Coord LLMs",
              "RL MM, aggressive PnL-only reward.",
              ["Z1", "Z2", "A1", "A2", "P3"], paper="P4",
              agent_overrides={"P3": {"reward_mode": "aggressive", "training": True}}))
_add(Scenario("p4-transfer-test", "P4: RL Transfer Test (Single->Coord)",
              "RL MM trained on single-LLM, tested on coordinated, no further learning.",
              ["Z1", "Z2", "A1", "A2", "P3"], paper="P4",
              agent_overrides={"P3": {"training": False, "q_table_path": "artifacts/ql_single.json"}}))

# ══════════════════════════════════════════════════════════════════════
# Part B — future-paper batteries (P3, P5, P6): sweep-aware
# ══════════════════════════════════════════════════════════════════════

# P3 / RQ-F4 — transaction cost floor sweep (0..50 bps)
_add(Scenario("p3-cost-sweep-adversary", "P3: Transaction-Cost Sweep (LLM adversaries)",
              "Sweep tc 0-50 bps against coordinated LLM adversaries; find tc* where "
              "mean adversary PnL < 0 (RQ-F4).",
              ["Z1", "Z2", "A1", "A2", "A3", "P2"], paper="P3/RQ-F4",
              sweep=SweepAxis("tc_bps", frange(0, 50, 5))))   # refine near crossing later
_add(Scenario("p3-cost-sweep-graph", "P3: Transaction-Cost Sweep (graph exploiters)",
              "Sweep tc 0-50 bps against graph exploiters (P2's 10bp result is one grid point).",
              ["Z1", "Z2", "G1", "G2", "G3", "D1"], paper="P3/RQ-F4",
              sweep=SweepAxis("tc_bps", frange(0, 50, 5))))
_add(Scenario("p3-circuit-breaker", "P3: Circuit Breaker vs FTT",
              "Compare intervention arms at a common stress trigger (RQ-F4 welfare).",
              ["Z1", "Z2", "A1", "A2", "P2"], paper="P3/RQ-F4",
              controller_config={"breaker": {"symbol": "SIVB", "drawdown_trigger": 0.15,
                                             "halt_ticks": 13}},
              labels={"ftt_tc_bps": 40.0},   # tc* from the cost sweep — update after 1 bps refinement
              sweep=SweepAxis("breaker", [0, 1, 2],
                              labels=["no-intervention", "ftt-at-tc*", "halt"])))

# P5 / RQ-F6 — labelled detection battery + market-share feasibility sweep
for _lab, _roster, _regime in [
    ("p5-clean-noise", ["Z1", "Z2"], "clean"),
    ("p5-clean-mm", ["Z1", "Z2", "P2"], "clean"),
    ("p5-clean-defense", ["Z1", "Z2", "P2", "P4", "D1"], "clean"),
    ("p5-manip-A1", ["Z1", "Z2", "A1", "P2"], "manipulated"),
    ("p5-manip-A2", ["Z1", "Z2", "A2", "P2"], "manipulated"),
    ("p5-manip-A3", ["Z1", "Z2", "A3", "P2"], "manipulated"),
    ("p5-manip-G1", ["Z1", "Z2", "G1", "P2"], "manipulated"),
    ("p5-manip-G2", ["Z1", "Z2", "G2", "P2"], "manipulated"),
    ("p5-manip-G3", ["Z1", "Z2", "G3", "P2"], "manipulated"),
]:
    _add(Scenario(_lab, f"P5 detection: {_lab}",
                  "Labelled run for the manipulation-detection classifier (RQ-F6).",
                  _roster, paper="P5/RQ-F6", labels={"regime": _regime}))

_add(Scenario("p5-share-sweep-llm", "P5: Detection Feasibility Sweep (LLM)",
              "Vary adversary market share 1-50%; find minimum detectable share (RQ-F6).",
              ["Z1", "Z2", "A1", "P2"], paper="P5/RQ-F6", labels={"regime": "manipulated"},
              sweep=SweepAxis("adversary_share",
                              [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])))
_add(Scenario("p5-share-sweep-graph", "P5: Detection Feasibility Sweep (graph)",
              "Vary graph-exploiter share 1-50%; graph strategies expected detectable "
              "at lower share than pure-vol (RQ-F6 H2).",
              ["Z1", "Z2", "G1", "P2"], paper="P5/RQ-F6", labels={"regime": "manipulated"},
              sweep=SweepAxis("adversary_share",
                              [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])))

# P6 / RQ-F7 — composition sweep & monoculture vs diversity
_add(Scenario("p6-composition-sweep", "P6: Algorithmic Composition Sweep",
              "Vary adversarial share 5-50% across the defended arena; locate the "
              "phase transition where stabilisation breaks down (RQ-F7 H1).",
              ["Z1", "Z2", "A1", "A2", "G1", "P2", "P4", "D1"], paper="P6/RQ-F7",
              sweep=SweepAxis("adversary_share",
                              [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])))
_add(Scenario("p6-monoculture", "P6: Monoculture Adversary",
              "Single dominant adversary strategy at each share level (RQ-F7 H2).",
              ["Z1", "Z2", "A1", "P2", "P4"], paper="P6/RQ-F7",
              sweep=SweepAxis("adversary_share",
                              [0.05, 0.10, 0.20, 0.30, 0.40, 0.50])))
_add(Scenario("p6-diversity", "P6: Mixed-Strategy Adversary",
              "Diverse adversary mix at each share level; expected to delay the phase "
              "transition vs monoculture (RQ-F7 H2).",
              ["Z1", "Z2", "A1", "A2", "G1", "G3", "P2", "P4"], paper="P6/RQ-F7",
              sweep=SweepAxis("adversary_share",
                              [0.05, 0.10, 0.20, 0.30, 0.40, 0.50])))


SCENARIOS = _S


def get(scenario_id: str) -> Scenario:
    if scenario_id not in _S:
        raise KeyError(f"unknown scenario '{scenario_id}'. "
                       f"Run `marketsim list` to see all {len(_S)} scenarios.")
    return _S[scenario_id]


def all_ids() -> list[str]:
    return list(_S.keys())

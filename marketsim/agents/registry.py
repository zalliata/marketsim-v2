"""Canonical agent catalogue: IDs, classes, default configs, info access, cash.

The registry is the single source of truth mapping the dissertation's agent
labels (Z1, A1, G1, P1, D1, M1, S1, ...) to implementations. Scenarios reference
agents by these IDs. Initial balances match v1 `constants.ts::INITIAL_BALANCE`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .base import Agent, InfoAccess
from .zero_intelligence import ZeroIntelligenceAgent
from .momentum_reversal import SignalAgent
from .market_makers import FixedSpreadMM, VolInventoryMM, SignalAwareMM
from .rl_market_maker import QLearningMM
from .adversarial_llm import AdversarialLLMAgent
from .graph_agents import (HubAttacker, PathwayExploiter, CrossSectorAmplifier,
                           CentralityLiquidityDefender, ContagionFirewallDefender,
                           AdaptiveGraphRebalancer)
from .controller import ScenarioController


@dataclass
class AgentSpec:
    id: str
    name: str
    agent_type: str
    factory: Callable[..., Agent]
    cash: float
    info: InfoAccess = field(default_factory=InfoAccess)
    config: dict = field(default_factory=dict)


CASH = {
    "zero_intelligence": 1_000_000, "adversarial_llm": 500_000,
    "adversarial_graph_exploiter": 500_000, "pricing_rule_based": 10_000_000,
    "pricing_rl": 10_000_000, "pricing_llm_signal_aware": 10_000_000,
    "pricing_graph_defender": 10_000_000, "momentum": 2_000_000,
    "short_term_reversal": 2_000_000, "scenario_controller": 0,
}

_full = InfoAccess(sentiment=True, graph_features=True, peer_messages=True)
_graph = InfoAccess(sentiment=True, graph_features=True)
_sent = InfoAccess(sentiment=True)


REGISTRY: dict[str, AgentSpec] = {
    # Zero-intelligence
    "Z1": AgentSpec("Z1", "Zero-Intelligence Liquidity", "zero_intelligence",
                    ZeroIntelligenceAgent, CASH["zero_intelligence"], _sent,
                    {"side_bias": 0.5}),
    "Z2": AgentSpec("Z2", "Side-Biased Zero-Intelligence", "zero_intelligence",
                    ZeroIntelligenceAgent, CASH["zero_intelligence"], _sent,
                    {"side_bias": 0.42}),
    # Momentum / reversal
    "M1": AgentSpec("M1", "Momentum Trend Follower", "momentum", SignalAgent,
                    CASH["momentum"], _sent, {"signal": "momentum", "lookback": 12}),
    "M2": AgentSpec("M2", "Short-Term Reversal Contrarian", "short_term_reversal",
                    SignalAgent, CASH["short_term_reversal"], _sent,
                    {"signal": "reversal", "lookback": 1}),
    # Market makers
    "P1": AgentSpec("P1", "Fixed-Spread Baseline MM", "pricing_rule_based",
                    FixedSpreadMM, CASH["pricing_rule_based"], _sent, {"spread_bps": 20}),
    "P2": AgentSpec("P2", "Vol-Inventory Aware MM", "pricing_rule_based",
                    VolInventoryMM, CASH["pricing_rule_based"], _sent, {}),
    "P3": AgentSpec("P3", "RL Market Maker", "pricing_rl", QLearningMM,
                    CASH["pricing_rl"], _sent, {"symbol": "SIVB"}),
    "P4": AgentSpec("P4", "Signal-Aware Defensive MM", "pricing_llm_signal_aware",
                    SignalAwareMM, CASH["pricing_llm_signal_aware"], _sent, {}),
    # Adversarial LLM
    "A1": AgentSpec("A1", "Pure Volatility Maximiser", "adversarial_llm",
                    AdversarialLLMAgent, CASH["adversarial_llm"], _full,
                    {"objective": "maximize_volatility", "volatility_weight": 1.0}),
    "A2": AgentSpec("A2", "Volatility-Profit Hybrid", "adversarial_llm",
                    AdversarialLLMAgent, CASH["adversarial_llm"], _full,
                    {"objective": "volatility_profit_hybrid", "volatility_weight": 0.6,
                     "profit_weight": 0.4}),
    "A3": AgentSpec("A3", "Limited-Info Adversary", "adversarial_llm",
                    AdversarialLLMAgent, CASH["adversarial_llm"],
                    InfoAccess(sentiment=False, graph_features=False,
                               peer_messages=True, max_messages_per_tick=1),
                    {"objective": "maximize_volatility"}),
    # Graph exploiters
    "G1": AgentSpec("G1", "Graph Hub Attacker", "adversarial_graph_exploiter",
                    HubAttacker, CASH["adversarial_graph_exploiter"], _graph, {}),
    "G2": AgentSpec("G2", "Graph Pathway Exploiter", "adversarial_graph_exploiter",
                    PathwayExploiter, CASH["adversarial_graph_exploiter"], _graph, {}),
    "G3": AgentSpec("G3", "Cross-Sector Amplifier", "adversarial_graph_exploiter",
                    CrossSectorAmplifier, CASH["adversarial_graph_exploiter"], _graph, {}),
    # Graph defenders
    "D1": AgentSpec("D1", "Centrality Liquidity Defender", "pricing_graph_defender",
                    CentralityLiquidityDefender, CASH["pricing_graph_defender"], _graph, {}),
    "D2": AgentSpec("D2", "Contagion Firewall Defender", "pricing_graph_defender",
                    ContagionFirewallDefender, CASH["pricing_graph_defender"], _graph, {}),
    "D3": AgentSpec("D3", "Adaptive Graph Rebalancer", "pricing_graph_defender",
                    AdaptiveGraphRebalancer, CASH["pricing_graph_defender"], _graph, {}),
    # Controller
    "S1": AgentSpec("S1", "Scenario Controller", "scenario_controller",
                    ScenarioController, CASH["scenario_controller"], _full, {}),
}


def build_agent(agent_id: str, seed: int, overrides: Optional[dict] = None,
                llm_client=None, instance_id: Optional[str] = None) -> Agent:
    """Build an agent from its registry base id. ``instance_id`` overrides the
    agent's id when replicating a type into a population (composition sweeps);
    the distinct id gives each replica its own seeded RNG stream, so replicas
    are independent and the run stays deterministic."""
    spec = REGISTRY[agent_id]
    cfg = dict(spec.config)
    if overrides:
        cfg.update(overrides)
    kwargs = dict(agent_id=instance_id or spec.id, name=spec.name,
                  agent_type=spec.agent_type, initial_cash=spec.cash,
                  info=spec.info, seed=seed, config=cfg)
    if spec.agent_type == "adversarial_llm":
        return spec.factory(**kwargs, llm_client=llm_client)
    return spec.factory(**kwargs)

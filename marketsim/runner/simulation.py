"""Run one seeded simulation of a scenario and collect metrics.

Wiring: build the market, instantiate the scenario's agents (with per-agent
overrides, info-access ablations, a shared coordination message bus, and the
chosen LLM client), run the tick loop, and return the tick-metrics log plus the
iteration summary. Deterministic given (scenario, seed, llm_mode).

Two market-share mechanisms are supported and must not be confused:

- ``adversary_share`` (P5) scales an adversary's CAPITAL. It does not change the
  adversary's order footprint, so it does not vary the adversarial fraction of
  market activity. Retained for backward compatibility with the P5 sweeps.
- ``composition_share`` (P6 / RQ-F7) scales the adversarial fraction of the
  trading POPULATION: the noise-trader and adversary agents are replicated so
  that adversaries make up the target fraction of the population, holding the
  market-maker and graph-defender infrastructure fixed. This genuinely varies
  the adversarial share of order flow and is the correct axis for the
  phase-transition analysis. Paper 5 established empirically that the capital
  axis is footprint-invariant, which is why P6 uses this population axis instead.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..agents.base import Agent, InfoAccess
from ..agents.controller import ScenarioController
from ..agents.registry import REGISTRY, build_agent
from ..calibration.anchors import CRISIS_TICK
from ..engine.market import Market, MarketConfig
from ..llm.providers import make_client
from ..metrics.iteration import summarize, IterationSummary
from ..scenarios.registry import Scenario
from ..types import TickMetrics

DEFAULT_POPULATION = 20   # trading-population size for composition sweeps


@dataclass
class RunResult:
    scenario_id: str
    seed: int
    ticks: list[TickMetrics]
    summary: IterationSummary
    agent_final: dict[str, dict]
    llm_mode: str
    agent_types: dict[str, str]       # instance_id -> base agent_type (covers replicas)


def _apply_ablations(info: InfoAccess, overrides: dict) -> InfoAccess:
    return InfoAccess(
        sentiment=info.sentiment and not overrides.get("_ablate_sentiment", False),
        graph_features=info.graph_features and not overrides.get("_ablate_graph", False),
        peer_messages=info.peer_messages,
        max_messages_per_tick=info.max_messages_per_tick,
    )


def _classify(agent_type: str) -> str:
    if agent_type.startswith("adversarial"):
        return "adversary"
    if agent_type == "zero_intelligence":
        return "noise"
    return "infrastructure"   # market makers, graph defenders, controller, signal traders


def _make(scenario, aid, instance_id, seed, llm):
    """Build one agent instance (possibly a replica) from a roster base id."""
    spec = REGISTRY[aid]
    ov = dict(scenario.agent_overrides.get(aid, {}))
    agent = build_agent(aid, seed=seed, overrides=ov, instance_id=instance_id,
                        llm_client=llm if spec.agent_type == "adversarial_llm" else None)
    agent.info = _apply_ablations(spec.info, ov)
    return agent, spec.agent_type


def _population_roster(scenario, composition_share, seed, llm):
    """Return [(agent, base_agent_type)] with the adversarial population fraction
    set to ``composition_share``. Infrastructure agents are kept as-is; noise and
    adversary agents are replicated to fill the population."""
    pop = int(scenario.labels.get("population_size", DEFAULT_POPULATION))
    adv_ids = [a for a in scenario.agent_ids if _classify(REGISTRY[a].agent_type) == "adversary"]
    noise_ids = [a for a in scenario.agent_ids if _classify(REGISTRY[a].agent_type) == "noise"] or ["Z1"]
    infra_ids = [a for a in scenario.agent_ids if _classify(REGISTRY[a].agent_type) == "infrastructure"]

    n_adv = int(round(composition_share * pop))
    if adv_ids and composition_share > 0:
        n_adv = max(1, n_adv)          # at least one adversary once share > 0
    n_noise = max(0, pop - n_adv)

    built = []
    for i in range(n_adv):             # adversaries cycle roster types (mono=1, diversity=many)
        base = adv_ids[i % len(adv_ids)]
        built.append(_make(scenario, base, f"{base}#{i}", seed, llm))
    for i in range(n_noise):           # legitimate noise-trader population
        base = noise_ids[i % len(noise_ids)]
        built.append(_make(scenario, base, f"{base}#{i}", seed, llm))
    for aid in infra_ids:              # fixed infrastructure (MMs, defenders, controller)
        built.append(_make(scenario, aid, aid, seed, llm))
    return built


def run_once(scenario: Scenario, seed: int = 0, llm_mode: str = "scripted",
             tc_bps: float | None = None, adversary_share: float | None = None,
             composition_share: float | None = None,
             baseline_peak_vol: float | None = None) -> RunResult:
    mkt = Market(MarketConfig(
        n_ticks=scenario.n_ticks,
        tc_bps=tc_bps if tc_bps is not None else scenario.tc_bps,
        seed=seed, crisis_enabled=scenario.crisis_enabled))

    llm = make_client(llm_mode, seed=seed)
    agents: list[Agent] = []
    agent_types: dict[str, str] = {}
    controller: ScenarioController | None = None

    if composition_share is not None:
        # population-composition mode (P6): replicate to hit adversarial fraction
        for agent, base_type in _population_roster(scenario, composition_share, seed, llm):
            mkt.register(agent.agent_id, agent.initial_cash)
            agent_types[agent.agent_id] = base_type
            if isinstance(agent, ScenarioController):
                controller = agent
            agents.append(agent)
    else:
        # standard / capital-scaling mode (P1-P5)
        for aid in scenario.agent_ids:
            spec = REGISTRY[aid]
            agent, base_type = _make(scenario, aid, aid, seed, llm)
            if adversary_share is not None and base_type.startswith("adversarial"):
                agent.initial_cash *= max(adversary_share / 0.10, 0.01)  # 10% = nominal
            mkt.register(aid, agent.initial_cash)
            agent_types[aid] = base_type
            if isinstance(agent, ScenarioController):
                controller = agent
            agents.append(agent)

    # ensure a controller exists if the scenario configures one
    if scenario.controller_config and controller is None:
        controller = build_agent("S1", seed=seed, overrides=scenario.controller_config)  # type: ignore
        mkt.register("S1", 0)
        agent_types["S1"] = "scenario_controller"
        agents.append(controller)
    if controller is not None:
        controller.config.update(scenario.controller_config)
        controller.bind(mkt)

    all_ticks: list[TickMetrics] = []
    message_bus: list[dict] = []
    for _t in range(scenario.n_ticks):
        is_crisis = mkt.tick >= CRISIS_TICK and scenario.crisis_enabled
        orders_by_agent = {}
        new_messages: list[dict] = []
        for agent in agents:
            view = agent.build_view(mkt, is_crisis, message_bus)
            if hasattr(agent, "observe_tick_metrics") and all_ticks:
                last = {t.symbol: t.signed_ofi for t in all_ticks[-len(mkt.books):]}
                agent.observe_tick_metrics(last)
            orders_by_agent[agent.agent_id] = agent.on_tick(view)
            new_messages.extend(agent.outbox)
        message_bus = new_messages
        all_ticks.extend(mkt.step(orders_by_agent))

    share_tag = composition_share if composition_share is not None else (adversary_share or 0.0)
    summary = summarize(all_ticks, mkt.accounts, agent_types,
                        baseline_peak_vol=baseline_peak_vol,
                        llm_mode=llm_mode,
                        adversary_share=share_tag,
                        final_marks=dict(mkt.last_price))

    for agent in agents:
        if getattr(agent, "config", {}).get("save_q_table") and hasattr(agent, "save_q_table"):
            agent.save_q_table(agent.config["q_table_path"])

    agent_final = {
        aid: {"cash": a.cash, "positions": dict(a.positions), "fees_paid": a.fees_paid,
              "trades_count": a.trades_count, "equity": a.equity(mkt.last_price)}
        for aid, a in mkt.accounts.items()}
    return RunResult(scenario.id, seed, all_ticks, summary, agent_final, llm_mode, agent_types)

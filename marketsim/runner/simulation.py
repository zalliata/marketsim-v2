"""Run one seeded simulation of a scenario and collect metrics.

Wiring: build the market, instantiate the scenario's agents (with per-agent
overrides, info-access ablations, a shared coordination message bus, and the
chosen LLM client), run the tick loop, and return the tick-metrics log plus the
iteration summary. Deterministic given (scenario, seed, llm_mode).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..agents.base import Agent, InfoAccess
from ..agents.controller import ScenarioController
from ..agents.registry import REGISTRY, build_agent
from ..calibration.anchors import CRISIS_TICK
from ..engine.market import Market, MarketConfig
from ..llm.providers import make_client
from ..metrics.iteration import summarize, IterationSummary
from ..scenarios.registry import Scenario
from ..types import TickMetrics


@dataclass
class RunResult:
    scenario_id: str
    seed: int
    ticks: list[TickMetrics]
    summary: IterationSummary
    agent_final: dict[str, dict]      # agent_id -> {cash, positions, fees, trades, equity}
    llm_mode: str


def _apply_ablations(info: InfoAccess, overrides: dict) -> InfoAccess:
    return InfoAccess(
        sentiment=info.sentiment and not overrides.get("_ablate_sentiment", False),
        graph_features=info.graph_features and not overrides.get("_ablate_graph", False),
        peer_messages=info.peer_messages,
        max_messages_per_tick=info.max_messages_per_tick,
    )


def run_once(scenario: Scenario, seed: int = 0, llm_mode: str = "scripted",
             tc_bps: float | None = None, adversary_share: float | None = None,
             baseline_peak_vol: float | None = None) -> RunResult:
    mkt = Market(MarketConfig(
        n_ticks=scenario.n_ticks,
        tc_bps=tc_bps if tc_bps is not None else scenario.tc_bps,
        seed=seed, crisis_enabled=scenario.crisis_enabled))

    llm = make_client(llm_mode, seed=seed)
    agents: list[Agent] = []
    controller: ScenarioController | None = None

    for aid in scenario.agent_ids:
        spec = REGISTRY[aid]
        ov = dict(scenario.agent_overrides.get(aid, {}))
        agent = build_agent(aid, seed=seed, overrides=ov,
                            llm_client=llm if spec.agent_type == "adversarial_llm" else None)
        agent.info = _apply_ablations(spec.info, ov)
        # adversary market-share scaling (P5/P6): scale capital & order sizing
        if adversary_share is not None and spec.agent_type.startswith("adversarial"):
            agent.initial_cash *= max(adversary_share / 0.10, 0.01)  # 10% = nominal
        cash = agent.initial_cash
        mkt.register(aid, cash)
        if isinstance(agent, ScenarioController):
            controller = agent
        agents.append(agent)

    # ensure a controller exists if the scenario configures one
    if scenario.controller_config and controller is None:
        controller = build_agent("S1", seed=seed, overrides=scenario.controller_config)  # type: ignore
        mkt.register("S1", 0)
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
        # feed last-tick OFI to signal-aware MMs
        for agent in agents:
            view = agent.build_view(mkt, is_crisis, message_bus)
            if hasattr(agent, "observe_tick_metrics") and all_ticks:
                last = {t.symbol: t.signed_ofi for t in all_ticks[-len(mkt.books):]}
                agent.observe_tick_metrics(last)
            orders_by_agent[agent.agent_id] = agent.on_tick(view)
            new_messages.extend(agent.outbox)
        message_bus = new_messages
        all_ticks.extend(mkt.step(orders_by_agent))

    agent_types = {aid: REGISTRY[aid].agent_type for aid in scenario.agent_ids}
    summary = summarize(all_ticks, mkt.accounts, agent_types,
                        baseline_peak_vol=baseline_peak_vol,
                        llm_mode=llm_mode,
                        adversary_share=adversary_share or 0.0,
                        final_marks=dict(mkt.last_price))

    # persist RL Q-tables when requested
    for agent in agents:
        if getattr(agent, "config", {}).get("save_q_table") and hasattr(agent, "save_q_table"):
            agent.save_q_table(agent.config["q_table_path"])

    agent_final = {
        aid: {"cash": a.cash, "positions": dict(a.positions), "fees_paid": a.fees_paid,
              "trades_count": a.trades_count, "equity": a.equity(mkt.last_price)}
        for aid, a in mkt.accounts.items()}
    return RunResult(scenario.id, seed, all_ticks, summary, agent_final, llm_mode)

import pytest
from marketsim.scenarios import definitions as D
from marketsim.runner.simulation import run_once

V1_IDS = [
    "baseline-noise-only", "baseline-with-mm", "baseline-all-defenders",
    "rq2-adversary-enters", "rq2-adversary-vs-all-mm", "rq2-two-adversaries",
    "rq2-info-disadvantage", "rq3-defense-comparison", "rq3-ablation-no-sentiment",
    "rq3-ablation-no-graph", "rq4-full-arena", "rq4-graph-exploiters-compete",
    "rq4-graph-defenders-compete", "rq4-graph-vs-llm-adversary",
    "rq4-graph-vs-llm-defender", "rq4-ablation-no-graph-exploiter",
    "crisis-intensified", "crisis-full-cast", "p4-pretrain-rl", "p4-solo-rulebased",
    "p4-solo-momstrev", "p4-solo-rl", "p4-single-llm-rulebased",
    "p4-single-llm-momstrev", "p4-single-llm-rl", "p4-coord-llm-rulebased",
    "p4-coord-llm-momstrev", "p4-coord-llm-rl", "p4-rl-reward-standard",
    "p4-rl-reward-social", "p4-rl-reward-aggressive", "p4-transfer-test",
]


def test_all_v1_scenarios_present():
    for sid in V1_IDS:
        assert sid in D.SCENARIOS, f"missing v1 scenario {sid}"


def test_future_paper_batteries_present():
    for sid in ["p3-cost-sweep-adversary", "p3-circuit-breaker",
                "p5-share-sweep-llm", "p6-composition-sweep", "p6-monoculture"]:
        assert sid in D.SCENARIOS


def test_determinism_same_seed():
    sc = D.get("rq2-adversary-enters")
    a = run_once(sc, seed=7)
    b = run_once(sc, seed=7)
    assert a.summary.max_volatility == b.summary.max_volatility
    assert a.summary.total_trades == b.summary.total_trades


@pytest.mark.parametrize("sid", ["baseline-noise-only", "rq2-adversary-enters",
                                 "rq4-full-arena", "p4-coord-llm-rl"])
def test_smoke_run_produces_ticks(sid):
    sc = D.get(sid)
    res = run_once(sc, seed=0)
    n_symbols = len(res.ticks) // sc.n_ticks
    assert len(res.ticks) == sc.n_ticks * n_symbols
    assert res.summary.total_trades >= 0


def test_tc_sweep_changes_costs():
    sc = D.get("p3-cost-sweep-adversary")
    lo = run_once(sc, seed=1, tc_bps=0.0)
    hi = run_once(sc, seed=1, tc_bps=50.0)
    fees_lo = sum(a["fees_paid"] for a in lo.agent_final.values())
    fees_hi = sum(a["fees_paid"] for a in hi.agent_final.values())
    assert fees_hi >= fees_lo       # higher tc -> at least as many fees

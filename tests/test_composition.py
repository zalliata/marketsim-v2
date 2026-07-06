"""P6 composition-scaling tests (RQ-F7).

Validates that composition_share genuinely varies the adversarial fraction of
the trading population and, crucially, of realized order flow — the property
the capital-share knob lacked (see Paper 5) and the reason P6 uses this axis.
"""
import pytest
from marketsim.scenarios import definitions as D
from marketsim.runner.simulation import run_once


def _adversary_trade_share(res):
    """Fraction of executed trades that came from adversary agents."""
    total = sum(f["trades_count"] for f in res.agent_final.values())
    adv = sum(f["trades_count"] for aid, f in res.agent_final.items()
              if res.agent_types.get(aid, "").startswith("adversarial"))
    return adv / total if total else 0.0


def test_population_counts_scale_with_share():
    sc = D.get("p6-composition-sweep")
    for share, exp_adv in [(0.0, 0), (0.05, 1), (0.25, 5), (0.50, 10)]:
        res = run_once(sc, seed=0, composition_share=share)
        n_adv = sum(1 for t in res.agent_types.values() if t.startswith("adversarial"))
        assert n_adv == exp_adv, f"share {share}: expected {exp_adv} adversaries, got {n_adv}"


def test_adversary_trade_share_rises_with_composition():
    sc = D.get("p6-composition-sweep")
    lo = _adversary_trade_share(run_once(sc, seed=1, composition_share=0.05))
    hi = _adversary_trade_share(run_once(sc, seed=1, composition_share=0.50))
    # the whole point: more adversaries in the population -> more adversarial flow
    assert hi > lo, f"adversary trade share did not rise: {lo:.3f} -> {hi:.3f}"


def test_zero_share_has_no_adversaries():
    sc = D.get("p6-composition-sweep")
    res = run_once(sc, seed=0, composition_share=0.0)
    assert _adversary_trade_share(res) == 0.0
    assert not any(t.startswith("adversarial") for t in res.agent_types.values())


def test_monoculture_is_single_strategy():
    res = run_once(D.get("p6-monoculture"), seed=0, composition_share=0.30)
    adv_types = {t for t in res.agent_types.values() if t.startswith("adversarial")}
    assert len(adv_types) == 1


def test_diversity_is_multi_strategy():
    res = run_once(D.get("p6-diversity"), seed=0, composition_share=0.40)
    adv_bases = {aid.split("#")[0] for aid, t in res.agent_types.items()
                 if t.startswith("adversarial")}
    assert len(adv_bases) >= 3   # diversity roster mixes several adversary strategies


def test_determinism_under_composition():
    sc = D.get("p6-composition-sweep")
    a = run_once(sc, seed=7, composition_share=0.30)
    b = run_once(sc, seed=7, composition_share=0.30)
    assert a.summary.max_volatility == b.summary.max_volatility
    assert a.summary.total_trades == b.summary.total_trades

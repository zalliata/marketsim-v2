"""Monte Carlo batch runner and parameter sweeps.

`run_batch` runs N seeded iterations of one scenario (optionally at a fixed
tc_bps / adversary_share) and writes every iteration to the storage backend,
returning aggregate statistics (Welch-ready means/stds, percentiles).

`run_sweep` walks a scenario's SweepAxis, calling run_batch per grid point;
this is how P3 (cost), P5 (share), and P6 (composition) batteries execute.
"""
from __future__ import annotations

import statistics
import uuid
from dataclasses import dataclass, field

from ..agents.registry import REGISTRY
from ..scenarios.registry import Scenario
from .simulation import run_once


@dataclass
class BatchStats:
    scenario_id: str
    n: int
    grid_key: str = ""
    grid_value: float = 0.0
    vol_mean: float = 0.0
    vol_std: float = 0.0
    adversary_pnl_mean: float = 0.0
    adversary_pnl_std: float = 0.0
    cascade_freq_mean: float = 0.0
    stabilisation_mean: float = 0.0


def _agg(scenario_id, results, grid_key="", grid_value=0.0) -> BatchStats:
    vol = [r.summary.max_volatility for r in results]
    apnl = [r.summary.adversary_pnl for r in results]
    casc = [r.summary.cascade_frequency for r in results]
    stab = [r.summary.stabilisation_effectiveness for r in results]
    sd = (lambda xs: statistics.pstdev(xs) if len(xs) > 1 else 0.0)
    mn = (lambda xs: sum(xs) / len(xs) if xs else 0.0)
    return BatchStats(scenario_id, len(results), grid_key, grid_value,
                      mn(vol), sd(vol), mn(apnl), sd(apnl), mn(casc), mn(stab))


def run_batch(scenario: Scenario, iterations: int, backend, base_seed: int = 0,
              llm_mode: str = "scripted", tc_bps=None, adversary_share=None,
              composition_share=None, run_id: str | None = None,
              baseline_peak_vol=None) -> BatchStats:
    run_id = run_id or f"{scenario.id}-{uuid.uuid4().hex[:8]}"
    label = scenario.labels.get("regime", "")
    results = []
    for i in range(iterations):
        res = run_once(scenario, seed=base_seed + i, llm_mode=llm_mode,
                       tc_bps=tc_bps, adversary_share=adversary_share,
                       composition_share=composition_share,
                       baseline_peak_vol=baseline_peak_vol)
        # res.agent_types covers replicated instance ids (composition sweeps)
        backend.write_run(run_id, i, res, res.agent_types, label=label)
        results.append(res)
    if tc_bps is not None:
        gk, gv = "tc_bps", tc_bps
    elif composition_share is not None:
        gk, gv = "composition_share", composition_share
    elif adversary_share is not None:
        gk, gv = "adversary_share", adversary_share
    else:
        gk, gv = "", 0.0
    return _agg(scenario.id, results, gk, gv or 0.0)


def run_sweep(scenario: Scenario, iterations: int, backend, base_seed: int = 0,
              llm_mode: str = "scripted") -> list[BatchStats]:
    if scenario.sweep is None:
        return [run_batch(scenario, iterations, backend, base_seed, llm_mode)]
    axis = scenario.sweep
    out: list[BatchStats] = []
    for j, val in enumerate(axis.values):
        kwargs = {}
        sc_run = scenario          # per-arm copy; never rebind the sweep base
        if axis.kind == "tc_bps":
            kwargs["tc_bps"] = val
        elif axis.kind == "adversary_share":
            kwargs["adversary_share"] = val
        elif axis.kind == "composition_share":
            kwargs["composition_share"] = val   # P6: adversarial population fraction
        elif axis.kind == "breaker":
            # arms: 0 = no intervention, 1 = FTT at tc*, 2 = circuit-breaker halt
            arm = int(val)
            if arm != 2:           # only the halt arm keeps the dynamic breaker
                sc_run = Scenario(**{**scenario.__dict__, "controller_config": {}})
            if arm == 1:           # FTT arm: same market, cost floor at tc*
                kwargs["tc_bps"] = float(scenario.labels.get("ftt_tc_bps",
                                                             scenario.tc_bps))
        rid = f"{scenario.id}-{axis.kind}{val}-{base_seed}"
        st = run_batch(sc_run, iterations, backend, base_seed, llm_mode,
                       run_id=rid, **kwargs)
        st.grid_key, st.grid_value = axis.kind, float(val)  # honest sweep labels
        out.append(st)
    return out

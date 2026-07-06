"""Paper 3 (RQ-F4) — locate the transaction-cost deterrence threshold tc*.

Two definitions, matched to the two adversary configs:

- A-config (p3-cost-sweep-adversary, vol-maximising A1-A3): **harm-based tc***
  = min tc where BOTH max-volatility and cascade-frequency are significantly
  below the tc=0 baseline (one-sided paired t-test, common random numbers:
  grid points share seeds, so tests pair by seed). PnL-based tests are also
  reported per adversary type for the robustness table, but vol-maximisers
  have ~zero expected PnL by design, so no PnL crossing is expected here.

- G-config (p3-cost-sweep-graph, profit-seeking G1-G3): **PnL-based tc***
  = min tc where mean adversary PnL < 0 (one-sided one-sample t-test).

Usage (from the repo root, after the sweep batteries have run):
    python analysis/p3_tc_star.py results [results_archive/* ...]

Pass every folder that holds sweep output (the housekeeping cell archives
old batches, so the coarse pass may live under results_archive/). Folders
from coarse + refinement passes are merged; duplicate tc values keep the
first occurrence. Writes p3_tc_star_summary.csv and p3_tc_star.png next to
this script.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from statistics import NormalDist, mean, stdev

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from marketsim.agents.registry import REGISTRY  # noqa: E402

ALPHA = 0.05
_norm = NormalDist()


def _one_sided_p(t: float) -> float:
    """P(T <= t) under H0, normal approximation (n=100 per point)."""
    return _norm.cdf(t)


def paired_less_p(baseline: list[float], treated: list[float]) -> float:
    """H1: treated < baseline. Pair by position (same seeds via CRN)."""
    d = [b - t for b, t in zip(baseline, treated)]
    s = stdev(d)
    if s == 0:
        return 0.5 if mean(d) == 0 else (0.0 if mean(d) > 0 else 1.0)
    return 1 - _one_sided_p(mean(d) / (s / math.sqrt(len(d))))


def mean_below_zero_p(xs: list[float]) -> float:
    """H1: E[x] < 0."""
    s = stdev(xs)
    if s == 0:
        return 0.5 if mean(xs) == 0 else (0.0 if mean(xs) < 0 else 1.0)
    return _one_sided_p(mean(xs) / (s / math.sqrt(len(xs))))


def load_sweep(roots: list[Path], scenario_id: str) -> dict[float, dict]:
    """tc -> {'iters': DataFrame, 'agents': DataFrame} from results folders."""
    pat = re.compile(re.escape(scenario_id) + r"-tc_bps([0-9.]+)-\d+$")
    out: dict[float, dict] = {}
    for root in roots:
        for d in sorted(Path(root).glob(f"{scenario_id}-tc_bps*")):
            m = pat.match(d.name)
            if not m or not (d / "scenario_iterations.csv").exists():
                continue
            tc = float(m.group(1))
            if tc in out:        # coarse + refined overlap: keep first
                continue
            out[tc] = {"iters": pd.read_csv(d / "scenario_iterations.csv"),
                       "agents": pd.read_csv(d / "agent_snapshots.csv")}
    return dict(sorted(out.items()))


def per_type_pnl(agents: pd.DataFrame, ids: list[str]) -> dict[str, list[float]]:
    res = {}
    for aid in ids:
        sub = agents[agents.agent_id == aid].sort_values("iteration")
        res[aid] = (sub.equity - REGISTRY[aid].cash).tolist()
    return res


def analyse(roots: list[Path]) -> pd.DataFrame:
    rows = []

    # ── A-config: harm-based ───────────────────────────────────────────────
    sweeps = load_sweep(roots, "p3-cost-sweep-adversary")
    if 0.0 not in sweeps:
        sys.exit("A-config: no tc_bps0 folder found — include the coarse-pass "
                 "results folder (e.g. results_archive/<stamp>) on the command line.")
    base = sweeps[0.0]["iters"]
    tc_star_harm = None
    for tc, d in sweeps.items():
        it = d["iters"]
        p_vol = p_casc = float("nan")
        if tc > 0:
            p_vol = paired_less_p(base.max_volatility.tolist(),
                                  it.max_volatility.tolist())
            p_casc = paired_less_p(base.cascade_frequency.tolist(),
                                   it.cascade_frequency.tolist())
            if tc_star_harm is None and p_vol < ALPHA and p_casc < ALPHA:
                tc_star_harm = tc
        row = {"config": "A", "tc_bps": tc, "n": len(it),
               "vol": it.max_volatility.mean(), "p_vol": p_vol,
               "cascade": it.cascade_frequency.mean(), "p_cascade": p_casc,
               "adv_pnl": it.adversary_pnl.mean()}
        for aid, pnl in per_type_pnl(d["agents"], ["A1", "A2", "A3"]).items():
            row[f"{aid}_pnl"] = mean(pnl)
            row[f"{aid}_p"] = mean_below_zero_p(pnl)
        rows.append(row)

    # ── G-config: PnL-based ────────────────────────────────────────────────
    tc_star_pnl = None
    for tc, d in load_sweep(roots, "p3-cost-sweep-graph").items():
        it = d["iters"]
        p = mean_below_zero_p(it.adversary_pnl.tolist())
        if tc_star_pnl is None and p < ALPHA:
            tc_star_pnl = tc
        rows.append({"config": "G", "tc_bps": tc, "n": len(it),
                     "vol": it.max_volatility.mean(),
                     "cascade": it.cascade_frequency.mean(),
                     "adv_pnl": it.adversary_pnl.mean(), "p_pnl_neg": p})

    df = pd.DataFrame(rows)
    df.attrs["tc_star_harm"] = tc_star_harm
    df.attrs["tc_star_pnl"] = tc_star_pnl
    return df


def figure(df: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = df[df.config == "A"].sort_values("tc_bps")
    g = df[df.config == "G"].sort_values("tc_bps")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.plot(a.tc_bps, a.vol, "o-", label="max volatility")
    sig = a[(a.p_vol < ALPHA) & (a.p_cascade < ALPHA)]
    ax.plot(sig.tc_bps, sig.vol, "o", mfc="none", ms=12,
            label=f"vol & cascade sig. (p<{ALPHA})")
    ts = df.attrs["tc_star_harm"]
    if ts is not None:
        ax.axvline(ts, ls="--", c="grey", label=f"tc* (harm) = {ts:g} bps")
    ax.set(xlabel="transaction cost (bps)", ylabel="mean max volatility",
           title="A-config: volatility vs tc")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(a.tc_bps, a.cascade, "s-", color="tab:orange")
    if ts is not None:
        ax.axvline(ts, ls="--", c="grey")
    ax.set(xlabel="transaction cost (bps)", ylabel="mean cascade frequency",
           title="A-config: cascades vs tc")

    ax = axes[2]
    ax.plot(g.tc_bps, g.adv_pnl, "d-", color="tab:green", label="G-config")
    ax.axhline(0, c="k", lw=0.8)
    tsp = df.attrs["tc_star_pnl"]
    ax.set(xlabel="transaction cost (bps)", ylabel="mean adversary PnL",
           title="G-config: exploiter PnL vs tc"
                 + ("" if tsp else " (no crossing in grid)"))
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"figure -> {path}")


def main() -> None:
    roots = [Path(p) for p in (sys.argv[1:] or ["results"])]
    df = analyse(roots)
    here = Path(__file__).resolve().parent
    out_csv = here / "p3_tc_star_summary.csv"
    df.to_csv(out_csv, index=False)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(f"\nsummary -> {out_csv}")
    print(f"tc* (A-config, harm-based) = {df.attrs['tc_star_harm']} bps")
    print(f"tc* (G-config, PnL-based)  = {df.attrs['tc_star_pnl']} "
          f"{'bps' if df.attrs['tc_star_pnl'] else '(not reached in grid)'}")
    figure(df, here / "p3_tc_star.png")


if __name__ == "__main__":
    main()

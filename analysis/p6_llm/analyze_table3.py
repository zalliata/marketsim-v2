#!/usr/bin/env python3
"""P6 Section 7 — fill Table 3 (scripted vs genuine-LLM adversary).

Reads the run_p6_llm.py outputs from results/p6-llm-* and prints, per tc level:
    peak realised volatility | adversarial order-flow share | adversary PnL |
    cascade frequency        — scripted vs LLM, ratio, and paired p-values.

Paired inference over common seeds (same protocol as the scripted batteries);
adv order-flow share computed from agent_snapshots exactly as p6_analyze.py.
Also summarises the mini-sweep and picks which pre-drafted interpretation
paragraph ("contained" vs "more damaging") the data supports.

Usage:  python analysis/p6_llm/analyze_table3.py [--provider anthropic]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results"


def load_cell(tag):
    d = RES / f"p6-llm-{tag}"
    if not (d / "scenario_iterations.csv").exists():
        return None
    si = pd.read_csv(d / "scenario_iterations.csv")
    ag = pd.read_csv(d / "agent_snapshots.csv")
    ag["is_adv"] = ag["agent_type"].astype(str).str.startswith("adversarial")
    tot = ag.groupby("iteration")["trades_count"].sum()
    adv = ag[ag.is_adv].groupby("iteration")["trades_count"].sum()
    flow = (adv / tot).reindex(tot.index).fillna(0.0)
    si = si.sort_values("iteration").reset_index(drop=True)
    si["adv_flow_share"] = si["iteration"].map(flow).fillna(0.0)
    return si


def paired_p(a, b):
    """One-sided paired t-test p-value (H1: mean(b-a) != 0, reported two-sided)."""
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    if n < 2:
        return float("nan")
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1)
    if var == 0:
        return 1.0
    t = m / math.sqrt(var / n)
    # normal approximation is fine at n>=20; report two-sided
    from statistics import NormalDist
    return 2 * (1 - NormalDist().cdf(abs(t)))


METRICS = [
    ("Peak realised volatility", "max_volatility"),
    ("Adversarial order-flow share", "adv_flow_share"),
    ("Adversary PnL", "adversary_pnl"),
    ("Cascade frequency", "cascade_frequency"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="anthropic")
    ap.add_argument("--tc-levels", default="10,20,30")
    args = ap.parse_args()

    contained_votes = []
    for tc in [int(x) for x in args.tc_levels.split(",")]:
        s = load_cell(f"scripted-tc{tc}")
        l = load_cell(f"{args.provider}-tc{tc}")
        if s is None or l is None:
            print(f"tc={tc}: missing run(s) — skipped"); continue
        n = min(len(s), len(l))
        s, l = s.iloc[:n], l.iloc[:n]
        print(f"\n=== Table 3 @ tc = {tc} bps (n = {n} paired seeds) ===")
        print(f"{'Metric':<32s}{'Scripted':>12s}{'LLM':>12s}{'Ratio':>9s}{'p':>10s}")
        for label, col in METRICS:
            a, b = s[col].tolist(), l[col].tolist()
            ma, mb = sum(a) / n, sum(b) / n
            ratio = (mb / ma) if ma not in (0, 0.0) else float("inf") if mb else 1.0
            p = paired_p(a, b)
            print(f"{label:<32s}{ma:>12.4f}{mb:>12.4f}{ratio:>9.2f}{p:>10.3f}")
            if col in ("max_volatility", "adv_flow_share"):
                contained_votes.append(mb <= ma * 1.25 or p > 0.05)

    print("\n=== Mini-sweep (appendix): LLM adversary at rising population share ===")
    print(f"{'share':>6s}{'scripted vol':>14s}{'LLM vol':>10s}{'scripted flow':>15s}{'LLM flow':>10s}")
    for share in (10, 20, 30, 40):
        s = load_cell(f"scripted-share{share}")
        l = load_cell(f"{args.provider}-share{share}")
        if s is None or l is None:
            continue
        print(f"{share:>5d}%{s['max_volatility'].mean():>14.4f}{l['max_volatility'].mean():>10.4f}"
              f"{s['adv_flow_share'].mean():>15.3f}{l['adv_flow_share'].mean():>10.3f}")

    if contained_votes:
        verdict = "CONTAINED" if all(contained_votes) else "NOT (fully) CONTAINED"
        print(f"\nVerdict for Section 7 interpretation paragraph: {verdict}")
        print("  contained      -> keep the 'fee gate is a property of the market' paragraph")
        print("  not contained  -> keep the 'floor must target the most capable adversary' paragraph,")
        print("                    and note the tc level (20/30 bps cells) at which containment returns")


if __name__ == "__main__":
    main()

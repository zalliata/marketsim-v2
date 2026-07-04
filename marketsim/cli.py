"""marketsim command-line interface.

    marketsim list [--paper P3]
    marketsim run   <scenario_id> [--seed 0] [--llm scripted] [--tc-bps 10]
    marketsim batch <scenario_id> --iterations 100 [--seed 0] [--storage local]
    marketsim sweep <scenario_id> --iterations 100
    marketsim show  <scenario_id>

Storage defaults to 'auto' (Supabase if SUPABASE_URL/SUPABASE_SERVICE_KEY are
set, else local CSV under ./results/).
"""
from __future__ import annotations

import argparse
import sys

from .scenarios import definitions as D
from .storage import make_backend


def _cmd_list(args):
    rows = []
    for sid, sc in D.SCENARIOS.items():
        if args.paper and args.paper.lower() not in sc.paper.lower():
            continue
        rows.append((sid, sc.paper, "sweep" if sc.sweep else "", sc.name))
    w = max((len(r[0]) for r in rows), default=10)
    print(f"{len(rows)} scenarios:")
    for sid, paper, sweep, name in rows:
        print(f"  {sid:<{w}}  {paper:<10} {sweep:<6} {name}")


def _cmd_show(args):
    sc = D.get(args.scenario_id)
    print(f"{sc.id}  ({sc.paper})\n  {sc.name}\n  {sc.description}")
    print(f"  agents: {', '.join(sc.agent_ids)}")
    print(f"  ticks: {sc.n_ticks}   tc_bps: {sc.tc_bps}   crisis: {sc.crisis_enabled}")
    if sc.sweep:
        print(f"  sweep: {sc.sweep.kind} over {sc.sweep.values}")
    if sc.labels:
        print(f"  labels: {sc.labels}")


def _cmd_run(args):
    from .runner.simulation import run_once
    sc = D.get(args.scenario_id)
    res = run_once(sc, seed=args.seed, llm_mode=args.llm,
                   tc_bps=args.tc_bps)
    s = res.summary
    print(f"[{sc.id}] seed={args.seed} llm={res.llm_mode}")
    print(f"  max_vol={s.max_volatility:.4f}  final_price={s.final_price:.2f}  "
          f"drawdown={s.price_drawdown:.3f}")
    print(f"  adversary_pnl={s.adversary_pnl:,.0f}  defense_pnl={s.defense_pnl:,.0f}")
    print(f"  trades={s.total_trades}  volume={s.total_volume}  "
          f"cascade_freq={s.cascade_frequency:.3f}  halts={s.halts}")


def _cmd_batch(args):
    from .runner.batch import run_batch
    sc = D.get(args.scenario_id)
    backend = make_backend(args.storage)
    try:
        stats = run_batch(sc, args.iterations, backend, base_seed=args.seed, llm_mode=args.llm)
    finally:
        backend.close()
    print(f"[{sc.id}] n={stats.n}  vol_mean={stats.vol_mean:.4f}±{stats.vol_std:.4f}  "
          f"adv_pnl_mean={stats.adversary_pnl_mean:,.0f}  "
          f"cascade_freq={stats.cascade_freq_mean:.3f}")


def _cmd_sweep(args):
    from .runner.batch import run_sweep
    sc = D.get(args.scenario_id)
    backend = make_backend(args.storage)
    try:
        stats = run_sweep(sc, args.iterations, backend, base_seed=args.seed, llm_mode=args.llm)
    finally:
        backend.close()
    print(f"[{sc.id}] sweep '{sc.sweep.kind if sc.sweep else '-'}' — {len(stats)} grid points:")
    for st in stats:
        print(f"  {st.grid_key}={st.grid_value:<8}  vol={st.vol_mean:.4f}  "
              f"adv_pnl={st.adversary_pnl_mean:,.0f}  cascade={st.cascade_freq_mean:.3f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("marketsim", description="Crisis-calibrated market simulator (v2)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list", help="list scenarios")
    pl.add_argument("--paper", default="")
    pl.set_defaults(func=_cmd_list)

    ps = sub.add_parser("show", help="show one scenario")
    ps.add_argument("scenario_id")
    ps.set_defaults(func=_cmd_show)

    common_llm = dict(default="scripted",
                      choices=["scripted", "anthropic", "openai", "gemini"])
    pr = sub.add_parser("run", help="run one iteration")
    pr.add_argument("scenario_id")
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--llm", **common_llm)
    pr.add_argument("--tc-bps", type=float, default=None)
    pr.set_defaults(func=_cmd_run)

    pb = sub.add_parser("batch", help="Monte Carlo batch")
    pb.add_argument("scenario_id")
    pb.add_argument("--iterations", type=int, default=100)
    pb.add_argument("--seed", type=int, default=0)
    pb.add_argument("--llm", **common_llm)
    pb.add_argument("--storage", default="auto", choices=["auto", "local", "supabase"])
    pb.set_defaults(func=_cmd_batch)

    pw = sub.add_parser("sweep", help="parameter sweep (P3/P5/P6)")
    pw.add_argument("scenario_id")
    pw.add_argument("--iterations", type=int, default=100)
    pw.add_argument("--seed", type=int, default=0)
    pw.add_argument("--llm", **common_llm)
    pw.add_argument("--storage", default="auto", choices=["auto", "local", "supabase"])
    pw.set_defaults(func=_cmd_sweep)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

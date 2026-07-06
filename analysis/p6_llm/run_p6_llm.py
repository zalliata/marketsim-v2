#!/usr/bin/env python3
"""P6 Section 7 — genuine-LLM adversary battery.

Fills the placeholders in "Robustness: A Genuine-LLM Adversary":
  1. HEAD-TO-HEAD  scripted vs genuine-LLM adversary, fully defended arena,
     single adversary (composition_share = 0.05 of a 20-trader population),
     paired seeds, at tc = 10 bps (programme default) — plus 20 and 30 bps,
     which simultaneously delivers P3's promised small-batch tc* replication.
  2. MINI-SWEEP    LLM adversary at composition shares 10-40%, tc = 10 bps
     (online-appendix figure).

Reproducibility: temperature 0 (default), model pinned via LLM_MODEL, every
prompt/decision cached to disk (LLM_CACHE) so the battery is replayable and
re-runs are free. Every adversarial decision (with the model's rationale) is
also logged to a JSONL for the paper's qualitative analysis.

Usage (from the marketsim-v2 repo root):
    set ANTHROPIC_API_KEY=...          # or export on mac/linux
    python analysis/p6_llm/run_p6_llm.py --provider anthropic
    python analysis/p6_llm/run_p6_llm.py --provider anthropic --iterations 20
    python analysis/p6_llm/run_p6_llm.py --scripted-only    # free dry run

Cost guard: LLM_MAX_CALLS defaults to 120000 here; at ~39 ticks x 1 adversary
x 20 iterations x 3 tc levels + sweep, expect ~5-8k real calls (far below cap).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = REPO / "results"
KIT = REPO / "analysis" / "p6_llm"

from marketsim.scenarios import definitions as D           # noqa: E402
from marketsim.runner.batch import run_batch                # noqa: E402
from marketsim.storage.local_backend import LocalCSVBackend  # noqa: E402
from marketsim.llm import providers                         # noqa: E402

SCENARIO = "p6-composition-sweep"     # fully defended arena (Z1 Z2 A1 A2 G1 + P2 P4 D1)
HEAD_TO_HEAD_SHARE = 0.05             # 1 adversary in a 20-trader population
TC_LEVELS = [10.0, 20.0, 30.0]        # 10 = programme default; 20/30 = P3 tc* replication
SWEEP_SHARES = [0.10, 0.20, 0.30, 0.40]
DECISION_LOG = KIT / "llm_decisions.jsonl"


def _install_decision_logger():
    """Wrap make_client so every real-LLM decision (incl. rationale) is logged
    and the on-disk cache is flushed after construction-site usage."""
    orig = providers.make_client
    log_f = open(DECISION_LOG, "a", encoding="utf-8")

    def wrapped(llm_mode, seed=0, model=None):
        client = orig(llm_mode, seed=seed, model=model)
        if llm_mode == "scripted":
            return client
        inner_decide = client.decide

        def decide(system_prompt, state):
            d = inner_decide(system_prompt, state)
            log_f.write(json.dumps({"t": time.time(), "seed": seed,
                                    "state": state, "decision": d}) + "\n")
            log_f.flush()
            return d

        client.decide = decide
        wrapped.clients.append(client)
        return client

    wrapped.clients = []
    providers.make_client = wrapped
    # run_once imported make_client by name — patch its reference too
    from marketsim.runner import simulation
    simulation.make_client = wrapped
    return wrapped


def run_cell(tag, llm_mode, tc_bps=None, share=HEAD_TO_HEAD_SHARE,
             iterations=20, base_seed=0):
    sc = D.get(SCENARIO)
    run_id = f"p6-llm-{tag}"
    backend = LocalCSVBackend(root=str(OUT))
    t0 = time.time()
    stats = run_batch(sc, iterations, backend, base_seed=base_seed,
                      llm_mode=llm_mode, tc_bps=tc_bps,
                      composition_share=share, run_id=run_id)
    backend.close()
    print(f"  {run_id:<44s} {llm_mode:<10s} tc={tc_bps} share={share} "
          f"n={iterations}  [{time.time()-t0:.1f}s]")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="anthropic",
                    choices=["anthropic", "openai", "gemini"])
    ap.add_argument("--iterations", type=int, default=20,
                    help="per cell; matches the scripted P6 batteries")
    ap.add_argument("--sweep-iterations", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scripted-only", action="store_true",
                    help="run only the free scripted arm (pipeline dry run)")
    args = ap.parse_args()

    os.environ.setdefault("LLM_TEMPERATURE", "0.0")
    os.environ.setdefault("LLM_MAX_TOKENS", "300")
    os.environ.setdefault("LLM_MAX_CALLS", "120000")
    os.environ.setdefault("LLM_CACHE", str(KIT / f"llm_cache_{args.provider}.json"))

    key_var = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY",
               "gemini": "GEMINI_API_KEY"}[args.provider]
    if not args.scripted_only and not os.environ.get(key_var):
        sys.exit(f"error: {key_var} not set (or use --scripted-only)")

    wrapped = _install_decision_logger()
    print(f"provider={args.provider}  temperature={os.environ['LLM_TEMPERATURE']}  "
          f"cache={os.environ['LLM_CACHE']}")

    print("\n[1/2] Head-to-head: single adversary, defended arena, paired seeds")
    for tc in TC_LEVELS:
        run_cell(f"scripted-tc{int(tc)}", "scripted", tc_bps=tc,
                 iterations=args.iterations, base_seed=args.seed)
        if not args.scripted_only:
            run_cell(f"{args.provider}-tc{int(tc)}", args.provider, tc_bps=tc,
                     iterations=args.iterations, base_seed=args.seed)

    print("\n[2/2] Composition mini-sweep (appendix), tc = 10 bps")
    for share in SWEEP_SHARES:
        run_cell(f"scripted-share{int(share*100)}", "scripted", tc_bps=10.0,
                 share=share, iterations=args.sweep_iterations, base_seed=args.seed)
        if not args.scripted_only:
            run_cell(f"{args.provider}-share{int(share*100)}", args.provider,
                     tc_bps=10.0, share=share,
                     iterations=args.sweep_iterations, base_seed=args.seed)

    # flush decision caches and report API usage
    calls = errors = 0
    for c in wrapped.clients:
        if hasattr(c, "flush"):
            c.flush()
        calls += getattr(c, "call_count", 0)
        errors += getattr(c, "error_count", 0)
    print(f"\ndone. real API calls: {calls}  errors→hold: {errors}")
    print(f"decisions+rationales: {DECISION_LOG}")
    print("next: python analysis/p6_llm/analyze_table3.py")


if __name__ == "__main__":
    main()

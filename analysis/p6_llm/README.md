# P6 genuine-LLM battery — run kit

Fills P6 Section 7 (Table 3 + appendix mini-sweep) and doubles as P3's promised
small-batch genuine-LLM replication of tc*.

## Before you run
- The repo's `.py` files are OneDrive cloud placeholders on some machines; make
  sure the folder is marked **"Always keep on this device"** first. A truncated
  `providers.py` was already restored from git (July 6).
- `pip install pandas` if not present (analysis only; the simulator itself is
  stdlib-only).

## Run (repo root)
```bash
export ANTHROPIC_API_KEY=sk-...        # Windows: set ANTHROPIC_API_KEY=...
python analysis/p6_llm/run_p6_llm.py --scripted-only   # free pipeline check
python analysis/p6_llm/run_p6_llm.py --provider anthropic
python analysis/p6_llm/analyze_table3.py
```

What it runs:
1. **Head-to-head** — `p6-composition-sweep` scenario, fully defended arena,
   single adversary (5% of the 20-trader population), 20 paired seeds,
   scripted vs LLM, at tc ∈ {10, 20, 30} bps.
2. **Mini-sweep** — LLM adversary at 10/20/30/40% population share, tc = 10 bps,
   10 iterations each (appendix figure).

Reproducibility: temperature 0; pin the model with `LLM_MODEL` (default
`claude-haiku-4-5-20251001` — consider the strongest model you can afford, since
Section 7's argument is about adversary capability); every prompt/decision is
cached in `analysis/p6_llm/llm_cache_<provider>.json` (re-runs are free and
deterministic) and every decision + rationale is appended to
`analysis/p6_llm/llm_decisions.jsonl` for the qualitative analysis.

Cost: ~39 crisis ticks × 1 adversary × 20 seeds × 3 tc levels ≈ 2,400 calls,
plus the sweep (≈ 8 × 10 × 39 × avg 5 adversaries ≈ capped by cache) — expect
mid-single-digit dollars on a small model. `LLM_MAX_CALLS=120000` is a hard stop.

## Filling the paper
`analyze_table3.py` prints Table 3 rows (scripted, LLM, ratio, paired p) per tc
level, the mini-sweep summary, and a verdict line telling you which of the two
pre-drafted interpretation paragraphs in Section 7 to keep. Then:
1. Fill Table 3 in `P6_..._v2.docx` §7 with the tc = 10 row values.
2. Delete the unsupported interpretation paragraph; keep the supported one.
3. Quote 2–3 rationales from `llm_decisions.jsonl` (hub identification, crisis
   timing, cost reasoning) as the paper promises.
4. Update thesis Ch. 7.6 ("Robustness in Progress" → completed result) and, if
   the verdict is NOT contained, revisit Instrument 1's calibration language.
5. Record model string, date, and cache hash in the paper's reproducibility note.

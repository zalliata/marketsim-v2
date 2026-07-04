# marketsim-v2

A crisis-calibrated multi-agent market simulator for the dissertation programme
*"Algorithmic Trading, Financial Contagion, and the Architecture of Regulatory
Resilience"* (Zorina Alliata, ASE Bucharest).

This is the clean-room rewrite of the original Lovable/React/Supabase prototype
(`../Original Version/`). It is a single-runtime Python package with one order-book
engine, one agent framework, one scenario system, and pluggable storage. It runs
every scenario from Papers 0–2 and the sweep-based batteries needed for Papers
3, 5, and 6. See `ARCHITECTURE.md` for the full design and `docs/` for details.

## Why the rewrite
The original had five structural problems, chief among them that its
"LLM adversaries" never called an LLM in the batch path — they were hard-coded
heuristics. See `docs/MIGRATION_NOTES.md` for the complete list and how each is
resolved here.

## Install
```bash
cd marketsim-v2
pip install -e .            # core, standard-library only
pip install -e ".[supabase]"  # optional: Supabase storage backend
pip install -e ".[dev]"       # optional: pytest
```

## Quick start
```bash
marketsim list                       # all scenarios (v1 + P3/P5/P6 batteries)
marketsim list --paper P3            # just the P3 battery
marketsim show rq2-adversary-enters  # scenario detail
marketsim run rq2-adversary-enters --seed 0          # one iteration, printed
marketsim batch rq2-adversary-enters --iterations 100  # Monte Carlo -> storage
marketsim sweep p3-cost-sweep-adversary --iterations 100   # tc 0-50 bps sweep
```

## LLM modes
Adversarial agents run through a pluggable `LLMClient`:
```bash
marketsim run rq2-adversary-enters --llm scripted   # deterministic (default, free)
marketsim run rq2-adversary-enters --llm anthropic  # real API (ANTHROPIC_API_KEY)
marketsim run rq2-adversary-enters --llm openai     # (OPENAI_API_KEY)
marketsim run rq2-adversary-enters --llm gemini     # (GEMINI_API_KEY)
```
Every output row records which mode produced it (`llm_mode` column), so genuine-LLM
and scripted-policy results are always distinguishable.

## Storage
Defaults to `auto`: Supabase if `SUPABASE_URL` + `SUPABASE_SERVICE_KEY` are set,
otherwise local CSV under `./results/<run_id>/`. Tables match the v1 export schema
(plus additive microstructure columns), so existing P2–P4 analysis scripts keep
working. Force with `--storage local` or `--storage supabase`.

## Reproducibility
Every run is seeded; batch iteration *i* uses `seed + i`; scripted-LLM and RL
exploration draw only from the seeded generator. Same seed → identical output.

## Running scenarios in the cloud (Google Colab)
This is a batch simulation package, **not** a web app — it does not deploy to Vercel/Netlify
(those run code for seconds; the sweeps run for minutes to hours). To run the P3/P5/P6
batteries without tying up your own machine, open `notebooks/run_on_colab.ipynb` in
[Google Colab](https://colab.research.google.com) (File → Open notebook → GitHub → paste the
repo URL). It clones the repo, installs the package, runs any scenario or sweep, and saves the
result CSVs to your Google Drive or Supabase. No server setup, handles long jobs.

## Tests
```bash
pytest        # order book, signals, Q-learning, scenario registry, determinism, smoke runs
```

## Layout
```
marketsim/
  engine/       order book + multi-asset market (tick loop, fees, halts, metrics)
  calibration/  10-asset SVB cohort: network, anchor prices, sentiment process
  agents/       ZI, MOM/STREV, market makers, RL MM, adversarial LLM, graph agents, controller
  llm/          LLMClient (scripted + Anthropic/OpenAI/Gemini), prompts
  scenarios/    scenario dataclass + all definitions (v1 + P3/P5/P6)
  runner/       one run, Monte Carlo batch, parameter sweeps
  metrics/      microstructure features, systemic-risk / phase-transition, iteration summary
  storage/      local CSV + Supabase backends
  cli.py        command-line entry point
```

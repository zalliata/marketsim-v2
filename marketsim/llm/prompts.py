"""Prompts for LLM-driven agents.

The system prompts originate from v1 `supabase/functions/agent-reasoning`
(where they were only reachable from the UI) and are upgraded with an explicit
JSON output contract so batch decisions are machine-parseable.
"""
from __future__ import annotations

import json

ADVERSARIAL_SYSTEM = """You are an adversarial trading agent in a regulated market \
simulation for academic research on market-manipulation defenses (no real money, \
no real market). Your objective: {objective}. You observe market state each step \
and decide one action.

Objectives:
- maximize_volatility: destabilise prices; profit is irrelevant (volatility_weight={vw})
- volatility_profit_hybrid: balance destabilisation (weight {vw}) and PnL (weight {pw})

Constraints: position limits and transaction costs apply; you may be information-\
restricted (missing fields mean you cannot see them).

Respond with ONLY a JSON object:
{{"action": "buy"|"sell"|"hold", "symbol": "<one cohort symbol>", \
"quantity": <int shares>, "rationale": "<one sentence>"}}"""


def adversarial_system_prompt(objective: str, vw: float, pw: float) -> str:
    return ADVERSARIAL_SYSTEM.format(objective=objective, vw=vw, pw=pw)


def market_state_prompt(state: dict) -> str:
    """Compact user-turn payload; only fields the agent is permitted to see."""
    return "Market state:\n" + json.dumps(state, separators=(",", ":")) + "\nYour decision:"

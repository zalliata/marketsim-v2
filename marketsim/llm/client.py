"""LLM client abstraction for adversarial agents.

`LLMClient.decide(system_prompt, state)` must return a dict:
    {"action": "buy"|"sell"|"hold", "symbol": str, "quantity": int, "rationale": str}

Backends:
- ScriptedClient  — deterministic, seeded policy encoding the adversarial
  strategy described in P1 §3 (contrarian volatility amplification with
  crisis exploitation). Zero cost; the default for Monte Carlo batteries.
  Faithfully reproduces the *intent* of v1's heuristics, but through the same
  interface as a real LLM so results are comparable mode-to-mode.
- Provider clients (providers.py) — real API calls; selected with
  llm_mode="anthropic"|"openai"|"gemini". Malformed/failed responses degrade
  to hold and are counted, never raised (a batch must not die at iteration 87).

Every simulation records which mode produced its decisions (`llm_mode` column).
"""
from __future__ import annotations

import json
import random
from typing import Protocol


class LLMClient(Protocol):
    mode: str

    def decide(self, system_prompt: str, state: dict) -> dict: ...


class ScriptedClient:
    """Deterministic adversarial policy, seeded.

    Policy (mirrors the documented P1 adversary):
    - crisis + very negative sentiment  → buy the panic (amplify the rebound leg)
    - pre-crisis negative drift + long  → sell into weakness (accelerate decline)
    - strong positive sentiment         → contrarian sell
    - otherwise                         → hold (participation ~35%/tick, seeded)
    Objectives modulate sizing: volatility maximisers trade the vol-weighted
    size; hybrids shade size by profit weight.
    """
    mode = "scripted"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed ^ 0xADE5)

    def decide(self, system_prompt: str, state: dict) -> dict:
        d = self._policy(state)
        # P3 cost gate — applied AFTER the policy (all rng draws already made,
        # so the random stream is identical at every tc_bps grid point).
        # Strict '>' keeps tc=0 behaviour bit-identical to the fee-blind runs.
        cost = float(state.get("round_trip_cost_bps", 0.0))
        edge = float(state.get("expected_edge_bps", 0.0))
        if d.get("action") != "hold" and cost > edge:
            return {"action": "hold", "symbol": d.get("symbol"), "quantity": 0,
                    "rationale": f"scripted: round-trip cost {cost:.1f}bps "
                                 f"exceeds expected edge {edge:.1f}bps"}
        return d

    def _policy(self, state: dict) -> dict:
        s = state
        sym = s.get("target_symbol", "SIVB")
        sent = float(s.get("sentiment", 0.0))
        crisis = bool(s.get("is_crisis", False))
        position = int(s.get("position", 0))
        base = int(s.get("base_size", 100))
        vw = float(s.get("volatility_weight", 1.0))
        pw = float(s.get("profit_weight", 0.0))
        size = max(1, int(base * (0.6 + 0.4 * vw) * (1.0 - 0.3 * pw)))

        if self.rng.random() > 0.35:
            return {"action": "hold", "symbol": sym, "quantity": 0,
                    "rationale": "scripted: waiting"}
        if crisis and sent < -0.5:
            return {"action": "buy", "symbol": sym, "quantity": int(size * 1.5),
                    "rationale": "scripted: buy panic to amplify volatility"}
        if sent < -0.2 and position > 0:
            return {"action": "sell", "symbol": sym, "quantity": max(1, int(position * 0.3)),
                    "rationale": "scripted: sell into weakness to accelerate decline"}
        if sent > 0.2:
            return {"action": "sell", "symbol": sym, "quantity": size,
                    "rationale": "scripted: contrarian sell during optimism"}
        if crisis:
            act = "buy" if self.rng.random() > 0.5 else "sell"
            return {"action": act, "symbol": sym, "quantity": int(size * 1.5),
                    "rationale": "scripted: crisis exploitation"}
        return {"action": "hold", "symbol": sym, "quantity": 0,
                "rationale": "scripted: no setup"}


def parse_llm_json(text: str) -> dict | None:
    """Extract the first JSON object from a model response; None if impossible."""
    try:
        start = text.index("{")
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:i + 1])
    except (ValueError, json.JSONDecodeError):
        return None
    return None

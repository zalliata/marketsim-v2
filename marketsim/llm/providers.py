"""Real LLM provider clients. API keys come from environment variables only.

All clients share the same contract as ScriptedClient. Failures (network,
rate-limit, malformed output) return a hold decision and increment
`error_count` — a Monte Carlo batch must never crash on a provider hiccup.

Cost control: each client counts calls; the batch runner enforces --llm-budget.
"""
from __future__ import annotations

import json
import os
import urllib.request

from .client import parse_llm_json

_HOLD = {"action": "hold", "symbol": "SIVB", "quantity": 0, "rationale": "provider error"}


class _HttpClient:
    mode = "base"
    url = ""

    def __init__(self, model: str):
        self.model = model
        self.call_count = 0
        self.error_count = 0

    def _post(self, payload: dict, headers: dict) -> dict | None:
        req = urllib.request.Request(self.url, method="POST",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json", **headers})
        with urllib.request.urlopen(req, timeout=45) as r:
            return json.loads(r.read().decode())

    def decide(self, system_prompt: str, state: dict) -> dict:
        from .prompts import market_state_prompt
        self.call_count += 1
        try:
            text = self._complete(system_prompt, market_state_prompt(state))
            out = parse_llm_json(text or "")
            if out and out.get("action") in ("buy", "sell", "hold"):
                out.setdefault("symbol", state.get("target_symbol", "SIVB"))
                out["quantity"] = int(out.get("quantity", 0) or 0)
                return out
        except Exception:
            pass
        self.error_count += 1
        return dict(_HOLD, symbol=state.get("target_symbol", "SIVB"))

    def _complete(self, system: str, user: str) -> str:
        raise NotImplementedError


class AnthropicClient(_HttpClient):
    mode = "anthropic"
    url = "https://api.anthropic.com/v1/messages"

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        super().__init__(model)
        self.key = os.environ.get("ANTHROPIC_API_KEY", "")

    def _complete(self, system: str, user: str) -> str:
        resp = self._post(
            {"model": self.model, "max_tokens": 200, "system": system,
             "messages": [{"role": "user", "content": user}]},
            {"x-api-key": self.key, "anthropic-version": "2023-06-01"})
        return resp["content"][0]["text"]


class OpenAIClient(_HttpClient):
    mode = "openai"
    url = "https://api.openai.com/v1/chat/completions"

    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__(model)
        self.key = os.environ.get("OPENAI_API_KEY", "")

    def _complete(self, system: str, user: str) -> str:
        resp = self._post(
            {"model": self.model, "max_tokens": 200,
             "messages": [{"role": "system", "content": system},
                          {"role": "user", "content": user}]},
            {"Authorization": f"Bearer {self.key}"})
        return resp["choices"][0]["message"]["content"]


class GeminiClient(_HttpClient):
    mode = "gemini"

    def __init__(self, model: str = "gemini-2.5-flash"):
        super().__init__(model)
        self.key = os.environ.get("GEMINI_API_KEY", "")
        self.url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{model}:generateContent?key={self.key}")

    def _complete(self, system: str, user: str) -> str:
        resp = self._post(
            {"systemInstruction": {"parts": [{"text": system}]},
             "contents": [{"parts": [{"text": user}]}]},
            {})
        return resp["candidates"][0]["content"]["parts"][0]["text"]


def make_client(llm_mode: str, seed: int = 0, model: str | None = None):
    """Factory: 'scripted' (default) or a provider name."""
    from .client import ScriptedClient
    if llm_mode == "scripted":
        return ScriptedClient(seed)
    if llm_mode == "anthropic":
        return AnthropicClient(model or "claude-haiku-4-5-20251001")
    if llm_mode == "openai":
        return OpenAIClient(model or "gpt-4o-mini")
    if llm_mode == "gemini":
        return GeminiClient(model or "gemini-2.5-flash")
    raise ValueError(f"unknown llm_mode: {llm_mode}")

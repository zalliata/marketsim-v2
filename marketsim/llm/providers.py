"""Real LLM provider clients for genuine-LLM adversary runs.

API keys come from environment variables only. All clients share the same
contract as ScriptedClient (``decide(system_prompt, state) -> decision dict``),
so any adversarial-LLM agent runs unchanged against a real model when the
``--llm anthropic|openai|gemini`` flag is set.

For rigorous, cost-controlled experiments the provider is wrapped by
``BudgetedCachingClient`` (below), configured entirely through environment
variables so the run pipeline (run_once / run_batch / run_sweep) needs no
changes:

    LLM_MODEL          override the default model string
    LLM_TEMPERATURE    sampling temperature (default 0.0 for reproducibility)
    LLM_MAX_TOKENS     max output tokens per call (default 300)
    LLM_MAX_CALLS      hard per-process API-call budget; further decisions
                       return a safe 'hold' (default: unlimited)
    LLM_CACHE          path to a JSON cache; identical (model, prompt, state)
                       decisions are reused, cutting cost and adding determinism

Failures (network, rate-limit, malformed output) return a hold decision and
increment ``error_count`` — a Monte Carlo batch must never crash on a hiccup.
Token usage is accumulated in ``usage`` for post-hoc cost reporting.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request

from .client import parse_llm_json

_HOLD = {"action": "hold", "symbol": "SIVB", "quantity": 0, "rationale": "provider unavailable"}


class _HttpClient:
    mode = "base"
    url = ""

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 300):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.call_count = 0
        self.error_count = 0
        self.usage = {"input_tokens": 0, "output_tokens": 0}

    def _post(self, payload: dict, headers: dict) -> dict:
        req = urllib.request.Request(self.url, method="POST",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json", **headers})
        with urllib.request.urlopen(req, timeout=60) as r:
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

    def __init__(self, model: str = "claude-haiku-4-5-20251001", **kw):
        super().__init__(model, **kw)
        self.key = os.environ.get("ANTHROPIC_API_KEY", "")

    def _complete(self, system: str, user: str) -> str:
        resp = self._post(
            {"model": self.model, "max_tokens": self.max_tokens,
             "temperature": self.temperature, "system": system,
             "messages": [{"role": "user", "content": user}]},
            {"x-api-key": self.key, "anthropic-version": "2023-06-01"})
        u = resp.get("usage", {})
        self.usage["input_tokens"] += int(u.get("input_tokens", 0))
        self.usage["output_tokens"] += int(u.get("output_tokens", 0))
        return resp["content"][0]["text"]


class OpenAIClient(_HttpClient):
    mode = "openai"
    url = "https://api.openai.com/v1/chat/completions"

    def __init__(self, model: str = "gpt-4o-mini", **kw):
        super().__init__(model, **kw)
        self.key = os.environ.get("OPENAI_API_KEY", "")

    def _complete(self, system: str, user: str) -> str:
        resp = self._post(
            {"model": self.model, "max_tokens": self.max_tokens,
             "temperature": self.temperature,
             "messages": [{"role": "system", "content": system},
                          {"role": "user", "content": user}]},
            {"Authorization": f"Bearer {self.key}"})
        u = resp.get("usage", {})
        self.usage["input_tokens"] += int(u.get("prompt_tokens", 0))
        self.usage["output_tokens"] += int(u.get("completion_tokens", 0))
        return resp["choices"][0]["message"]["content"]


class GeminiClient(_HttpClient):
    mode = "gemini"

    def __init__(self, model: str = "gemini-2.5-flash", **kw):
        super().__init__(model, **kw)
        self.key = os.environ.get("GEMINI_API_KEY", "")
        self.url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{model}:generateContent?key={self.key}")

    def _complete(self, system: str, user: str) -> str:
        resp = self._post(
            {"systemInstruction": {"parts": [{"text": system}]},
             "contents": [{"parts": [{"text": user}]}],
             "generationConfig": {"temperature": self.temperature,
                                  "maxOutputTokens": self.max_tokens}},
            {})
        u = resp.get("usageMetadata", {})
        self.usage["input_tokens"] += int(u.get("promptTokenCount", 0))
        self.usage["output_tokens"] += int(u.get("candidatesTokenCount", 0))
        return resp["candidates"][0]["content"]["parts"][0]["text"]


class BudgetedCachingClient:
    """Wraps a provider with a call budget and an on-disk decision cache.

    - Budget: after ``max_calls`` real API calls, further decisions return a
      safe 'hold' (logged), so a runaway sweep cannot exceed a known cost.
    - Cache: identical (model, system, state) keys reuse the prior decision,
      cutting cost and making runs reproducible at temperature 0.
    Exposes the inner client's ``mode``, ``usage``, ``call_count`` for logging.
    """

    def __init__(self, inner, max_calls: int | None = None, cache_path: str | None = None):
        self.inner = inner
        self.mode = inner.mode
        self.max_calls = max_calls
        self.cache_path = cache_path
        self.budget_skips = 0
        self.cache_hits = 0
        self._cache: dict[str, dict] = {}
        if cache_path and os.path.exists(cache_path):
            try:
                self._cache = json.load(open(cache_path))
            except Exception:
                self._cache = {}

    @property
    def usage(self):
        return self.inner.usage

    @property
    def call_count(self):
        return self.inner.call_count

    @property
    def error_count(self):
        return self.inner.error_count

    def _key(self, system: str, state: dict) -> str:
        blob = self.inner.model + "|" + system + "|" + json.dumps(state, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def decide(self, system_prompt: str, state: dict) -> dict:
        key = self._key(system_prompt, state)
        if key in self._cache:
            self.cache_hits += 1
            return dict(self._cache[key])
        if self.max_calls is not None and self.inner.call_count >= self.max_calls:
            self.budget_skips += 1
            return dict(_HOLD, symbol=state.get("target_symbol", "SIVB"),
                        rationale="llm budget exhausted")
        decision = self.inner.decide(system_prompt, state)
        self._cache[key] = decision
        return decision

    def flush(self):
        if self.cache_path:
            try:
                json.dump(self._cache, open(self.cache_path, "w"))
            except Exception:
                pass


def _provider(llm_mode: str, model: str | None, temperature: float, max_tokens: int):
    if llm_mode == "anthropic":
        return AnthropicClient(model or "claude-haiku-4-5-20251001",
                               temperature=temperature, max_tokens=max_tokens)
    if llm_mode == "openai":
        return OpenAIClient(model or "gpt-4o-mini",
                            temperature=temperature, max_tokens=max_tokens)
    if llm_mode == "gemini":
        return GeminiClient(model or "gemini-2.5-flash",
                            temperature=temperature, max_tokens=max_tokens)
    raise ValueError(f"unknown llm_mode: {llm_mode}")


def make_client(llm_mode: str, seed: int = 0, model: str | None = None):
    """Factory. 'scripted' -> deterministic ScriptedClient; a provider name ->
    that provider wrapped in BudgetedCachingClient, configured via env vars
    (LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_MAX_CALLS, LLM_CACHE)."""
    from .client import ScriptedClient
    if llm_mode == "scripted":
        return ScriptedClient(seed)
    model = model or os.environ.get("LLM_MODEL") or None
    temperature = float(os.environ.get("LLM_TEMPERATURE", "0.0"))
    max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "300"))
    max_calls = os.environ.get("LLM_MAX_CALLS")
    cache = os.environ.get("LLM_CACHE")
    inner = _provider(llm_mode, model, temperature, max_tokens)
    return BudgetedCachingClient(inner, max_calls=int(max_calls) if max_calls else None,
                                 cache_path=cache)

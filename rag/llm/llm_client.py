"""Provider-independent LLM client: LLMClient.generate(prompt). Mock by default."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class LLMError(Exception):
    def __init__(self, message: str, code: str = "llm_error"):
        super().__init__(message)
        self.code = code


class LLMClient:
    provider: str = "mock"
    model: str = "mock-llm"

    def generate(self, prompt: str) -> dict[str, Any]:
        raise NotImplementedError


class MockLLMClient(LLMClient):
    """Deterministic template — no network, no keys. Used in dev/tests."""

    provider = "mock"
    model = "mock-llm"

    def generate(self, prompt: str) -> dict[str, Any]:
        first = next((ln for ln in prompt.splitlines() if ln.startswith("TARGETS:")), "TARGETS: ?")
        text = (
            f"## summary\nMock research summary for {first}. No clinical claims made.\n"
            "## biological_interpretation\nMock: interpretation placeholder derived from discretized features.\n"
            "## mechanistic_rationale\nMock: pathway/embedding rationale placeholder.\n"
            "## safety_considerations\nMock: safety requires real tumor/normal data; none claimed.\n"
            "## clinical_evidence\nMock: no retrieved trials; no efficacy claimed.\n"
            "## limitations\n- Mock LLM output; not real biology.\n- ML score is not a clinical probability.\n"
            "## confidence\nlow"
        )
        return {"text": text, "model": self.model, "prompt_tokens": len(prompt) // 4, "completion_tokens": 120}


class OpenAICompatibleClient(LLMClient):
    """OpenAI / OpenRouter via /chat/completions. Controlled errors, no raw tracebacks."""

    def __init__(self, provider: str, model: str, api_key: str, base_url: str, timeout: float = 60.0):
        self.provider = provider
        self.model = model
        self._key = api_key
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._timeout = timeout

    def generate(self, prompt: str) -> dict[str, Any]:
        import httpx

        try:
            r = httpx.post(
                self._url,
                headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
                json={"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2},
                timeout=self._timeout,
            )
        except Exception as e:
            raise LLMError(f"LLM provider unreachable ({self.provider}): {type(e).__name__}", "provider_unavailable") from e
        if r.status_code == 401:
            raise LLMError("LLM authentication failed (check API key)", "auth_error")
        if r.status_code == 429:
            raise LLMError("LLM rate limit exceeded", "rate_limit")
        if r.status_code >= 400:
            raise LLMError(f"LLM provider error HTTP {r.status_code}", "provider_error")
        try:
            data = r.json()
            text = data["choices"][0]["message"]["content"]
        except Exception as e:
            raise LLMError("LLM returned a malformed response", "malformed_response") from e
        if not (text or "").strip():
            raise LLMError("LLM returned an empty response", "empty_response")
        usage = data.get("usage", {}) or {}
        return {"text": text, "model": self.model, "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")}


def get_llm_client() -> LLMClient:
    provider = os.getenv("LLM_PROVIDER", "mock").lower()
    model = os.getenv("LLM_MODEL", "mock-llm")
    if provider in ("", "mock", "test"):
        c = MockLLMClient()
        c.model = model if model != "mock-llm" else "mock-llm"
        return c
    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise LLMError("OPENAI_API_KEY not set", "auth_error")
        return OpenAICompatibleClient(provider, model or "gpt-4o-mini", key, "https://api.openai.com/v1")
    if provider == "openrouter":
        key = os.getenv("OPENROUTER_API_KEY", "")
        if not key:
            raise LLMError("OPENROUTER_API_KEY not set", "auth_error")
        return OpenAICompatibleClient(provider, model or "openai/gpt-4o-mini", key, "https://openrouter.ai/api/v1")
    raise LLMError(f"Unknown LLM_PROVIDER={provider!r}", "config_error")

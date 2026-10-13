"""Model providers.

Three providers cover almost everything a student or small team runs:

    openai      any OpenAI compatible chat endpoint. This includes LM Studio,
                Ollama, vLLM and llama.cpp servers, so one provider covers
                both cloud and local models.
    anthropic   the Anthropic Messages API.
    mock        a deterministic fake used by the test suite and for dry runs.

Each provider returns a Completion with the text, token counts (when the
endpoint reports them) and wall clock latency in milliseconds.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx


@dataclass
class Completion:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


class Provider:
    name = "base"

    def complete(self, prompt: str, system: str | None = None) -> Completion:  # pragma: no cover
        raise NotImplementedError


class MockProvider(Provider):
    """Answers from a fixed dictionary; unknown prompts get a fixed fallback."""

    name = "mock"

    def __init__(self, answers: dict[str, str] | None = None, fallback: str = "I do not know."):
        self.answers = answers or {}
        self.fallback = fallback
        self.calls = 0

    def complete(self, prompt: str, system: str | None = None) -> Completion:
        self.calls += 1
        text = self.answers.get(prompt, self.fallback)
        return Completion(text=text, input_tokens=len(prompt.split()), output_tokens=len(text.split()), latency_ms=1.0)


class OpenAICompatibleProvider(Provider):
    name = "openai"

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: float = 120.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "lm-studio")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = httpx.Client(timeout=timeout)

    def complete(self, prompt: str, system: str | None = None) -> Completion:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        t0 = time.perf_counter()
        r = self.client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
        latency = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        data = r.json()
        usage = data.get("usage", {}) or {}
        return Completion(
            text=data["choices"][0]["message"]["content"] or "",
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=latency,
        )


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: float = 120.0,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = httpx.Client(timeout=timeout)

    def complete(self, prompt: str, system: str | None = None) -> Completion:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        t0 = time.perf_counter()
        r = self.client.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers)
        latency = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        data = r.json()
        text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
        usage = data.get("usage", {}) or {}
        return Completion(
            text=text,
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            latency_ms=latency,
        )


def make_provider(kind: str, model: str, base_url: str | None = None, **kwargs) -> Provider:
    if kind == "mock":
        return MockProvider()
    if kind == "openai":
        return OpenAICompatibleProvider(model=model, base_url=base_url or "https://api.openai.com/v1", **kwargs)
    if kind == "lmstudio":
        return OpenAICompatibleProvider(model=model, base_url=base_url or "http://localhost:1234/v1", **kwargs)
    if kind == "anthropic":
        return AnthropicProvider(model=model, **kwargs)
    raise ValueError(f"unknown provider '{kind}' (valid: mock, openai, lmstudio, anthropic)")

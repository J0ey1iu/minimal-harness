"""LLM provider factory — creates provider instances from config.

This is a Layer 2 service that knows how to instantiate concrete
LLM provider implementations based on configuration.
"""

from __future__ import annotations

from typing import Any

from minimal_harness.llm.anthropic import AnthropicLLMProvider
from minimal_harness.llm.llm import LLMProvider
from minimal_harness.llm.openai import OpenAILLMProvider


def create_llm_provider(cfg: dict[str, Any]) -> LLMProvider:
    provider = cfg.get("provider", "openai")
    kwargs: dict[str, Any] = {}
    if cfg.get("base_url"):
        kwargs["base_url"] = cfg["base_url"]
    if cfg.get("api_key"):
        kwargs["api_key"] = cfg["api_key"]

    model = cfg.get("model", "")
    if provider == "anthropic":
        from anthropic import AsyncAnthropic

        return AnthropicLLMProvider(
            client=AsyncAnthropic(**kwargs),
            model=model,
        )
    from openai import AsyncOpenAI

    return OpenAILLMProvider(client=AsyncOpenAI(**kwargs), model=model)

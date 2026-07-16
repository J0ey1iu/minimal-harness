"""LLM provider factory — creates provider instances from config.

This is a Layer 2 service that knows how to instantiate concrete
LLM provider implementations based on configuration.
"""

from __future__ import annotations

import logging
from typing import Any

from minimal_harness.llm.anthropic import AnthropicLLMProvider
from minimal_harness.llm.llm import LLMProvider, ProviderFactory
from minimal_harness.llm.openai import OpenAILLMProvider
from minimal_harness.types import ExtraHeadersProvider

logger = logging.getLogger(__name__)


def _openai_factory(
    cfg: dict[str, Any],
) -> LLMProvider:
    from openai import AsyncOpenAI

    api_key = cfg.get("api_key", "")
    if not api_key:
        raise ValueError("openai provider requires api_key")
    base_url = cfg.get("base_url", "")
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    logger.info(
        "llm.factory.openai has_key=True base_url=%s model=%s",
        base_url or "(default)",
        cfg.get("model", ""),
    )
    client = AsyncOpenAI(**kwargs)
    logger.info("llm.factory.openai.client base_url=%s", client.base_url)
    return OpenAILLMProvider(
        client=client,
        model=cfg.get("model", ""),
        llm_extra_headers_provider=cfg.get("_extra_headers_provider"),
    )


def _anthropic_factory(
    cfg: dict[str, Any],
) -> LLMProvider:
    from anthropic import AsyncAnthropic

    api_key = cfg.get("api_key", "")
    if not api_key:
        raise ValueError("anthropic provider requires api_key")
    base_url = cfg.get("base_url", "")
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return AnthropicLLMProvider(
        client=AsyncAnthropic(**kwargs),
        model=cfg.get("model", ""),
        llm_extra_headers_provider=cfg.get("_extra_headers_provider"),
    )


def register_builtin_providers(
    registry: ProviderFactory,
) -> None:
    registry.register("openai", _openai_factory)
    registry.register("anthropic", _anthropic_factory)


def create_llm_provider(
    cfg: dict[str, Any],
    llm_extra_headers_provider: ExtraHeadersProvider | None = None,
) -> LLMProvider:
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
            llm_extra_headers_provider=llm_extra_headers_provider,
        )
    from openai import AsyncOpenAI

    return OpenAILLMProvider(
        client=AsyncOpenAI(**kwargs),
        model=model,
        llm_extra_headers_provider=llm_extra_headers_provider,
    )

"""Application context that owns configuration, registry, memory store, and LLM provider."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from minimal_harness.tool.base import Tool

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from minimal_harness.client.built_in.config import (
    add_model,
    collect_tools,
    load_config,
    save_config,
)
from minimal_harness.llm import AnthropicLLMProvider, LLMProvider, OpenAILLMProvider
from minimal_harness.memory_store import MemoryStore
from minimal_harness.tool.registry import ToolRegistry


class TUIConfig:
    """Configuration loading, saving, and model management."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = config or load_config()

    def update_config(self, result: dict[str, Any]) -> None:
        self.config.update(result)
        if "model" in result:
            add_model(result["model"])
        save_config(self.config)


def create_llm_provider(cfg: dict[str, Any]) -> LLMProvider:
    """Create an LLM provider from a config dict."""
    provider = cfg.get("provider", "openai")
    kwargs: dict[str, Any] = {}
    if cfg.get("base_url"):
        kwargs["base_url"] = cfg["base_url"]
    if cfg.get("api_key"):
        kwargs["api_key"] = cfg["api_key"]

    if provider == "anthropic":
        return AnthropicLLMProvider(
            client=AsyncAnthropic(**kwargs),
            model=cfg.get("model", ""),
        )
    return OpenAILLMProvider(client=AsyncOpenAI(**kwargs), model=cfg.get("model", ""))


class AppContext:
    """Application context — facade over TUIConfig, ToolRegistry, and MemoryStore."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        registry: ToolRegistry | None = None,
    ) -> None:
        self._config_manager = TUIConfig(config=config)
        self._registry: ToolRegistry = registry or ToolRegistry()
        self._memory_store = MemoryStore()

    @property
    def config(self) -> dict[str, Any]:
        return self._config_manager.config

    @config.setter
    def config(self, value: dict[str, Any]) -> None:
        self._config_manager.config = value

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    @property
    def all_tools(self) -> dict[str, "Tool"]:
        return {t.name: t for t in self._registry.get_all()}

    @property
    def memory_store(self) -> MemoryStore:
        return self._memory_store

    def rebuild(self) -> None:
        self._registry.clear()
        collect_tools(self.config, self._registry)

    def refresh_tools(self) -> None:
        self.rebuild()

    def update_config(self, result: dict[str, Any]) -> None:
        self._config_manager.update_config(result)

    def create_llm_provider(self, cfg: dict[str, Any] | None = None) -> LLMProvider:
        effective = cfg if cfg is not None else self.config
        return create_llm_provider(effective)

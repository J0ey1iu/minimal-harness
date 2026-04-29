"""Application context that owns configuration, registry, and agent lifecycle."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

if TYPE_CHECKING:
    from minimal_harness.agent import Agent

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from minimal_harness.agent import SimpleAgent
from minimal_harness.client.built_in.config import (
    add_model,
    collect_tools,
    load_config,
    save_config,
)
from minimal_harness.llm import AnthropicLLMProvider, LLMProvider, OpenAILLMProvider
from minimal_harness.memory import Memory
from minimal_harness.tool.base import Tool
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


class ToolManager:
    """Tool lifecycle: collection, registry, active tool selection."""

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry: ToolRegistry = registry or ToolRegistry()
        self._all_tools: dict[str, Tool] = {}
        self.active_tools: list[Tool] = []

    def rebuild(self, config: dict[str, Any]) -> None:
        self.registry.clear()
        self._all_tools = collect_tools(config, self.registry)
        for t in self._all_tools.values():
            self.registry.register(t)
        self.active_tools = list(self._all_tools.values())

    def refresh_tools(self, config: dict[str, Any]) -> None:
        self.registry.clear()
        self._all_tools = collect_tools(config, self.registry)
        for t in self._all_tools.values():
            self.registry.register(t)

    def select_tools(self, chosen: list[str]) -> None:
        self.active_tools = [self._all_tools[n] for n in chosen if n in self._all_tools]

    @property
    def all_tools(self) -> dict[str, Tool]:
        return self._all_tools


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
    """Application context — facade over TUIConfig, ToolManager, and factories."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        registry: ToolRegistry | None = None,
        llm_provider_factory: Callable[[dict[str, Any]], LLMProvider] | None = None,
        agent_factory: Callable[..., Agent] | None = None,
    ) -> None:
        self._config_manager = TUIConfig(config=config)
        self._tool_manager = ToolManager(registry=registry)
        self._llm_provider_factory = llm_provider_factory
        self._agent_factory = agent_factory or _create_simple_agent

    @property
    def config(self) -> dict[str, Any]:
        return self._config_manager.config

    @config.setter
    def config(self, value: dict[str, Any]) -> None:
        self._config_manager.config = value

    @property
    def registry(self) -> ToolRegistry:
        return self._tool_manager.registry

    @registry.setter
    def registry(self, value: ToolRegistry) -> None:
        self._tool_manager.registry = value

    @property
    def all_tools(self) -> dict[str, Tool]:
        return self._tool_manager.all_tools

    @property
    def _all_tools(self) -> dict[str, Tool]:
        return self._tool_manager.all_tools

    @_all_tools.setter
    def _all_tools(self, value: dict[str, Tool]) -> None:
        self._tool_manager._all_tools = value

    @property
    def active_tools(self) -> list[Tool]:
        return self._tool_manager.active_tools

    @active_tools.setter
    def active_tools(self, value: list[Tool]) -> None:
        self._tool_manager.active_tools = value

    def rebuild(self) -> None:
        self._tool_manager.rebuild(self.config)

    def refresh_tools(self) -> None:
        self._tool_manager.refresh_tools(self.config)

    def update_config(self, result: dict[str, Any]) -> None:
        self._config_manager.update_config(result)

    def select_tools(self, chosen: list[str]) -> None:
        self._tool_manager.select_tools(chosen)

    def _create_llm_provider(self, cfg: dict[str, Any]) -> LLMProvider:
        if self._llm_provider_factory is not None:
            return self._llm_provider_factory(cfg)
        return create_llm_provider(cfg)


def _create_simple_agent(
    llm_provider: LLMProvider,
    tools: Sequence[Tool] | None,
    memory: Memory,
) -> Agent:
    return SimpleAgent(llm_provider=llm_provider, tools=tools, memory=memory)

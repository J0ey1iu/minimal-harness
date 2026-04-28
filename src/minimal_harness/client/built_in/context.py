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
from minimal_harness.client.built_in.memory import PersistentMemory
from minimal_harness.llm import AnthropicLLMProvider, LLMProvider, OpenAILLMProvider
from minimal_harness.tool.base import Tool
from minimal_harness.tool.registry import ToolRegistry


class AppContext:
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        registry: ToolRegistry | None = None,
        llm_provider_factory: Callable[[dict[str, Any]], LLMProvider] | None = None,
        agent_factory: Callable[..., Agent] | None = None,
    ) -> None:
        self.config = config or load_config()
        self.registry: ToolRegistry = registry or ToolRegistry()
        self._all_tools: dict[str, Tool] = {}
        self.active_tools: list[Tool] = []
        self.memory: PersistentMemory | None = None
        self._llm_provider_factory = llm_provider_factory
        self._agent_factory = agent_factory or _create_simple_agent

    def _create_llm_provider(self, cfg: dict[str, Any]) -> LLMProvider:
        if self._llm_provider_factory is not None:
            return self._llm_provider_factory(cfg)

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
        return OpenAILLMProvider(
            client=AsyncOpenAI(**kwargs), model=cfg.get("model", "")
        )

    def rebuild(self, system_prompt: str | None = None) -> None:
        cfg = self.config
        self._all_tools = collect_tools(cfg, self.registry)
        for t in self._all_tools.values():
            self.registry.register(t)
        self.active_tools = list(self._all_tools.values())

        if system_prompt is None:
            system_prompt = ""
        if self.memory is None:
            self.memory = PersistentMemory(system_prompt=system_prompt)
        else:
            msgs = self.memory.get_all_messages()
            if (
                msgs
                and msgs[0].get("role") == "system"
                and msgs[0].get("content") != system_prompt
            ):
                self.memory.update_system_prompt(system_prompt)

    def refresh_tools(self) -> None:
        self.registry.clear()
        self._all_tools = collect_tools(self.config, self.registry)
        for t in self._all_tools.values():
            self.registry.register(t)

    def update_config(self, result: dict[str, Any]) -> None:
        self.config.update(result)
        if "model" in result:
            add_model(result["model"])
        save_config(self.config)

    def select_tools(self, chosen: list[str]) -> None:
        self.active_tools = [self._all_tools[n] for n in chosen if n in self._all_tools]

    def reset_memory(self, system_prompt: str | None = None) -> None:
        if system_prompt is None:
            system_prompt = ""
        self.memory = PersistentMemory(system_prompt=system_prompt)

    @property
    def all_tools(self) -> dict[str, Tool]:
        return self._all_tools


def _create_simple_agent(
    llm_provider: LLMProvider,
    tools: Sequence[Tool] | None,
    memory: PersistentMemory,
) -> Agent:
    return SimpleAgent(llm_provider=llm_provider, tools=tools, memory=memory)

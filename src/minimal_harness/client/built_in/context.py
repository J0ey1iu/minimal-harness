"""Application context that owns configuration, registry, and agent lifecycle."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from minimal_harness.agent import Agent, SimpleAgent
from minimal_harness.client.built_in.config import (
    DEFAULT_CONFIG,
    add_model,
    collect_tools,
    load_config,
    read_system_prompt,
    save_config,
)
from minimal_harness.client.built_in.memory import PersistentMemory
from minimal_harness.llm import AnthropicLLMProvider, LLMProvider, OpenAILLMProvider
from minimal_harness.tool.base import StreamingTool
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
        self._all_tools: dict[str, StreamingTool] = {}
        self.active_tools: list[StreamingTool] = []
        self.memory: PersistentMemory | None = None
        self.agent: Agent | None = None
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

    def rebuild(self) -> None:
        cfg = self.config
        self._all_tools = collect_tools(cfg, self.registry)
        selected = cfg.get("selected_tools") or []
        if selected:
            self.active_tools = [
                self._all_tools[n] for n in selected if n in self._all_tools
            ]
        else:
            self.active_tools = list(self._all_tools.values())

        llm = self._create_llm_provider(cfg)

        prompt_path = cfg.get("system_prompt", DEFAULT_CONFIG["system_prompt"])
        prompt = read_system_prompt(Path(prompt_path)) if prompt_path else ""
        if self.memory is None:
            self.memory = PersistentMemory(system_prompt=prompt)
        else:
            msgs = self.memory.get_all_messages()
            if (
                msgs
                and msgs[0].get("role") == "system"
                and msgs[0].get("content") != prompt
            ):
                self.memory.update_system_prompt(prompt)

        self.agent = self._agent_factory(
            llm_provider=llm, tools=self.active_tools, memory=self.memory
        )

    def update_config(self, result: dict[str, Any]) -> None:
        self.config.update(result)
        if "model" in result:
            add_model(result["model"])
        save_config(self.config)

    def select_tools(self, chosen: list[str]) -> None:
        self.active_tools = [self._all_tools[n] for n in chosen if n in self._all_tools]
        self.config["selected_tools"] = chosen
        save_config(self.config)

    def reset_memory(self, system_prompt: str | None = None) -> None:
        if system_prompt is None:
            prompt_path = self.config.get(
                "system_prompt", DEFAULT_CONFIG["system_prompt"]
            )
            system_prompt = read_system_prompt(Path(prompt_path)) if prompt_path else ""
        self.memory = PersistentMemory(system_prompt=system_prompt)

    @property
    def all_tools(self) -> dict[str, StreamingTool]:
        return self._all_tools


def _create_simple_agent(
    llm_provider: LLMProvider,
    tools: Sequence[StreamingTool] | None,
    memory: PersistentMemory,
) -> Agent:
    return SimpleAgent(llm_provider=llm_provider, tools=tools, memory=memory)

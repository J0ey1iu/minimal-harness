from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

if TYPE_CHECKING:
    from minimal_harness.agent.protocol import Agent
    from minimal_harness.client.built_in.memory import PersistentMemory
    from minimal_harness.llm import LLMProvider
    from minimal_harness.tool.base import StreamingTool
    from minimal_harness.types import AgentEvent


DEFAULT_QUEUE_SIZE = 1000


@dataclass
class AgentMetadata:
    name: str
    description: str
    agent: Agent


@runtime_checkable
class AgentRegistryProtocol(Protocol):
    def register(
        self, agent: Agent, *, name: str | None = None, description: str | None = None
    ) -> None: ...
    def unregister(self, name: str) -> bool: ...
    def get(self, name: str) -> AgentMetadata | None: ...
    def get_all(self) -> list[AgentMetadata]: ...
    def names(self) -> list[str]: ...
    def clear(self) -> None: ...
    def add_listener(self, listener: Callable[[], None]) -> None: ...
    def remove_listener(self, listener: Callable[[], None]) -> None: ...


@dataclass
class Session:
    session_id: str
    name: str
    agent: Agent
    memory: PersistentMemory
    tools: list[StreamingTool]
    event_queue: asyncio.Queue["AgentEvent"] | None = None

    def __post_init__(self) -> None:
        if self.event_queue is None:
            self.event_queue = asyncio.Queue(maxsize=DEFAULT_QUEUE_SIZE)

    def has_events(self) -> bool:
        return not self.event_queue.empty() if self.event_queue else False

    async def drain_events(self) -> list["AgentEvent"]:
        events: list["AgentEvent"] = []
        while not self.event_queue.empty():  # type: ignore[union-attr]
            try:
                events.append(self.event_queue.get_nowait())  # type: ignore[union-attr]
            except asyncio.QueueEmpty:
                break
        return events

    def rebuild(
        self,
        config: dict[str, Any],
        tools: list[StreamingTool] | None = None,
        agent_factory: Callable[..., Agent] | None = None,
    ) -> None:
        from minimal_harness.agent.simple import SimpleAgent

        if tools is not None:
            self.tools = tools

        llm_provider = self._create_llm_provider(config)
        factory = agent_factory or SimpleAgent
        self.agent = factory(
            llm_provider=llm_provider, tools=self.tools, memory=self.memory
        )

    def _create_llm_provider(self, config: dict[str, Any]) -> "LLMProvider":
        from anthropic import AsyncAnthropic
        from openai import AsyncOpenAI

        from minimal_harness.llm import AnthropicLLMProvider, OpenAILLMProvider

        provider = config.get("provider", "openai")
        kwargs: dict[str, Any] = {}
        if config.get("base_url"):
            kwargs["base_url"] = config["base_url"]
        if config.get("api_key"):
            kwargs["api_key"] = config["api_key"]

        if provider == "anthropic":
            return AnthropicLLMProvider(
                client=AsyncAnthropic(**kwargs),
                model=config.get("model", ""),
            )
        return OpenAILLMProvider(
            client=AsyncOpenAI(**kwargs), model=config.get("model", "")
        )


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, AgentMetadata] = {}
        self._listeners: list[Callable[[], None]] = []

    def register(
        self,
        agent: Agent,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        agent_name = name or getattr(agent, "name", None) or agent.__class__.__name__
        agent_description = description or getattr(agent, "description", None) or ""
        self._agents[agent_name] = AgentMetadata(
            name=agent_name,
            description=agent_description,
            agent=agent,
        )
        self._notify()

    def unregister(self, name: str) -> bool:
        if name in self._agents:
            del self._agents[name]
            self._notify()
            return True
        return False

    def get(self, name: str) -> AgentMetadata | None:
        return self._agents.get(name)

    def get_all(self) -> list[AgentMetadata]:
        return list(self._agents.values())

    def names(self) -> list[str]:
        return list(self._agents.keys())

    def clear(self) -> None:
        self._agents.clear()
        self._notify()

    def add_listener(self, listener: Callable[[], None]) -> None:
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[], None]) -> None:
        self._listeners.remove(listener)

    def _notify(self) -> None:
        for listener in self._listeners:
            listener()

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Iterable, Sequence

from minimal_harness.agent.protocol import Agent
from minimal_harness.agent.registry import AgentRegistryProtocol, Session

if TYPE_CHECKING:
    from minimal_harness.client.built_in.memory import PersistentMemory
    from minimal_harness.llm import LLMProvider
    from minimal_harness.memory import ExtendedInputContentPart
    from minimal_harness.tool.base import StreamingTool
    from minimal_harness.types import AgentEvent


class AgentRuntime:
    def __init__(self, registry: AgentRegistryProtocol) -> None:
        self.registry = registry
        self._sessions: dict[str, Session] = {}
        self._running_tasks: dict[str, asyncio.Task[None]] = {}

    def get_running_session_ids(self) -> list[str]:
        return list(self._running_tasks.keys())

    def is_session_running(self, session_id: str) -> bool:
        return session_id in self._running_tasks

    async def run(
        self,
        user_input: "Iterable[ExtendedInputContentPart]",
        stop_event: asyncio.Event | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator["AgentEvent"]:
        if session_id is None:
            return
        session = self._sessions.get(session_id)
        if session is None:
            return
        async for event in session.agent.run(
            user_input=user_input,
            stop_event=stop_event,
            memory=session.memory,
            tools=session.tools,
        ):
            if session.event_queue is not None and not session.event_queue.full():
                session.event_queue.put_nowait(event)
            yield event

    def create_session(
        self,
        config: dict[str, Any],
        tools: Sequence[StreamingTool],
        memory: PersistentMemory,
        agent_factory: Callable[..., Agent],
    ) -> Session:
        llm = self._create_llm_provider(config)
        agent = agent_factory(llm_provider=llm, tools=list(tools), memory=memory)
        session = Session(
            session_id=memory._session_id,
            name=memory.title or "New Chat",
            agent=agent,
            memory=memory,
            tools=list(tools),
        )
        self._sessions[session.session_id] = session
        return session

    def load_session(
        self,
        session_id: str,
        config: dict[str, Any],
        tools: Sequence[StreamingTool],
        agent_factory: Callable[..., Agent],
        memory_dir: str | None = None,
    ) -> Session:
        from pathlib import Path

        from minimal_harness.client.built_in.memory import PersistentMemory

        directory = Path(memory_dir) if memory_dir else None
        memory = PersistentMemory.from_session(session_id, memory_dir=directory)
        llm = self._create_llm_provider(config)
        agent = agent_factory(llm_provider=llm, tools=list(tools), memory=memory)
        session = Session(
            session_id=memory._session_id,
            name=memory.title or "Untitled",
            agent=agent,
            memory=memory,
            tools=list(tools),
        )
        self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def list_sessions(self) -> list[Session]:
        return list(self._sessions.values())

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

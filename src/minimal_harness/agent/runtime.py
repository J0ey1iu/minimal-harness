from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

from minimal_harness.agent.protocol import Agent
from minimal_harness.agent.registry import AgentRegistryProtocol, Session

if TYPE_CHECKING:
    from minimal_harness.client.built_in.memory import PersistentMemory
    from minimal_harness.llm import LLMProvider
    from minimal_harness.tool.base import StreamingTool


class AgentRuntime:
    def __init__(self, registry: AgentRegistryProtocol) -> None:
        self.registry = registry
        self._sessions: dict[str, Session] = {}
        self._current: Session | None = None

    @property
    def current_session(self) -> Session | None:
        return self._current

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
        self._current = session
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
        self._current = session
        return session

    def set_current_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            self._current = self._sessions[session_id]
            return True
        return False

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

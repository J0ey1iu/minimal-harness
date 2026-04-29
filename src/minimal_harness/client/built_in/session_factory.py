"""Session creation and loading — builds memory, tools, LLM, and agent for a session."""

from __future__ import annotations

from typing import Any

from minimal_harness.agent.simple import SimpleAgent
from minimal_harness.client.built_in.context import AppContext
from minimal_harness.client.built_in.memory import PersistentMemory
from minimal_harness.client.built_in.session import ConversationSession
from minimal_harness.tool.base import Tool


class SessionFactory:
    """Creates and loads ConversationSession instances with all dependencies wired."""

    def __init__(self, ctx: AppContext) -> None:
        self._ctx = ctx

    def create_session(
        self,
        agent_name: str = "general_assistant",
        system_prompt: str | None = None,
        default_tools: list[str] | None = None,
    ) -> ConversationSession:
        self._ctx.rebuild()
        memory = PersistentMemory(system_prompt=system_prompt or "")
        memory.agent_name = agent_name

        base_tools = self._ctx.all_tools
        if default_tools is not None:
            tools = [base_tools[n] for n in default_tools if n in base_tools]
        else:
            tools = self._ctx.active_tools

        llm = self._ctx._create_llm_provider(self._ctx.config)
        agent = SimpleAgent(llm_provider=llm, tools=list(tools), memory=memory)

        session = ConversationSession(
            session_id=memory.session_id,
            agent=agent,
            memory=memory,
            tools=list(tools),
            name=agent_name,
        )
        if default_tools is not None:
            session.memory.selected_tools = default_tools
        return session

    def load_session_from_disk(self, session_id: str) -> ConversationSession | None:
        try:
            memory = PersistentMemory.from_session(session_id)
        except (FileNotFoundError, ValueError):
            return None

        llm = self._ctx._create_llm_provider(self._ctx.config)
        tools = self._ctx.active_tools
        if memory.selected_tools:
            restored = [
                self._ctx.all_tools[n]
                for n in memory.selected_tools
                if n in self._ctx.all_tools
            ]
            if restored:
                tools = restored

        agent = SimpleAgent(llm_provider=llm, tools=tools, memory=memory)
        return ConversationSession(
            session_id=session_id,
            agent=agent,
            memory=memory,
            tools=list(tools),
            name=memory.agent_name or memory.title or "Untitled",
        )

    def rebuild_current_session(
        self,
        session: ConversationSession,
        llm_provider: Any,
        tools: list[Tool] | None = None,
        agent_factory: Any = None,
    ) -> None:
        if tools is not None:
            session.tools = list(tools)
        factory = agent_factory or SimpleAgent
        session.agent = factory(
            llm_provider=llm_provider, tools=session.tools, memory=session.memory
        )

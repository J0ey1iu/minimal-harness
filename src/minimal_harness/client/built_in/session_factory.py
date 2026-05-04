"""Session creation and loading — builds memory, resolves tools, and creates session structures."""

from __future__ import annotations

from typing import Any

from minimal_harness.client.built_in.context import AppContext
from minimal_harness.client.built_in.session import ConversationSession


class SessionFactory:
    """Creates and loads ConversationSession instances using Layer 2 services.

    Tools are resolved via the ToolRegistry. Memory is managed via MemoryStore.
    No agent instances are created here — agents are created by AgentRuntime.
    """

    def __init__(self, ctx: AppContext) -> None:
        self._ctx = ctx

    def create_session(
        self,
        agent_name: str = "general_assistant",
        system_prompt: str | None = None,
        default_tools: list[str] | None = None,
    ) -> ConversationSession:
        self._ctx.rebuild()
        store = self._ctx.memory_store
        memory = store.create_memory(
            system_prompt=system_prompt or "",
            agent_name=agent_name,
        )

        tool_names = default_tools or []

        return ConversationSession(
            session_id=memory.memory_id,
            agent_metadata_id=agent_name,
            memory_id=memory.memory_id,
            tool_names=list(tool_names),
            name=agent_name,
        )

    def load_session_from_disk(self, session_id: str) -> ConversationSession | None:
        store = self._ctx.memory_store
        memory = store.get_memory(session_id)
        if memory is None:
            return None

        return ConversationSession(
            session_id=session_id,
            agent_metadata_id=memory.agent_name or "general_assistant",
            memory_id=memory.memory_id,
            tool_names=[],
            name=memory.agent_name or memory.title or "Untitled",
        )

    def rebuild_current_session(
        self,
        session: ConversationSession,
        tools: list[Any] | None = None,
    ) -> None:
        if tools is not None:
            session.tool_names = [t.name for t in tools]

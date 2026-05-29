"""Session creation and loading — builds memory, resolves tools, and creates session structures."""

from __future__ import annotations

from typing import Any

from minimal_harness.client.built_in.context import AppContext
from minimal_harness.client.built_in.session import ConversationSession


class SessionFactory:
    """Creates and loads ConversationSession instances using Layer 2 services.

    Tools are resolved via the ToolRegistry. Sessions are managed via JsonlSessionStore.
    No agent instances are created here — agents are created by AgentRuntime.
    """

    def __init__(self, ctx: AppContext) -> None:
        self._ctx = ctx

    async def create_session(
        self,
        agent_name: str = "general_assistant",
        default_tools: list[str] | None = None,
    ) -> ConversationSession:
        store = self._ctx.session_store
        session = await store.create_session(
            agent_name=agent_name,
        )

        tool_names = default_tools or []

        return ConversationSession(
            session=session,
            agent_metadata_id=agent_name,
            tool_names=list(tool_names),
        )

    async def load_session_from_disk(
        self, session_id: str
    ) -> ConversationSession | None:
        store = self._ctx.session_store
        session = await store.get_session(session_id)
        if session is None:
            return None

        return ConversationSession(
            session=session,
            agent_metadata_id=session.agent_name or "general_assistant",
            tool_names=[],
        )

    def rebuild_current_session(
        self,
        session: ConversationSession,
        tools: list[Any] | None = None,
    ) -> None:
        if tools is not None:
            session.tool_names = [t.name for t in tools]  # type: ignore[union-attr]

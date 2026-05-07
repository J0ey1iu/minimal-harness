from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from minimal_harness.types import (
        AgentEnd,
        LLMEnd,
        ToolCall,
    )


class Middleware:
    """Observability hooks into the agent lifecycle.

    Subclass and override the hooks you need. Every hook defaults to a no-op,
    so you only implement the lifecycle events you care about.

    Middleware runs synchronously within the agent's async event loop — hooks
    are ``async`` so that implementations can perform I/O (logging, metrics,
    tracing) without blocking the agent.

    Usage::

        class CostTracker(Middleware):
            async def on_llm_end(self, event: LLMEnd) -> None:
                if event.usage:
                    print(f"Tokens used: {event.usage['total_tokens']}")
    """

    async def on_agent_start(self, user_input: Any) -> None:
        """Called before the agent begins processing user input."""

    async def on_agent_end(self, event: AgentEnd) -> None:
        """Called when the agent run finishes (success, error, or interrupt)."""

    async def on_llm_start(self, messages: list[dict[str, Any]], tools: Any) -> None:
        """Called before each LLM chat call."""

    async def on_llm_end(self, event: LLMEnd) -> None:
        """Called after each LLM chat call completes."""

    async def on_tool_start(self, tool_call: ToolCall) -> None:
        """Called before an individual tool begins executing."""

    async def on_tool_end(self, tool_call: ToolCall, result: Any) -> None:
        """Called after an individual tool finishes successfully."""

    async def on_tool_error(self, tool_call: ToolCall, error: Exception) -> None:
        """Called when an individual tool execution fails with an exception."""

    async def should_allow_tool(
        self, tool_call: ToolCall, *args: Any, **kwargs: Any
    ) -> bool:
        """Return ``False`` to veto tool execution before it starts.

        The default implementation allows every tool call. Override this
        to enforce permission / safety policies at runtime.

        Accepts ``*args, **kwargs`` so implementations can receive
        caller-supplied context (e.g. user info, session data) without
        the base signature needing to know about every use case.

        When ``False`` is returned the tool is **not** executed; a synthetic
        ``ToolEnd`` with a ``PermissionError`` result is emitted instead so
        that the LLM receives feedback about the denial.
        """
        return True

    async def on_error(self, error: BaseException) -> None:
        """Called on unhandled errors during the agent loop.

        Note: ``asyncio.CancelledError`` is handled separately (it triggers
        ``on_agent_end`` with ``interrupted=True``) and does *not* reach this
        hook.
        """

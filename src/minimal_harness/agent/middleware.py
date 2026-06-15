from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from minimal_harness.types import (
        AgentEnd,
        CompactionEnd,
        CompactionStart,
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

    async def on_compaction_start(self, event: CompactionStart) -> None:
        """Called before ``Memory.compact()`` starts streaming the summary.

        Only fires for ``agent_type="compacting"`` runs that actually cross
        the configured ``prompt_token_threshold``. The default implementation
        is a no-op; override to log, meter, or veto the compaction at the
        data-collection layer.
        """

    async def on_compaction_end(self, event: CompactionEnd) -> None:
        """Called after ``Memory.compact()`` finishes (success or failure).

        Receives the same event the agent emits, so failed compactions
        (``event.error`` is set) are visible here too.
        """

    async def on_tool_start(self, tool_call: ToolCall) -> None:
        """Called before an individual tool begins executing."""

    async def on_tool_end(self, tool_call: ToolCall, result: Any) -> None:
        """Called after an individual tool finishes successfully."""

    async def on_tool_error(self, tool_call: ToolCall, error: Exception) -> None:
        """Called when an individual tool execution fails with an exception."""

    async def should_allow_tool(
        self, tool_call: ToolCall, *args: Any, **kwargs: Any
    ) -> bool | str:
        """Return ``False`` or a reason string to veto tool execution.

        The default implementation allows every tool call. Override this
        to enforce permission / safety policies at runtime.

        Accepts ``*args, **kwargs`` so implementations can receive
        caller-supplied context (e.g. user info, session data) without
        the base signature needing to know about every use case.

        Return values:

        * ``True`` — tool is allowed to execute.
        * ``False`` — tool is denied with a generic "denied by policy" message.
        * A ``str`` — tool is denied; the string is used as the error message
          so the LLM receives a specific reason for the denial.

        When the tool is denied a synthetic ``ToolEnd`` with a
        ``PermissionError`` result is emitted instead.
        """
        return True

    async def on_error(self, error: BaseException) -> None:
        """Called on unhandled errors during the agent loop.

        Note: ``asyncio.CancelledError`` is handled separately (it triggers
        ``on_agent_end`` with ``interrupted=True``) and does *not* reach this
        hook.
        """

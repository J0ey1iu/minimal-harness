"""Tool-compaction agent loop -- strips tool call/result pairs from
the forward buffer after the LLM has produced a final answer, and
optionally compacts the full conversation history when prompt-token
usage exceeds a configured threshold.

After the LLM returns a response without ``tool_calls`` (i.e. a
final answer), all preceding ``role="tool"`` messages and
``tool_calls`` on assistant messages are removed from the forward
buffer.  The LLM API constraint (every ``tool_calls`` must be
followed by tool responses) is met _during_ the round because tool
messages are present when the next LLM call is made; they are only
cleaned up afterwards, once they are no longer needed.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Callable, Sequence

from minimal_harness.llm.llm import LLMProvider
from minimal_harness.memory import Memory, Message
from minimal_harness.types import AgentEvent

from minimal_harness.types import (
    CompactionEnd,
    CompactionStart,
)

from .base import BaseAgent
from .middleware import Middleware
from .protocol import InputContentConversionFunction

logger = logging.getLogger(__name__)


class ToolCompactionAgent(BaseAgent):
    """Agent loop that strips tool call/result pairs after a final answer.

    Parameters
    ----------
    llm_provider : LLMProvider
        The LLM backend.
    summarizer : Callable
        Streaming summariser that takes ``(list[Message], existing_summary)``
        and yields summary text chunks (used for full conversation
        compaction).
    prompt_token_threshold : int
        Cumulative prompt-token threshold for full conversation
        compaction. When exceeded, older messages are folded into a
        summary. Set to ``0`` to disable. Default ``0``.
    keep_recent : int
        Number of most recent messages to preserve verbatim during
        full conversation compaction. Default ``6``.
    max_iterations : int
        Maximum number of LLM-tool cycles. Default ``100``.
    custom_input_conversion : callable, optional
        Optional input conversion function.
    middleware : sequence of Middleware
        Middleware instances. Default ``()``.
    emit_message_events : bool
        Whether to emit ``MessageEvent`` for each conversation message.
        Default ``True``.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        summarizer: Callable[[list[Message], str | None], AsyncIterator[str]],
        prompt_token_threshold: int = 0,
        keep_recent: int = 6,
        max_iterations: int = 100,
        custom_input_conversion: InputContentConversionFunction | None = None,
        middleware: Sequence[Middleware] = (),
        emit_message_events: bool = True,
    ):
        super().__init__(
            llm_provider=llm_provider,
            max_iterations=max_iterations,
            custom_input_conversion=custom_input_conversion,
            middleware=middleware,
            emit_message_events=emit_message_events,
        )
        self._summarizer = summarizer
        self._prompt_token_threshold = prompt_token_threshold
        self._keep_recent = keep_recent

    async def _post_llm_response(
        self,
        llm_response: Any,
        memory: Memory,
    ) -> AsyncIterator[AgentEvent]:
        """Strip tool call/result pairs after a final answer.

        Full conversation compaction is deferred to ``_on_run_end``.
        """
        if not llm_response.tool_calls:
            async for evt in memory.strip_tool_call_pairs():
                if isinstance(evt, CompactionStart):
                    for m in self._middleware:
                        await m.on_compaction_start(evt)
                elif isinstance(evt, CompactionEnd):
                    for m in self._middleware:
                        await m.on_compaction_end(evt)
                yield evt

    async def _on_run_end(
        self,
        memory: Memory,
    ) -> AsyncIterator[AgentEvent]:
        """Full conversation compaction, after all rounds complete."""
        if self._prompt_token_threshold <= 0:
            return
            yield  # pragma: no cover

        cumulative_tokens = memory.get_message_usage().get("total_tokens", 0)
        if cumulative_tokens <= self._prompt_token_threshold:
            return
            yield  # pragma: no cover

        compaction_error: str | None = None
        compaction_summary: str = ""
        compaction_meta: dict[str, Any] = {}

        async for evt in memory.compact(
            self._summarizer,
            self._keep_recent,
            total_tokens=cumulative_tokens,
        ):
            if isinstance(evt, CompactionStart):
                for m in self._middleware:
                    await m.on_compaction_start(evt)
            elif isinstance(evt, CompactionEnd):
                for m in self._middleware:
                    await m.on_compaction_end(evt)
                compaction_error = evt.error
                compaction_summary = evt.summary
                compaction_meta = {
                    "dropped_count": evt.dropped_message_count,
                    "keep_recent": self._keep_recent,
                    "new_offset": evt.new_offset,
                    "duration": evt.duration,
                }
            yield evt

        if compaction_error is not None:
            return
            yield  # pragma: no cover

        memory.reset_message_usage()

        from minimal_harness.types import MessageEvent

        if self._emit_message_events and compaction_summary:
            yield MessageEvent(
                message={
                    "role": "compaction",
                    "content": compaction_summary,
                    "meta": compaction_meta,
                }
            )


__all__ = ["ToolCompactionAgent"]

"""Tool-compaction agent loop — compresses tool results when their
estimated token count exceeds a configured threshold.

This agent extends :class:`BaseAgent` with two hooks:

1. :meth:`_post_tool_execution` — runs *after* tool execution but
   *before* the next LLM call in the same iteration. If the tool
   messages in the forward buffer exceed *tool_token_threshold*
   (estimated as ``len(content) // 2``), they are folded into a
   single ``role="tool"`` summary message. This prevents the next
   LLM prompt from bloating past the context limit.

2. :meth:`_post_llm_response` — runs at the end of each round
   (after the LLM has responded). If *round_compress* is enabled,
   any remaining uncompressed tool messages are folded. This keeps
   the session compact for future rounds.

Tool results are compressed *after* the LLM has consumed them, so
the assistant's reasoning is unaffected. The original tool messages
are preserved in the replay history for session export / debugging.

Event stream
------------
Both hooks reuse the existing ``CompactionStart / CompactionChunk /
CompactionEnd`` event types so the front-end can render the progress
without changes. The ``CompactionEnd.meta`` carries
``{"compressed": True, "dropped_count": N}``.
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
    """Agent loop with automatic tool-result compression.

    Parameters
    ----------
    llm_provider : LLMProvider
        The LLM backend.
    summarizer : Callable
        Streaming summariser that takes ``(list[Message], existing_summary)``
        and yields summary text chunks.
    tool_token_threshold : int
        Estimated token count threshold. When the combined content length
        of ``role="tool"`` messages in the forward buffer exceeds this
        (``len(content) // 2``), compression triggers.
    round_compress : bool
        If ``True``, also compress tool messages at the end of each round
        (in :meth:`_post_llm_response`). Default ``True``.
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
        tool_token_threshold: int,
        round_compress: bool = True,
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
        self._tool_token_threshold = tool_token_threshold
        self._round_compress = round_compress

    async def _post_tool_execution(
        self,
        memory: Memory,
    ) -> AsyncIterator[AgentEvent]:
        """Within-round compression: compress tool results if they exceed
        *tool_token_threshold*, preventing the next LLM call from seeing
        an oversized prompt.

        This hook runs after :meth:`BaseAgent._execute_tools` completes
        and before the loop returns to the LLM for the next iteration.
        """
        if self._tool_token_threshold <= 0:
            return
            yield  # pragma: no cover

        async for evt in memory.compress_tool_messages(
            self._summarizer,
            self._tool_token_threshold,
        ):
            if isinstance(evt, CompactionStart):
                for m in self._middleware:
                    await m.on_compaction_start(evt)
            elif isinstance(evt, CompactionEnd):
                for m in self._middleware:
                    await m.on_compaction_end(evt)
            yield evt

    async def _post_llm_response(
        self,
        llm_response: Any,
        memory: Memory,
    ) -> AsyncIterator[AgentEvent]:
        """End-of-round compression: fold remaining uncompressed tool
        messages so the session stays compact for future rounds.

        Only runs when *round_compress* is ``True``.
        """
        if not self._round_compress:
            return
            yield  # pragma: no cover

        async for evt in memory.compress_tool_messages(
            self._summarizer,
            tool_token_threshold=0,
        ):
            if isinstance(evt, CompactionStart):
                for m in self._middleware:
                    await m.on_compaction_start(evt)
            elif isinstance(evt, CompactionEnd):
                for m in self._middleware:
                    await m.on_compaction_end(evt)
            yield evt


__all__ = ["ToolCompactionAgent"]

"""Tool-compaction agent loop — compresses tool results at the end of
each round and optionally compacts the full conversation history when
prompt-token usage exceeds a configured threshold.

The single hook :meth:`_post_llm_response` runs after the LLM has
responded:

1. If *round_compress* is enabled, any uncompressed ``role="tool"``
   messages from the round are summarised into a single tool message.
2. If the cumulative prompt-token count exceeds
   *prompt_token_threshold*, the entire conversation history is
   compacted (older messages folded into a summary, keeping the
   *keep_recent* most recent messages verbatim).

The original tool results are preserved in the replay history for
session export / debugging.

Event stream
------------
Reuses the existing ``CompactionStart / CompactionChunk /
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
    round_compress : bool
        If ``True``, compress tool messages at the end of each round.
        Default ``True``.
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
        round_compress: bool = True,
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
        self._round_compress = round_compress
        self._prompt_token_threshold = prompt_token_threshold
        self._keep_recent = keep_recent

    async def _post_llm_response(
        self,
        llm_response: Any,
        memory: Memory,
    ) -> AsyncIterator[AgentEvent]:
        """End-of-round housekeeping:

        1. Tool compression — fold uncompressed tool messages (gated by
           *round_compress*).
        2. Full conversation compaction — when cumulative prompt-token
           count exceeds *prompt_token_threshold*, compact older messages
           into a summary, keeping *keep_recent* most recent verbatim.
        """
        # ── 1. Tool compression ──
        if self._round_compress:
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

        # ── 2. Full conversation compaction ──
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

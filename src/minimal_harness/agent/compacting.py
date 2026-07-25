"""Compaction agent loop — folds older messages into a summary when
the LLM's prompt-token usage exceeds a configured threshold.

This is structurally identical to :class:`SimpleAgent` except for one
overridden hook (:meth:`CompactionAgent._post_llm_response`) that runs
:meth:`Memory.compact` after the LLM turn completes. The shared
agentic loop, tool execution, error handling, and event emission all
live in :class:`BaseAgent` — see that module for the lifecycle.

On a successful fold the agent emits the same
``CompactionStart / CompactionChunk / CompactionEnd`` event stream
that the rest of the SDK sees, plus a trailing
``MessageEvent(role="compaction")`` so persistence layers can write
the synthetic summary into the session log.

On a failed fold (e.g. summarizer raised) the agent records the
LLM's reply, emits ``CompactionEnd(error=...)`` so the front-end can
render the failure, and continues to the next iteration. This is a
deliberate change from the original "raise and end the run"
behaviour: the assistant turn is the primary content of the turn and
must reach the user even if housekeeping fails. The next turn will
retry compaction on the same unchanged buffer.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Callable, Sequence

from minimal_harness.llm.llm import LLMProvider
from minimal_harness.memory import (
    Memory,
    Message,
)
from minimal_harness.types import (
    AgentEvent,
    CompactionEnd,
    CompactionStart,
    MessageEvent,
)

from .base import BaseAgent
from .middleware import Middleware
from .protocol import InputContentConversionFunction

logger = logging.getLogger(__name__)


class CompactionAgent(BaseAgent):
    """Agent loop with automatic context compaction.

    Compaction runs after each LLM response in ``_post_llm_response``,
    so it may trigger mid-turn when tool-call rounds accumulate
    enough tokens.  Use ``ToolCompactionAgent`` if you need per-round
    tool-call stripping BEFORE compaction.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        summarizer: Callable[[list[Message], str | None], AsyncIterator[str]],
        prompt_token_threshold: int,
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
        cumulative_tokens = memory.get_message_usage().get("total_tokens", 0)
        if cumulative_tokens <= self._prompt_token_threshold:
            return
            yield

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
            logger.warning(
                "agent.compaction.soft-fail threshold=%d tokens=%d error=%s",
                self._prompt_token_threshold,
                cumulative_tokens,
                compaction_error,
            )
            return
            yield

        memory.reset_message_usage()

        if self._emit_message_events and compaction_summary:
            yield MessageEvent(
                message={
                    "role": "compaction",
                    "content": compaction_summary,
                    "meta": compaction_meta,
                }
            )


__all__ = ["CompactionAgent"]

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

from minimal_harness.agent._compaction import (
    compute_effective_keep_recent,
    estimate_prompt_tokens,
    should_fold,
    soft_limit_threshold,
)
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

    RFC #57 additions (all default to the pre-existing behaviour):

    - ``soft_limit_ratio`` + ``max_context_tokens`` (positive) enable the
      OR'd soft-limit trigger; ``estimate_leading_edge`` adds the
      byte→token pre-flight that fires via ``_pre_llm_hook`` BEFORE an
      LLM call, so a buffer already over the upstream limit folds
      instead of 400-looping.
    - ``anchor_keep_recent_on`` re-anchors the preserved tail
      (``"last_tool_round"`` keeps the live tool round verbatim),
      computed via :func:`compute_effective_keep_recent`.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        summarizer: Callable[[list[Message], str | None], AsyncIterator[str]],
        prompt_token_threshold: int,
        keep_recent: int = 0,
        max_iterations: int = 2000,
        custom_input_conversion: InputContentConversionFunction | None = None,
        middleware: Sequence[Middleware] = (),
        emit_message_events: bool = True,
        soft_limit_ratio: float = 0.0,
        max_context_tokens: int = 0,
        estimate_leading_edge: bool = True,
        anchor_keep_recent_on: str = "last_tool_round",
        max_tool_rounds: int | None = None,
        emit_delta_events: bool = False,
        tool_result_trimmer=None,
    ):
        super().__init__(
            llm_provider=llm_provider,
            max_iterations=max_iterations,
            custom_input_conversion=custom_input_conversion,
            middleware=middleware,
            emit_message_events=emit_message_events,
            max_tool_rounds=max_tool_rounds,
            emit_delta_events=emit_delta_events,
            tool_result_trimmer=tool_result_trimmer,
        )
        self._summarizer = summarizer
        self._prompt_token_threshold = prompt_token_threshold
        self._keep_recent = keep_recent
        self._soft_limit_ratio = soft_limit_ratio
        self._max_context_tokens = max_context_tokens
        self._estimate_leading_edge = estimate_leading_edge
        self._anchor_keep_recent_on = anchor_keep_recent_on

    def _effective_keep_recent(self, memory: Memory) -> int:
        return compute_effective_keep_recent(
            memory, self._keep_recent, self._anchor_keep_recent_on
        )

    async def _pre_llm_hook(
        self,
        memory: Memory,
    ) -> AsyncIterator[AgentEvent]:
        """RFC #57 leading-edge trigger, evaluated before EVERY LLM call.

        When the soft-limit is enabled, a buffer whose byte-estimated
        prompt size (or provider-reported usage) already exceeds
        ``int(max_context_tokens * soft_limit_ratio)`` is folded here —
        before the request is sent, so an oversized turn can never
        400-loop on ``context_length_exceeded`` and retry forever.
        The accurate post-hoc threshold keeps its existing role as the
        trailing backstop.
        """
        soft_threshold = soft_limit_threshold(
            self._soft_limit_ratio, self._max_context_tokens
        )
        if soft_threshold <= 0:
            return
            yield  # Make this an async generator.
        usage = int(memory.get_message_usage().get("total_tokens", 0)) or 0
        est = estimate_prompt_tokens(memory) if self._estimate_leading_edge else 0
        if usage <= soft_threshold and est <= soft_threshold:
            return
            yield  # Make this an async generator.
        async for evt in self._run_compaction(memory, total_tokens=usage):
            yield evt

    async def _post_llm_response(
        self,
        llm_response: Any,
        memory: Memory,
    ) -> AsyncIterator[AgentEvent]:
        fold, cumulative = should_fold(
            memory,
            self._prompt_token_threshold,
            soft_limit_ratio=self._soft_limit_ratio,
            max_context_tokens=self._max_context_tokens,
            estimate_leading_edge=self._estimate_leading_edge,
        )
        if not fold:
            return
            yield
        async for evt in self._run_compaction(memory, total_tokens=cumulative):
            yield evt

    async def _run_compaction(
        self,
        memory: Memory,
        *,
        total_tokens: int,
    ) -> AsyncIterator[AgentEvent]:
        compaction_error: str | None = None
        compaction_summary: str = ""
        compaction_meta: dict[str, Any] = {}
        effective_keep_recent = self._effective_keep_recent(memory)

        async for evt in memory.compact(
            self._summarizer,
            effective_keep_recent,
            total_tokens=total_tokens,
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
                    "keep_recent": effective_keep_recent,
                    "new_offset": evt.new_offset,
                    "duration": evt.duration,
                }
            yield evt

        if compaction_error is not None:
            logger.warning(
                "agent.compaction.soft-fail threshold=%d tokens=%d error=%s",
                self._prompt_token_threshold,
                total_tokens,
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

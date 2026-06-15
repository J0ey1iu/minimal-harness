"""Simple agent loop — no context compaction, just LLM ⇄ tools.

This is the default :class:`minimal_harness.agent.protocol.Agent`
implementation. It inherits the shared agentic loop from
:class:`BaseAgent` and only customises the constructor signature
(other agent types add extra dependencies). The post-LLM hook
:meth:`_post_llm_response` is a no-op, so the buffer is never folded
during a run.
"""

from __future__ import annotations

from typing import Sequence

from minimal_harness.llm.llm import LLMProvider

from .base import BaseAgent
from .middleware import Middleware
from .protocol import InputContentConversionFunction


class SimpleAgent(BaseAgent):
    def __init__(
        self,
        llm_provider: LLMProvider,
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


__all__ = ["SimpleAgent"]

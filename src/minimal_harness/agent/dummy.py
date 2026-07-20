"""Dummy agent loop — echoes user input back as the assistant response.

This agent does NOT call any LLM. It reads the user's text content and
returns it verbatim. Useful for testing agent-type switching, the event
pipeline, and front-end rendering without incurring LLM costs.

The constructor signature mirrors :class:`SimpleAgent` (same parameters),
and the ``run()`` method emits the same event sequence
(``AgentStart / LLMStart / LLMChunk / LLMEnd / MessageEvent / AgentEnd``)
so that downstream consumers (persistence, SSE, front-end) do not need
special handling.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator, Iterable, Sequence

from minimal_harness.llm.llm import LLMProvider
from minimal_harness.memory import (
    ExtendedInputContentPart,
    Memory,
    assistant_message,
    user_message,
)
from minimal_harness.tool.base import Tool
from minimal_harness.types import (
    AgentEnd,
    AgentEvent,
    AgentStart,
    LLMChunk,
    LLMChunkDelta,
    LLMEnd,
    LLMStart,
    MessageEvent,
)

from .base import BaseAgent
from .middleware import Middleware
from .protocol import InputContentConversionFunction

logger = logging.getLogger(__name__)


class DummyAgent(BaseAgent):
    """Agent that echoes user input back as the assistant response."""

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

    def run(
        self,
        user_input: Iterable[ExtendedInputContentPart],
        stop_event: asyncio.Event | None = None,
        memory: Memory | None = None,
        tools: Sequence[Tool] | None = None,
        system_prompt: str = "",
        context: dict[str, Any] | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AgentEvent]:
        assert memory is not None, "memory must be provided"

        async def agen() -> AsyncIterator[AgentEvent]:
            for m in self._middleware:
                await m.on_agent_start(user_input)
            yield AgentStart(user_input)
            start_time = time.time()

            converted_user_input = list(user_input)
            if self._custom_input_conversion:
                converted_user_input = list(
                    await self._custom_input_conversion(converted_user_input)
                )
            await memory.add_message(user_message(converted_user_input))

            # Extract text from user input parts
            text_parts: list[str] = []
            for part in converted_user_input:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    text_parts.append(part)
            echo_text = "\n".join(text_parts).strip()

            # Informational message list for LLMStart event
            llm_messages: list = memory.get_forward_messages()
            if system_prompt:
                llm_messages = [
                    {"role": "system", "content": system_prompt}
                ] + llm_messages

            yield LLMStart(messages=llm_messages, tools=[])

            if stop_event and stop_event.is_set():
                yield LLMEnd(
                    content="",
                    reasoning_content=None,
                    tool_calls=[],
                    usage=None,
                    error="Interrupted",
                )
                yield AgentEnd("", time.time() - start_time, interrupted=True)
                return

            yield LLMChunk(chunk=LLMChunkDelta(content=echo_text))

            yield LLMEnd(
                content=echo_text,
                reasoning_content=None,
                tool_calls=[],
                usage=None,
            )

            await memory.add_message(assistant_message(echo_text))
            if self._emit_message_events:
                yield MessageEvent(message={"role": "assistant", "content": echo_text})

            yield AgentEnd(echo_text, time.time() - start_time)

        return agen()


__all__ = ["DummyAgent"]

import asyncio
from typing import AsyncIterator, Sequence

from openai import AsyncOpenAI

from minimal_harness.llm import (
    ChunkCallback,
    LLMResponse,
    Stream,
)
from minimal_harness.memory import Message
from minimal_harness.settings import Settings
from minimal_harness.tool.base import StreamingTool
from minimal_harness.types import (
    LLMChunkDelta,
    TokenUsage,
    ToolCall,
    ToolCallDelta,
    ToolCallFunction,
)


def _normalize_chunk(chunk) -> LLMChunkDelta | None:
    """Convert an OpenAI streaming chunk into a provider-agnostic delta."""
    if not chunk.choices:
        return None
    delta = chunk.choices[0].delta
    if delta is None:
        return None

    content = delta.content or None
    reasoning = getattr(delta, "reasoning_content", None) or None
    tool_call_deltas: list[ToolCallDelta] | None = None

    if delta.tool_calls:
        tool_call_deltas = []
        for tc in delta.tool_calls:
            tool_call_deltas.append(
                ToolCallDelta(
                    index=tc.index,
                    id=tc.id or None,
                    name=tc.function.name or None if tc.function else None,
                    arguments=tc.function.arguments or None if tc.function else None,
                )
            )

    if content is None and reasoning is None and tool_call_deltas is None:
        return None

    return LLMChunkDelta(
        content=content,
        reasoning=reasoning,
        tool_calls=tool_call_deltas,
    )


class OpenAILLMProvider:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str | None = None,
        on_chunk: ChunkCallback[LLMChunkDelta] | None = None,
    ):
        self._client = client
        self._model = model if model is not None else Settings.model()
        self._on_chunk = on_chunk

    async def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[StreamingTool],
        stop_event: asyncio.Event | None = None,
    ) -> Stream[LLMChunkDelta]:
        agen = self._chat(messages, tools, stop_event)
        return Stream(agen)

    async def _chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[StreamingTool],
        stop_event: asyncio.Event | None = None,
    ) -> AsyncIterator[LLMChunkDelta | LLMResponse]:
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            tools=[t.to_schema() for t in tools],  # type: ignore[arg-type]
            tool_choice="auto" if tools else "none",
            stream=True,
        )

        content_parts = []
        reasoning_parts = []
        tool_calls_acc: dict[int, ToolCall] = {}
        finish_reason = None
        usage: TokenUsage | None = None

        try:
            async with stream:
                async for raw_chunk in stream:
                    if stop_event and stop_event.is_set():
                        break

                    if getattr(raw_chunk, "usage") and raw_chunk.usage:
                        usage = {
                            "prompt_tokens": raw_chunk.usage.prompt_tokens,
                            "completion_tokens": raw_chunk.usage.completion_tokens,
                            "total_tokens": raw_chunk.usage.total_tokens,
                        }

                    delta = raw_chunk.choices[0].delta if raw_chunk.choices else None

                    if raw_chunk.choices and raw_chunk.choices[0].finish_reason:
                        finish_reason = raw_chunk.choices[0].finish_reason

                    if delta is None:
                        continue

                    if delta.content:
                        content_parts.append(delta.content)

                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning:
                        reasoning_parts.append(reasoning)

                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in tool_calls_acc:
                                tool_calls_acc[idx] = ToolCall(
                                    id="",
                                    type="function",
                                    function=ToolCallFunction(name="", arguments=""),
                                )
                            acc = tool_calls_acc[idx]
                            if tc_delta.id:
                                acc["id"] += tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    acc["function"]["name"] += tc_delta.function.name
                                if tc_delta.function.arguments:
                                    acc["function"]["arguments"] += (
                                        tc_delta.function.arguments
                                    )

                    normalized = _normalize_chunk(raw_chunk)
                    if normalized is not None:
                        if self._on_chunk:
                            await self._on_chunk(normalized, False)
                        yield normalized
        except asyncio.CancelledError:
            if self._on_chunk:
                await self._on_chunk(None, True)
            raise

        if self._on_chunk:
            await self._on_chunk(None, True)

        yield LLMResponse(
            content="".join(content_parts) or None,
            reasoning_content="".join(reasoning_parts) or None,
            tool_calls=list(tool_calls_acc.values()) if tool_calls_acc else [],
            finish_reason=finish_reason,
            usage=usage,
        )

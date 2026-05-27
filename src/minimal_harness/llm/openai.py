import asyncio
import logging
from typing import Any, AsyncIterator, Sequence

from openai import AsyncOpenAI

from minimal_harness.llm.llm import (
    LLMResponse,
    Stream,
    await_with_interrupt,
)
from minimal_harness.memory import Message
from minimal_harness.tool.base import Tool
from minimal_harness.types import (
    ExtraHeadersProvider,
    LLMChunkDelta,
    TokenUsage,
    ToolCall,
    ToolCallDelta,
    ToolCallFunction,
)

logger = logging.getLogger(__name__)


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


def _convert_messages(
    messages: Sequence[Message],
) -> list[dict[str, Any]]:
    """Convert unified messages to OpenAI native format."""
    openai_messages: list[dict[str, Any]] = []

    for msg in messages:
        if msg["role"] == "system":
            openai_messages.append({"role": "system", "content": msg["content"]})
        elif msg["role"] == "user":
            content: list[dict[str, Any]] = []
            for part in msg["content"]:
                if part["type"] == "text":
                    content.append({"type": "text", "text": part["text"]})
                elif part["type"] == "image":
                    url = part["url"]
                    data = part.get("data")
                    media_type = part.get("media_type")
                    if data is not None and media_type is not None:
                        url = f"data:{media_type};base64,{data}"
                    content.append({"type": "image_url", "image_url": {"url": url}})
                elif part["type"] == "file":
                    content.append(
                        {
                            "type": "text",
                            "text": f"[File: {part['file']['file_name']}]",
                        }
                    )
            openai_messages.append({"role": "user", "content": content})
        elif msg["role"] == "assistant":
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if msg.get("content"):
                assistant_msg["content"] = msg["content"]
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    }
                    for tc in tool_calls
                ]
            openai_messages.append(assistant_msg)
        elif msg["role"] == "tool":
            openai_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": msg["tool_call_id"],
                    "content": msg["content"],
                }
            )

    return openai_messages


class OpenAILLMProvider:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        llm_extra_headers_provider: ExtraHeadersProvider | None = None,
    ):
        self._client = client
        self._model = model
        self._llm_extra_headers_provider = llm_extra_headers_provider

    async def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool],
        stop_event: asyncio.Event | None = None,
        **kwargs: Any,
    ) -> Stream[LLMChunkDelta]:
        agen = self._chat(messages, tools, stop_event, **kwargs)
        return Stream(agen)

    async def _chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool],
        stop_event: asyncio.Event | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[LLMChunkDelta | LLMResponse]:
        openai_messages = _convert_messages(messages)
        timeout = kwargs.pop("timeout", 120)

        extra_headers = dict(kwargs.pop("extra_headers", {}))
        if self._llm_extra_headers_provider is not None:
            extra_headers.update(await self._llm_extra_headers_provider())

        tool_count = len(tools)
        msg_count = len(openai_messages)
        logger.debug(
            "OUTBOUND LLM call — model=%s msg_count=%d tool_count=%d timeout=%d",
            self._model,
            msg_count,
            tool_count,
            timeout,
        )
        stream = await await_with_interrupt(
            self._client.chat.completions.create(
                model=self._model,
                messages=openai_messages,  # type: ignore[arg-type]
                tools=[t.to_schema() for t in tools],  # type: ignore[arg-type]
                tool_choice="auto" if tools else "none",
                stream=True,
                timeout=timeout,
                extra_headers=extra_headers if extra_headers else None,
                **kwargs,
            ),
            stop_event,
        )

        content_parts = []
        reasoning_parts = []
        tool_calls_acc: dict[int, ToolCall] = {}
        finish_reason = None
        usage: TokenUsage | None = None

        try:
            async with stream:
                async for raw_chunk in stream:
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
                                acc["id"] = tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    acc["function"]["name"] = tc_delta.function.name
                                if tc_delta.function.arguments:
                                    acc["function"]["arguments"] += (
                                        tc_delta.function.arguments
                                    )

                    normalized = _normalize_chunk(raw_chunk)
                    if normalized is not None:
                        yield normalized
        except asyncio.CancelledError:
            raise

        yield LLMResponse(
            content="".join(content_parts) or None,
            reasoning_content="".join(reasoning_parts) or None,
            tool_calls=list(tool_calls_acc.values()) if tool_calls_acc else [],
            finish_reason=finish_reason,
            usage=usage,
        )

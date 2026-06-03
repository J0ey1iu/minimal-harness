import asyncio
import json
import logging
from typing import Any, AsyncIterator, Sequence

from anthropic import AsyncAnthropic
from anthropic.types import (
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStopEvent,
    TextDelta,
    ToolUseBlock,
)

from minimal_harness.llm.llm import LLMResponse, Stream, await_with_interrupt
from minimal_harness.memory import (
    Message,
)
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


def _convert_messages(
    messages: Sequence[Message],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert unified messages to Anthropic format.

    Returns ``(system_prompt, anthropic_messages)``.  Anthropic expects
    the system prompt as a top-level parameter rather than a message.
    """
    system_prompt: str | None = None
    anthropic_messages: list[dict[str, Any]] = []

    for msg in messages:
        if msg["role"] == "system":
            system_prompt = msg["content"]
        elif msg["role"] == "user":
            content: list[dict[str, Any]] = []
            for part in msg["content"]:
                if part["type"] == "text":
                    content.append({"type": "text", "text": part["text"]})
                elif part["type"] == "image":
                    data = part.get("data")
                    media_type = part.get("media_type")
                    if data is not None and media_type is not None:
                        content.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": data,
                                },
                            }
                        )
                    else:
                        content.append(
                            {"type": "text", "text": f"[Image: {part['url']}]"}
                        )
                elif part["type"] == "file":
                    content.append(
                        {
                            "type": "text",
                            "text": f"[File: {part['file']['file_name']}]",
                        }
                    )
            anthropic_messages.append({"role": "user", "content": content})
        elif msg["role"] == "assistant":
            content_blocks: list[dict[str, Any]] = []
            if msg.get("content"):
                content_blocks.append({"type": "text", "text": msg["content"]})
            tool_calls = msg.get("tool_calls")
            if tool_calls is not None:
                for tc in tool_calls:
                    raw_args = tc["function"]["arguments"]
                    try:
                        input_obj = json.loads(raw_args) if raw_args else {}
                    except json.JSONDecodeError:
                        logger.warning("tool.args.parse.error raw_args=%s", raw_args)
                        input_obj = {"raw_args": raw_args} if raw_args else {}
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["function"]["name"],
                            "input": input_obj,
                        }
                    )
            anthropic_messages.append({"role": "assistant", "content": content_blocks})
        elif msg["role"] == "tool":
            anthropic_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg["tool_call_id"],
                            "content": msg["content"],
                        }
                    ],
                }
            )

    return system_prompt, anthropic_messages


def _normalize_event(event) -> LLMChunkDelta | None:
    """Convert an Anthropic streaming event into a provider-agnostic delta."""
    if isinstance(event, ContentBlockStartEvent):
        block = event.content_block
        if block.type == "tool_use" and isinstance(block, ToolUseBlock):
            return LLMChunkDelta(
                tool_calls=[
                    ToolCallDelta(
                        index=event.index,
                        id=block.id,
                        name=block.name,
                    )
                ]
            )
        return None
    elif isinstance(event, ContentBlockDeltaEvent):
        delta = event.delta
        if isinstance(delta, TextDelta):
            return LLMChunkDelta(content=delta.text)
        elif delta.type == "input_json_delta":
            return LLMChunkDelta(
                tool_calls=[
                    ToolCallDelta(
                        index=event.index,
                        arguments=delta.partial_json,
                    )
                ]
            )
        return None
    return None


class AnthropicLLMProvider:
    """Anthropic-compatible LLM provider.

    Converts the project's unified :class:`~minimal_harness.memory.Message`
    types into Anthropic's native format and maps the streaming events back
    into the provider-agnostic :class:`~minimal_harness.llm.LLMResponse`.
    """

    def __init__(
        self,
        client: AsyncAnthropic,
        model: str,
        max_tokens: int = 4096,
        llm_extra_headers_provider: ExtraHeadersProvider | None = None,
    ):
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
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
        system_prompt, anthropic_messages = _convert_messages(messages)
        anthropic_tools = [t.to_anthropic_schema() for t in tools] if tools else []

        extra_headers = dict(kwargs.pop("extra_headers", {}))
        if self._llm_extra_headers_provider is not None:
            extra_headers.update(await self._llm_extra_headers_provider())

        request_kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": anthropic_messages,
            "stream": True,
        }
        if system_prompt is not None:
            request_kwargs["system"] = system_prompt
        if anthropic_tools:
            request_kwargs["tools"] = anthropic_tools
        if extra_headers:
            request_kwargs["extra_headers"] = extra_headers
        request_kwargs.update(kwargs)

        logger.info("llm.chat.connect.start model=%s", self._model)
        try:
            stream = await await_with_interrupt(
                self._client.messages.create(**request_kwargs),
                stop_event,
            )
        except Exception:
            logger.exception(
                "llm.chat.connect.error model=%s",
                self._model,
            )
            raise
        logger.info("llm.chat.connect.end model=%s", self._model)

        content_parts: list[str] = []
        tool_calls_acc: dict[int, ToolCall] = {}
        finish_reason: str | None = None
        usage: TokenUsage | None = None

        try:
            async with stream:
                async for event in stream:
                    if isinstance(event, MessageStartEvent):
                        if event.message.usage:
                            usage = {
                                "prompt_tokens": event.message.usage.input_tokens,
                                "completion_tokens": 0,
                                "total_tokens": event.message.usage.input_tokens,
                            }
                    elif isinstance(event, ContentBlockStartEvent):
                        block = event.content_block
                        if block.type == "tool_use" and isinstance(block, ToolUseBlock):
                            tool_calls_acc[event.index] = ToolCall(
                                id=block.id,
                                type="function",
                                function=ToolCallFunction(
                                    name=block.name, arguments=""
                                ),
                            )
                    elif isinstance(event, ContentBlockDeltaEvent):
                        delta = event.delta
                        if isinstance(delta, TextDelta):
                            content_parts.append(delta.text)
                        elif delta.type == "input_json_delta":
                            tc = tool_calls_acc.get(event.index)
                            if tc is not None:
                                tc["function"]["arguments"] += delta.partial_json
                    elif isinstance(event, MessageDeltaEvent):
                        if event.delta.stop_reason:
                            finish_reason = event.delta.stop_reason
                        if event.usage and usage is not None:
                            usage["completion_tokens"] = event.usage.output_tokens
                            usage["total_tokens"] = (
                                usage["prompt_tokens"] + event.usage.output_tokens
                            )
                    elif isinstance(event, MessageStopEvent):
                        pass

                    normalized = _normalize_event(event)
                    if normalized is not None:
                        yield normalized
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "llm.chat.stream.error model=%s content_parts=%d tool_calls=%d",
                self._model,
                len(content_parts),
                len(tool_calls_acc),
            )
            raise

        yield LLMResponse(
            content="".join(content_parts) or None,
            reasoning_content=None,
            tool_calls=list(tool_calls_acc.values()),
            finish_reason=finish_reason,
            usage=usage,
        )

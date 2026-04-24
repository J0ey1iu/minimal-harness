import asyncio
import json
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

from minimal_harness.llm.llm import ChunkCallback, LLMResponse, Stream
from minimal_harness.memory import (
    Message,
)
from minimal_harness.settings import Settings
from minimal_harness.tool.base import StreamingTool
from minimal_harness.types import TokenUsage, ToolCall, ToolCallFunction


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
                    content.append({"type": "text", "text": f"[Image: {part['url']}]"})
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
                        input_obj = {}
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


class AnthropicLLMProvider:
    """Anthropic-compatible LLM provider.

    Converts the project's unified :class:`~minimal_harness.memory.Message`
    types into Anthropic's native format and maps the streaming events back
    into the provider-agnostic :class:`~minimal_harness.llm.LLMResponse`.
    """

    def __init__(
        self,
        client: AsyncAnthropic,
        model: str | None = None,
        max_tokens: int = 4096,
        on_chunk: ChunkCallback[Any] | None = None,
    ):
        self._client = client
        self._model = model if model is not None else Settings.model()
        self._max_tokens = max_tokens
        self._on_chunk = on_chunk

    async def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[StreamingTool],
        stop_event: asyncio.Event | None = None,
    ) -> Stream[Any | LLMResponse]:
        agen = self._chat(messages, tools, stop_event)
        return Stream(agen)

    async def _chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[StreamingTool],
        stop_event: asyncio.Event | None = None,
    ) -> AsyncIterator[Any | LLMResponse]:
        system_prompt, anthropic_messages = _convert_messages(messages)
        anthropic_tools = [t.to_anthropic_schema() for t in tools] if tools else []

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": anthropic_messages,
            "stream": True,
        }
        if system_prompt is not None:
            kwargs["system"] = system_prompt
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        stream = await self._client.messages.create(**kwargs)

        content_parts: list[str] = []
        tool_calls_acc: dict[int, ToolCall] = {}
        finish_reason: str | None = None
        usage: TokenUsage | None = None

        try:
            async with stream:
                async for event in stream:
                    if stop_event and stop_event.is_set():
                        break

                    if self._on_chunk:
                        await self._on_chunk(event, False)

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

                    yield event
        except asyncio.CancelledError:
            if self._on_chunk:
                await self._on_chunk(None, True)
            raise

        if self._on_chunk:
            await self._on_chunk(None, True)

        yield LLMResponse(
            content="".join(content_parts) or None,
            tool_calls=list(tool_calls_acc.values()),
            finish_reason=finish_reason,
            usage=usage,
        )

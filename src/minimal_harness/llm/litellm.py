from __future__ import annotations

from typing import TYPE_CHECKING, AsyncIterator

if TYPE_CHECKING:
    from litellm.types.utils import ModelResponseStream

from minimal_harness.llm import (
    ChunkCallback,
    LLMResponse,
    Stream,
    TokenUsage,
    ToolCall,
    ToolCallFunction,
)
from minimal_harness.memory import Message
from minimal_harness.tool import Tool


class LiteLLMProvider:
    def __init__(self, base_url: str, api_key: str, model: str = "openai/qwen3.5-27b"):
        self._model = model
        self._base_url = base_url
        self._api_key: str = api_key

    async def chat(
        self,
        messages: list[Message],
        tools: list[Tool],
        on_chunk: ChunkCallback[ModelResponseStream] | None,
    ) -> Stream[ModelResponseStream | LLMResponse]:
        agen = self._chat(messages, tools, on_chunk)
        return Stream(agen)

    async def _chat(
        self,
        messages: list[Message],
        tools: list[Tool],
        on_chunk: ChunkCallback[ModelResponseStream] | None,
    ) -> AsyncIterator[ModelResponseStream | LLMResponse]:
        import litellm

        litellm.drop_params = True
        stream = await litellm.acompletion(
            model=self._model,
            messages=messages,
            tools=[t.to_schema() for t in tools],
            tool_choice="auto" if tools else "none",
            stream=True,
            stream_options={"include_usage": True},
            base_url=self._base_url,
            api_key=self._api_key,
        )

        content_parts = []
        tool_calls_acc: dict[int, ToolCall] = {}
        finish_reason = None
        usage: TokenUsage | None = None

        if not isinstance(stream, litellm.CustomStreamWrapper):
            raise Exception("Expected a CustomStreamWrapper")

        async for chunk in stream:
            if on_chunk:
                await on_chunk(chunk, False)

            delta = chunk.choices[0].delta if chunk.choices else None

            if hasattr(chunk, "usage") and chunk.usage:  # type: ignore[attr-defined]
                usage = {
                    "prompt_tokens": chunk.usage.prompt_tokens,  # type: ignore[attr-defined]
                    "completion_tokens": chunk.usage.completion_tokens,  # type: ignore[attr-defined]
                    "total_tokens": chunk.usage.total_tokens,  # type: ignore[attr-defined]
                }

            if delta is None:
                continue

            if delta.content:
                content_parts.append(delta.content)

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
                            acc["function"]["arguments"] += tc_delta.function.arguments

            if chunk.choices and chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

            yield chunk

        if on_chunk:
            await on_chunk(None, True)

        yield LLMResponse(
            content="".join(content_parts) or None,
            tool_calls=list(tool_calls_acc.values()) if tool_calls_acc else [],
            finish_reason=finish_reason,
            usage=usage,
        )

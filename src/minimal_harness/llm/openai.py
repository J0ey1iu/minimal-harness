from typing import AsyncIterator

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk

from minimal_harness.llm import (
    ChunkCallback,
    LLMResponse,
    Stream,
    ToolCall,
    ToolCallFunction,
)
from minimal_harness.memory import Message
from minimal_harness.tool import Tool


class OpenAILLMProvider:
    def __init__(self, client: AsyncOpenAI, model: str = "qwen3.5-27b"):
        self._client = client
        self._model = model

    async def chat(
        self,
        messages: list[Message],
        tools: list[Tool],
        on_chunk: ChunkCallback | None,
    ) -> Stream[ChatCompletionChunk | LLMResponse]:
        agen = self._chat(messages, tools, on_chunk)
        return Stream(agen)

    async def _chat(
        self,
        messages: list[Message],
        tools: list[Tool],
        on_chunk: ChunkCallback | None,
    ) -> AsyncIterator[ChatCompletionChunk | LLMResponse]:
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            tools=[t.to_schema() for t in tools],
            tool_choice="auto" if tools else "none",
            stream=True,
        )

        content_parts = []
        tool_calls_acc: dict[int, ToolCall] = {}
        finish_reason = None

        async for chunk in stream:
            if on_chunk:
                await on_chunk(chunk, False)

            delta = chunk.choices[0].delta if chunk.choices else None

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
        )

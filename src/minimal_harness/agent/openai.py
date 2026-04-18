import asyncio
import json
from typing import Any, Iterable, Sequence, cast

from minimal_harness.llm.openai import OpenAILLMProvider
from minimal_harness.memory import (
    ConversationMemory,
    ExtendedInputContentPart,
    Memory,
    Message,
    UserMessage,
)
from minimal_harness.tool.base import StreamingTool
from minimal_harness.types import (
    ChunkCallback,
    ExecutionStartCallback,
    ProgressCallback,
    ToolCall,
    ToolEndCallback,
    ToolStartCallback,
)

from .protocol import InputContentConversionFunction


class OpenAIAgent:
    def __init__(
        self,
        llm_provider: OpenAILLMProvider,
        tools: Sequence[StreamingTool] | None = None,
        max_iterations: int = 50,
        memory: Memory | None = None,
        on_tool_start: ToolStartCallback | None = None,
        on_tool_end: ToolEndCallback | None = None,
        on_execution_start: ExecutionStartCallback | None = None,
    ):
        self._llm_provider = llm_provider
        self._tools: dict[str, StreamingTool] = {t.name: t for t in (tools or [])}
        self._on_tool_start = on_tool_start
        self._on_tool_end = on_tool_end
        self._on_execution_start = on_execution_start
        self._max_iterations = max_iterations
        self._memory = memory or ConversationMemory()

    async def run(
        self,
        user_input: Iterable[ExtendedInputContentPart],
        custom_input_conversion: InputContentConversionFunction | None = None,
        on_tool_start: ToolStartCallback | None = None,
        on_tool_end: ToolEndCallback | None = None,
        on_execution_start: ExecutionStartCallback | None = None,
        on_tool_progress: ProgressCallback | None = None,
        on_chunk: ChunkCallback[Any] | None = None,
        stop_event: asyncio.Event | None = None,
    ) -> str:
        effective_on_tool_start = (
            on_tool_start if on_tool_start is not None else self._on_tool_start
        )
        effective_on_tool_end = (
            on_tool_end if on_tool_end is not None else self._on_tool_end
        )
        effective_on_execution_start = (
            on_execution_start
            if on_execution_start is not None
            else self._on_execution_start
        )

        if on_chunk:
            self._llm_provider._on_chunk = on_chunk

        converted_user_input = user_input
        if custom_input_conversion:
            converted_user_input = await custom_input_conversion(converted_user_input)
        self._memory.add_message(
            cast(UserMessage, {"role": "user", "content": converted_user_input})
        )

        try:
            for _ in range(self._max_iterations):
                if stop_event and stop_event.is_set():
                    break

                response = await self._llm_provider.chat(
                    messages=self._memory.get_all_messages(),
                    tools=list(self._tools.values()),
                    stop_event=stop_event,
                )

                async for _ in response:
                    if stop_event and stop_event.is_set():
                        break

                if stop_event and stop_event.is_set():
                    self._memory.add_message(
                        cast(
                            Message,
                            {
                                "role": "assistant",
                                "content": "[Response stopped by user]",
                                "tool_calls": None,
                            },
                        )
                    )
                    break

                llm_response = response.response
                self._memory.add_message(
                    cast(
                        Message,
                        {
                            "role": "assistant",
                            "content": llm_response.content,
                            "tool_calls": llm_response.tool_calls or None,
                        },
                    )
                )

                if llm_response.usage:
                    self._memory.add_usage(llm_response.usage)

                if not llm_response.tool_calls:
                    return str(llm_response.content) or ""

                results = await self._execute_tools(
                    llm_response.tool_calls,
                    effective_on_tool_start,
                    effective_on_tool_end,
                    effective_on_execution_start,
                    on_tool_progress,
                    stop_event,
                )
                for msg in results:
                    self._memory.add_message(msg)

                if stop_event and stop_event.is_set():
                    break

        except asyncio.CancelledError:
            return str(self._memory.get_all_messages()[-1].get("content", "")) or ""

        if stop_event and stop_event.is_set():
            last = self._memory.get_all_messages()[-1]
            return str(last.get("content", "")) or ""

        raise RuntimeError(
            f"Agent exceeded maximum iterations ({self._max_iterations})"
        )

    async def _execute_tools(
        self,
        tool_calls: list[ToolCall],
        on_tool_start: ToolStartCallback | None,
        on_tool_end: ToolEndCallback | None,
        on_execution_start: ExecutionStartCallback | None,
        on_tool_progress: ProgressCallback | None,
        stop_event: asyncio.Event | None,
    ) -> list[Message]:
        if on_execution_start:
            await on_execution_start(tool_calls)

        tasks = [
            self._execute_single_tool(
                tc,
                on_tool_start,
                on_tool_end,
                on_tool_progress,
                stop_event,
            )
            for tc in tool_calls
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        messages: list[Message] = []
        for tc, result in zip(tool_calls, results):
            if isinstance(result, asyncio.CancelledError):
                content = (
                    f"[Tool Execution Stopped] {tc['function']['name']}: cancelled"
                )
            elif isinstance(result, Exception):
                content = f"[Tool Error] {tc['function']['name']}: {result}"
            else:
                content = (
                    json.dumps(result, ensure_ascii=False)
                    if not isinstance(result, str)
                    else result
                )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": content,
                }
            )

        return messages

    async def _execute_single_tool(
        self,
        tc: ToolCall,
        on_tool_start: ToolStartCallback | None,
        on_tool_end: ToolEndCallback | None,
        on_tool_progress: ProgressCallback | None,
        stop_event: asyncio.Event | None,
    ) -> Any:
        name = tc["function"]["name"]
        raw_args = tc["function"]["arguments"]

        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")

        tool = self._tools[name]
        args = json.loads(raw_args) if raw_args else {}

        return await tool.execute(
            args,
            tc,
            on_tool_start,
            on_tool_end,
            on_tool_progress,
            stop_event,
        )

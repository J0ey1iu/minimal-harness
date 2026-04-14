import asyncio
import json
from typing import Any, Awaitable, Callable

from minimal_harness.llm import ToolCall, ToolResultCallback
from minimal_harness.memory import Message
from minimal_harness.tool import Tool

ToolStartCallback = ToolResultCallback
ToolEndCallback = ToolResultCallback


ExecutionStartCallback = Callable[[list[ToolCall]], Awaitable[None]]


class ToolExecutor:
    def __init__(
        self,
        tools: dict[str, Tool],
        on_tool_start: ToolStartCallback | None = None,
        on_tool_end: ToolEndCallback | None = None,
        on_execution_start: ExecutionStartCallback | None = None,
    ):
        self._tools = tools
        self._on_tool_start = on_tool_start
        self._on_tool_end = on_tool_end
        self._on_execution_start = on_execution_start

    async def execute(self, tool_calls: list[ToolCall]) -> list[Message]:
        if self._on_execution_start:
            await self._on_execution_start(tool_calls)

        tasks = [self._execute_single(tc) for tc in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        messages: list[Message] = []
        for tc, result in zip(tool_calls, results):
            if isinstance(result, Exception):
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

    async def _execute_single(self, tc: ToolCall) -> Any:
        name = tc["function"]["name"]
        raw_args = tc["function"]["arguments"]

        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")

        if self._on_tool_start:
            await self._on_tool_start(tc, None)

        args = json.loads(raw_args) if raw_args else {}

        try:
            result = await self._tools[name].fn(**args)
        except Exception as e:
            if self._on_tool_end:
                await self._on_tool_end(tc, e)
            raise

        if self._on_tool_end:
            await self._on_tool_end(tc, result)

        return result

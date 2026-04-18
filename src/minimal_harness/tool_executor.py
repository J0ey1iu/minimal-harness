import asyncio
import json
from typing import Any

from minimal_harness.memory import Message
from minimal_harness.tool import (
    InteractiveTool,
    StreamingTool,
    Tool,
)
from minimal_harness.types import (
    ExecutionStartCallback,
    ProgressCallback,
    ToolCall,
    ToolEndCallback,
    ToolStartCallback,
    UserInputCallback,
)


class ToolExecutor:
    def __init__(
        self,
        tools: dict[str, Tool],
        on_tool_start: ToolStartCallback | None = None,
        on_tool_end: ToolEndCallback | None = None,
        on_execution_start: ExecutionStartCallback | None = None,
        wait_for_user_input: UserInputCallback | None = None,
        on_tool_progress: ProgressCallback | None = None,
    ):
        self._tools = tools
        self._on_tool_start = on_tool_start
        self._on_tool_end = on_tool_end
        self._on_execution_start = on_execution_start
        self._wait_for_user_input = wait_for_user_input
        self._on_tool_progress = on_tool_progress

    async def execute(
        self, tool_calls: list[ToolCall], stop_event: asyncio.Event | None = None
    ) -> list[Message]:
        if self._on_execution_start:
            await self._on_execution_start(tool_calls)

        tasks = [self._execute_single(tc, stop_event) for tc in tool_calls]
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

    async def _execute_single(
        self, tc: ToolCall, stop_event: asyncio.Event | None = None
    ) -> Any:
        name = tc["function"]["name"]
        raw_args = tc["function"]["arguments"]

        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")

        tool = self._tools[name]

        if isinstance(tool, StreamingTool):
            args = json.loads(raw_args) if raw_args else {}
            return await self._execute_streaming(tc, tool, args, stop_event)

        if isinstance(tool, InteractiveTool):
            args = json.loads(raw_args) if raw_args else {}
            return await self._execute_interactive(tc, tool, args, stop_event)

        if self._on_tool_start:
            await self._on_tool_start(tc, None)

        args = json.loads(raw_args) if raw_args else {}

        try:
            result = await tool.fn(**args)
        except Exception as e:
            if self._on_tool_end:
                await self._on_tool_end(tc, e)
            raise

        if self._on_tool_end:
            await self._on_tool_end(tc, result)

        return result

    async def _execute_interactive(
        self,
        tc: ToolCall,
        tool: InteractiveTool,
        args: dict[str, Any],
        stop_event: asyncio.Event | None = None,
    ) -> Any:
        if stop_event and stop_event.is_set():
            raise asyncio.CancelledError("Execution cancelled by user")

        if self._on_tool_start:
            await self._on_tool_start(tc, None)

        try:
            first_result = await tool.execute_first(**args)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if self._on_tool_end:
                await self._on_tool_end(tc, e)
            raise

        if stop_event and stop_event.is_set():
            raise asyncio.CancelledError("Execution cancelled by user")

        if self._wait_for_user_input is None:
            raise RuntimeError("wait_for_user_input callback not provided")

        user_input = await self._wait_for_user_input(first_result)

        if stop_event and stop_event.is_set():
            raise asyncio.CancelledError("Execution cancelled by user")

        try:
            final_result = await tool.execute_final(user_input, **args)
            if self._on_tool_end:
                await self._on_tool_end(tc, final_result)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if self._on_tool_end:
                await self._on_tool_end(tc, e)
            raise

        return final_result

    async def _execute_streaming(
        self,
        tc: ToolCall,
        tool: StreamingTool,
        args: dict[str, Any],
        stop_event: asyncio.Event | None = None,
    ) -> Any:
        if stop_event and stop_event.is_set():
            raise asyncio.CancelledError("Execution cancelled by user")

        if self._on_tool_start:
            await self._on_tool_start(tc, None)

        final_result = None
        try:
            async for chunk in tool.fn(**args):
                if stop_event and stop_event.is_set():
                    raise asyncio.CancelledError("Execution cancelled by user")
                if self._on_tool_progress:
                    await self._on_tool_progress(tc, chunk)
                final_result = chunk

            if self._on_tool_end:
                await self._on_tool_end(tc, final_result)
        except asyncio.CancelledError:
            if self._on_tool_end:
                await self._on_tool_end(tc, "[Stopped]")
            raise
        except Exception as e:
            if self._on_tool_end:
                await self._on_tool_end(tc, e)
            raise

        return final_result

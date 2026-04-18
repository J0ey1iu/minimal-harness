from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from openai.types.chat import ChatCompletionToolUnionParam

from minimal_harness.types import (
    ProgressCallback,
    StreamingToolFunction,
    ToolCall,
    ToolEndCallback,
    ToolFunction,
    ToolStartCallback,
    UserInputCallback,
)

if TYPE_CHECKING:
    pass


class Tool:
    def __init__(self, name: str, description: str, parameters: dict, fn: ToolFunction):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.fn = fn

    def to_schema(self) -> ChatCompletionToolUnionParam:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def execute(
        self,
        args: dict[str, Any],
        tool_call: ToolCall,
        on_tool_start: ToolStartCallback | None,
        on_tool_end: ToolEndCallback | None,
        stop_event: asyncio.Event | None,
    ) -> Any:
        if on_tool_start:
            await on_tool_start(tool_call, None)

        try:
            result = await self.fn(**args)
        except Exception as e:
            if on_tool_end:
                await on_tool_end(tool_call, e)
            raise

        if on_tool_end:
            await on_tool_end(tool_call, result)

        return result


class InteractiveTool:
    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict,
        fn_first: ToolFunction,
        fn_final: ToolFunction,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.fn_first = fn_first
        self.fn_final = fn_final

    async def execute_first(self, **kwargs: Any) -> Any:
        return await self.fn_first(**kwargs)

    async def execute_final(self, user_input: str, **kwargs: Any) -> Any:
        return await self.fn_final(user_input, **kwargs)

    def to_schema(self) -> ChatCompletionToolUnionParam:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def execute(
        self,
        args: dict[str, Any],
        tool_call: ToolCall,
        on_tool_start: ToolStartCallback | None,
        on_tool_end: ToolEndCallback | None,
        stop_event: asyncio.Event | None,
        wait_for_user_input: UserInputCallback,
    ) -> Any:
        if stop_event and stop_event.is_set():
            raise asyncio.CancelledError("Execution cancelled by user")

        if on_tool_start:
            await on_tool_start(tool_call, None)

        try:
            first_result = await self.execute_first(**args)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if on_tool_end:
                await on_tool_end(tool_call, e)
            raise

        if stop_event and stop_event.is_set():
            raise asyncio.CancelledError("Execution cancelled by user")

        user_input = await wait_for_user_input(first_result)

        if stop_event and stop_event.is_set():
            raise asyncio.CancelledError("Execution cancelled by user")

        try:
            final_result = await self.execute_final(user_input, **args)
            if on_tool_end:
                await on_tool_end(tool_call, final_result)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if on_tool_end:
                await on_tool_end(tool_call, e)
            raise

        return final_result


class StreamingTool:
    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict,
        fn: StreamingToolFunction,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.fn = fn

    def to_schema(self) -> ChatCompletionToolUnionParam:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def execute(
        self,
        args: dict[str, Any],
        tool_call: ToolCall,
        on_tool_start: ToolStartCallback | None,
        on_tool_end: ToolEndCallback | None,
        on_tool_progress: ProgressCallback | None,
        stop_event: asyncio.Event | None,
    ) -> Any:
        if stop_event and stop_event.is_set():
            raise asyncio.CancelledError("Execution cancelled by user")

        if on_tool_start:
            await on_tool_start(tool_call, None)

        final_result = None
        try:
            async for chunk in self.fn(**args):
                if stop_event and stop_event.is_set():
                    raise asyncio.CancelledError("Execution cancelled by user")
                if on_tool_progress:
                    await on_tool_progress(tool_call, chunk)
                final_result = chunk

            if on_tool_end:
                await on_tool_end(tool_call, final_result)
        except asyncio.CancelledError:
            if on_tool_end:
                await on_tool_end(tool_call, "[Stopped]")
            raise
        except Exception as e:
            if on_tool_end:
                await on_tool_end(tool_call, e)
            raise

        return final_result


BaseTool = Tool | StreamingTool | InteractiveTool

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from openai.types.chat import ChatCompletionToolUnionParam

from minimal_harness.types import (
    ProgressCallback,
    StreamingToolFunction,
    ToolCall,
    ToolEndCallback,
    ToolStartCallback,
)

if TYPE_CHECKING:
    pass


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

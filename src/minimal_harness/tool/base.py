from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, AsyncIterator

from openai.types.chat import ChatCompletionToolUnionParam

from minimal_harness.types import (
    StreamingToolFunction,
    ToolCall,
    ToolEnd,
    ToolEvent,
    ToolProgress,
    ToolStart,
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
        stop_event: asyncio.Event | None,
    ) -> AsyncIterator[ToolEvent]:
        yield ToolStart(tool_call)

        final_result = None
        error_msg: str | None = None
        try:
            async for chunk in self.fn(**args):
                if stop_event and stop_event.is_set():
                    error_msg = "stopped by the user"
                    break
                yield ToolProgress(tool_call, chunk)
                final_result = chunk
        except asyncio.CancelledError:
            error_msg = "stopped by the user"
        except BaseException as e:
            error_msg = f"[Error] {type(e).__name__}: {e}"

        if error_msg is not None:
            yield ToolEnd(tool_call, error_msg)
        else:
            yield ToolEnd(tool_call, final_result)

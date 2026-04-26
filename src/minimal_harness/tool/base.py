from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Protocol, runtime_checkable

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


class ToolExecutionError(Exception):
    def __init__(self, message: str, stderr: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.stderr = stderr


@runtime_checkable
class ToolRegistrationProtocol(Protocol):
    def register(self, tool: "StreamingTool") -> None: ...

    def register_external_tool(
        self,
        name: str,
        description: str,
        parameters: dict,
        fn: StreamingToolFunction,
        uri: "Path | str | None" = None,
        **kwargs: Any,
    ) -> None: ...

    def unregister(self, name: str) -> bool: ...

    def get(self, name: str) -> "StreamingTool | None": ...

    def get_all(self) -> list["StreamingTool"]: ...

    def names(self) -> list[str]: ...

    def clear(self) -> None: ...


def create_streaming_tool(
    name: str,
    fn: StreamingToolFunction,
    description: str | None = None,
    parameters: dict | None = None,
) -> StreamingTool:
    tool_description = description or (fn.__doc__ or "").strip()
    tool_params = parameters or {}
    return StreamingTool(
        name=name,
        description=tool_description,
        parameters=tool_params,
        fn=fn,
    )


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

    def to_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
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
        except ToolExecutionError as e:
            error_msg = f"[Error] {e.message}"
        except BaseException as e:
            error_msg = f"[Error] {type(e).__name__}: {e}"

        if error_msg is not None:
            yield ToolEnd(tool_call, error_msg)
        else:
            yield ToolEnd(tool_call, final_result)

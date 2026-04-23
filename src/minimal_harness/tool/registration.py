from __future__ import annotations

from typing import TYPE_CHECKING

from minimal_harness.tool.base import ToolRegistrationProtocol, create_streaming_tool
from minimal_harness.types import StreamingToolFunction

if TYPE_CHECKING:
    pass


def register_tool(
    name: str | None = None,
    description: str | None = None,
    parameters: dict | None = None,
    *,
    registry: ToolRegistrationProtocol,
):
    def decorator(fn: StreamingToolFunction) -> StreamingToolFunction:
        tool_name = name or fn.__name__
        tool = create_streaming_tool(tool_name, fn, description, parameters)
        registry.register(tool)
        return fn

    return decorator


def register(
    name: str,
    description: str,
    parameters: dict,
    fn: StreamingToolFunction,
    registry: ToolRegistrationProtocol,
) -> None:
    tool = create_streaming_tool(name, fn, description, parameters)
    registry.register(tool)


def unregister(name: str, registry: ToolRegistrationProtocol) -> bool:
    return registry.unregister(name)

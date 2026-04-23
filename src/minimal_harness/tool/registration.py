from __future__ import annotations

from typing import TYPE_CHECKING

from minimal_harness.tool.base import ToolRegistrationProtocol, create_streaming_tool
from minimal_harness.tool.registry import ToolRegistry
from minimal_harness.types import StreamingToolFunction

if TYPE_CHECKING:
    pass


def register_tool(
    name: str | None = None,
    description: str | None = None,
    parameters: dict | None = None,
    registry: ToolRegistrationProtocol | None = None,
):
    def decorator(fn: StreamingToolFunction) -> StreamingToolFunction:
        tool_name = name or fn.__name__
        tool = create_streaming_tool(tool_name, fn, description, parameters)
        (registry or ToolRegistry.get_instance()).register(tool)
        return fn

    return decorator


def register(
    name: str,
    description: str,
    parameters: dict,
    fn: StreamingToolFunction,
    registry: ToolRegistrationProtocol | None = None,
) -> None:
    tool = create_streaming_tool(name, fn, description, parameters)
    (registry or ToolRegistry.get_instance()).register(tool)


def unregister(name: str, registry: ToolRegistrationProtocol | None = None) -> bool:
    return (registry or ToolRegistry.get_instance()).unregister(name)

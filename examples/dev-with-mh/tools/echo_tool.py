"""
Example custom tool for dev-with-mh.

This tool is passed directly to TUIApp for testing.
"""

import asyncio
from typing import Any, AsyncIterator

from minimal_harness.tool.base import StreamingTool
from minimal_harness.types import ToolCall


async def echo(message: str, count: int = 1) -> AsyncIterator[dict[str, Any]]:
    for i in range(1, count + 1):
        yield {"status": "progress", "message": f"[{i}/{count}] {message}"}
        await asyncio.sleep(0.3)
    yield {"success": True, "echoed": count, "message": message}


echo_tool = StreamingTool(
    name="echo",
    description="Echo a message multiple times with progress updates",
    parameters={
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The message to echo",
            },
            "count": {
                "type": "integer",
                "description": "Number of times to repeat (1-10)",
                "default": 1,
            },
        },
        "required": ["message"],
    },
    fn=echo,
)


async def test_tool(tool: StreamingTool, args: dict[str, Any]) -> None:
    tool_call: ToolCall = {
        "id": "test_call",
        "type": "function",
        "function": {
            "name": tool.name,
            "arguments": __import__("json").dumps(args),
        },
    }
    stop_event = asyncio.Event()
    print(f"Testing {tool.name} with args: {args}")
    async for event in tool.execute(args, tool_call, stop_event):
        print(f"  {event}")


if __name__ == "__main__":
    asyncio.run(test_tool(echo_tool, {"message": "Hello", "count": 3}))
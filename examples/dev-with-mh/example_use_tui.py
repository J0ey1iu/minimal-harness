"""
Example: Testing custom tools with the TUI.

This demonstrates how to build custom tools and register them with the
global ToolRegistry before launching the TUI.

Usage:
    python examples/dev-with-mh/example_use_tui.py

The TUI can be configured via Ctrl+O to set base_url, api_key,
model, and system_prompt before chatting.
"""

import asyncio

from tools.echo_tool import echo

from minimal_harness.client.built_in import TUIApp
from minimal_harness.tool.registry import ToolRegistry
from minimal_harness.types import LocalToolBinding, ToolMetadata


async def setup(registry: ToolRegistry) -> None:
    await registry.register(
        ToolMetadata(
            name="echo",
            display_name="Echo",
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
            binding=LocalToolBinding(fn=echo),
        )
    )


def main() -> None:
    registry = ToolRegistry()
    asyncio.run(setup(registry))
    app = TUIApp(registry=registry)
    app.run()


if __name__ == "__main__":
    main()

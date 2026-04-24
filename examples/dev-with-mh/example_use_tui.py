"""
Example: Testing custom tools with the TUI.

This demonstrates how to build custom tools and register them with the
global ToolRegistry before launching the TUI.

Usage:
    python examples/dev-with-mh/example_use_tui.py

The TUI can be configured via Ctrl+O to set base_url, api_key,
model, and system_prompt before chatting.
"""

from tools.echo_tool import echo_tool

from minimal_harness.client.built_in.tui import TUIApp
from minimal_harness.tool.registry import ToolRegistry


def main() -> None:
    registry = ToolRegistry()
    registry.register(echo_tool)
    app = TUIApp(registry=registry)
    app.run()


if __name__ == "__main__":
    main()
"""
Example: Testing custom tools with the TUI.

This demonstrates how to build custom tools and pass them to TUIApp
for testing/debugging.

Usage:
    python examples/dev-with-mh/example_use_tui.py

The TUI can be configured via Ctrl+O to set base_url, api_key,
model, and system_prompt before chatting.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from tools.echo_tool import echo_tool

from minimal_harness.client.built_in.tui import TUIApp


def main() -> None:
    app = TUIApp(tools=[echo_tool])
    app.run()


if __name__ == "__main__":
    main()
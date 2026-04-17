import argparse
import os

from minimal_harness.mhc import SimpleCli
from minimal_harness.tool.built_in import (
    ask_user,
    bash,
    create_file,
    delete_file,
    patch_file,
    read_file,
)
from minimal_harness.tool.registration import register_tool
from minimal_harness.tool.registry import ToolRegistry


@register_tool(
    name="get_weather",
    description="Get the current weather for a given location",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
        },
        "required": ["location"],
    },
)
async def get_weather(location: str) -> str:
    return f"The weather in {location} is sunny with a temperature of 72°F."


def _register_tools() -> None:
    registry = ToolRegistry.get_instance()
    for tool, fn in [
        (bash.bash_tool, bash.bash_handler),
        (create_file.create_file_tool, create_file.create_file_handler),
        (read_file.read_file_tool, read_file.read_file_handler),
        (patch_file.patch_file_tool, patch_file.patch_file_handler),
        (delete_file.delete_file_tool, delete_file.delete_file_handler),
        (ask_user.ask_user_tool, ask_user.ask_user_first),
    ]:
        registry.register(tool)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="simple-cli — Chat with LLMs via OpenAI-compatible API",
        prog="simple-cli",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("MH_BASE_URL"),
        help="API base URL (required if not set in MH_BASE_URL env var)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("MH_API_KEY") or os.environ.get("API_KEY"),
        help="API key (falls back to API_KEY env vars)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("MH_MODEL", "qwen3.5-27b"),
        help="Model name (default: qwen3.5-27b)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="System prompt for the assistant",
    )

    args = parser.parse_args()

    if not args.base_url:
        parser.error("--base-url is required (or set MH_BASE_URL env var)")

    _register_tools()

    cli = SimpleCli(
        api_key=args.api_key or "",
        base_url=args.base_url,
        model=args.model,
        system_prompt=args.system_prompt,
    )
    cli.run()


if __name__ == "__main__":
    main()

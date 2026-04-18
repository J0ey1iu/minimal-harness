# minimal-harness

**Documentation: [/docs](./docs/)**

A lightweight Python agent harness for building LLM-powered agents with tool-calling support.

**Latest version: 0.2.1**

## What This Project Is For

Minimal-harness is a lean framework for building agents that can call tools. It provides:

- **Pluggable LLM providers** - OpenAI, LiteLLM (optional), or implement your own
- **Tool system** - Create tools via decorators or directly; includes built-in tools (bash, file ops, grep, glob, ask_user)
- **Interactive tools** - Tools that pause execution to request user input mid-process
- **Conversation memory** - Tracks token usage across interactions
- **AsyncIterator events** - Real-time async iteration for chunks, tool start/end, execution start

## How to Build an App

### Project Structure

A typical app looks like this:

```
my-app/
├── cli.py          # Entry point with argparse
└── tools.py        # Your custom tools
```

### 1. Create Your Entry Point

```python
import argparse
from minimal_harness.mhc import SimpleCli
from minimal_harness.tool.built_in import bash, read_file, ask_user
from minimal_harness.tool.registry import ToolRegistry

def main():
    parser = argparse.ArgumentParser(description="My AI agent")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", default="qwen3.5-27b")
    args = parser.parse_args()

    # Register built-in tools
    registry = ToolRegistry.get_instance()
    registry.register(bash.bash_tool, bash.bash_handler)
    registry.register(read_file.read_file_tool, read_file.read_file_handler)
    registry.register(ask_user.ask_user_tool, ask_user.ask_user_first)

    # Run the CLI
    cli = SimpleCli(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
    )
    cli.run()

if __name__ == "__main__":
    main()
```

### 2. Add Custom Tools

Use the `@register_tool` decorator to add your own tools:

```python
from minimal_harness.tool.registration import register_tool

@register_tool(
    name="get_weather",
    description="Get weather for a location",
    parameters={
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    },
)
async def get_weather(location: str) -> str:
    return f"The weather in {location} is sunny."
```

The decorator auto-registers the tool. Just import it before `cli.run()`.

### 3. Run

```bash
python cli.py --base-url https://api.openai.com/v1 --api-key sk-... --model gpt-4o
```

Or set environment variables:

```bash
export MH_BASE_URL=https://api.openai.com/v1
export MH_API_KEY=sk-...
export MH_MODEL=gpt-4o
python cli.py
```

### Built-in Tools

| Tool          | Description            |
| ------------- | ---------------------- |
| `bash`        | Execute shell commands |
| `read_file`   | Read file contents     |
| `create_file` | Create new files       |
| `patch_file`  | Patch existing files   |
| `delete_file` | Delete files           |
| `ask_user`    | Request user input     |

### Environment Variables

| Variable      | Description                       |
| ------------- | --------------------------------- |
| `MH_BASE_URL` | API base URL                      |
| `MH_API_KEY`  | API key                           |
| `MH_MODEL`    | Model name (default: qwen3.5-27b) |

### Running the Built-in CLI

```bash
mhc --base-url https://api.openai.com/v1 --api-key sk-... --model gpt-4o
```

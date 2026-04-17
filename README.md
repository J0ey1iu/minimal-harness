# minimal-harness

A lightweight Python agent harness for building LLM-powered agents with tool-calling support.

**Latest version: 0.2.1**

## What This Project Is For

Minimal-harness is a lean framework for building agents that can call tools. It provides:

- **Pluggable LLM providers** - OpenAI, LiteLLM (optional), or implement your own
- **Tool system** - Create tools via decorators or directly; includes built-in tools (bash, file ops, grep, glob, ask_user)
- **Interactive tools** - Tools that pause execution to request user input mid-process
- **Conversation memory** - Tracks token usage across interactions
- **Streaming callbacks** - Real-time callbacks for chunks, tool start/end, execution start

## How to Build on This Project

### Quick Start

```python
import asyncio
from minimal_harness import OpenAIAgent, OpenAILLMProvider, ConversationMemory
from minimal_harness.tool.registration import register_tool
from minimal_harness.tool.registry import ToolRegistry

@register_tool(
    name="get_weather",
    description="Get the current weather for a location",
    parameters={
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    },
)
async def get_weather(location: str) -> str:
    return f"The weather in {location} is sunny."

async def main():
    client = OpenAILLMProvider(...)  # your LLM client
    agent = OpenAIAgent(
        llm_provider=client,
        tools=ToolRegistry.get_instance().get_all(),
        memory=ConversationMemory(system_prompt="You are helpful."),
    )
    result = await agent.run([{"type": "text", "text": "What's the weather in Tokyo?"}])
    print(result)

asyncio.run(main())
```

### Creating Tools

**Option 1: Decorator** (auto-registers)

```python
from minimal_harness.tool.registration import register_tool

@register_tool(
    name="my_tool",
    description="Does something useful",
    parameters={
        "type": "object",
        "properties": {"input": {"type": "string"}},
        "required": ["input"],
    },
)
async def my_tool(input: str) -> str:
    return f"Processed: {input}"
```

**Option 2: Direct registration**

```python
from minimal_harness.tool import Tool
from minimal_harness.tool.registry import ToolRegistry

async def my_handler(arg1: str, arg2: int) -> str:
    return f"{arg1} {arg2}"

tool = Tool(
    name="my_tool",
    description="My tool description",
    parameters={...},
    fn=my_handler,
)
ToolRegistry.get_instance().register(tool)
```

### Interactive Tools (User Input Mid-Execution)

For tools that need user input during execution:

```python
from minimal_harness.tool import InteractiveTool

async def ask_first(question: str) -> str:
    return question  # This gets shown to the user

async def ask_final(user_input: str, question: str) -> str:
    return user_input  # This is what the agent receives

tool = InteractiveTool(
    name="ask_user",
    description="Ask the user a question",
    parameters={...},
    fn_first=ask_first,
    fn_final=ask_final,
)
```

When calling `agent.run()`, pass `wait_for_user_input` callback:

```python
async def wait_for_user_input(question: str) -> str:
    print(f"[User Input Required] {question}")
    return input("Your answer: ")

await agent.run(
    [...],
    wait_for_user_input=wait_for_user_input,
)
```

### Streaming Callbacks

```python
async def on_chunk(chunk, is_done):
    # Handle streaming chunks (content, tool calls, thinking)
    ...

async def on_tool_start(tool_call, tool):
    print(f"Starting: {tool_call['function']['name']}")

async def on_tool_end(tool_call, result):
    print(f"Finished: {tool_call['function']['name']} -> {result}")

async def on_execution_start(tool_calls):
    print(f"Executing {len(tool_calls)} tools")

await agent.run(
    [...],
    on_chunk=on_chunk,
    on_tool_start=on_tool_start,
    on_tool_end=on_tool_end,
    on_execution_start=on_execution_start,
)
```

### Built-in Tools Reference

| Tool          | Description                         |
| ------------- | ----------------------------------- |
| `bash`        | Execute shell commands              |
| `read_file`   | Read file contents with line ranges |
| `create_file` | Create new files                    |
| `patch_file`  | Patch existing files                |
| `delete_file` | Delete files                        |
| `glob`        | Find files by pattern               |
| `grep`        | Search file contents                |
| `ask_user`    | Request user input                  |

### Environment Variables

| Variable      | Description                       |
| ------------- | --------------------------------- |
| `MH_BASE_URL` | API base URL                      |
| `MH_API_KEY`  | API key                           |
| `MH_MODEL`    | Model name (default: qwen3.5-27b) |

### Running the CLI

```bash
# TUI chat interface
mh --base-url https://api.openai.com/v1 --api-key sk-... --model gpt-4o

# Simple CLI (streaming)
simple-cli --base-url https://api.openai.com/v1 --api-key sk-... --model gpt-4o
```

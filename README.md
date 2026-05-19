# minimal-harness

**Documentation: [/docs](./docs/)**

A lightweight Python agent harness for building LLM-powered agents with tool-calling support.

Latest version: **0.6.0.post1**

## What This Project Is For

Minimal-harness is a lean framework for building agents that can call tools. It provides:

- **OpenAI/Anthropic-compatible API** - Works with OpenAI, Anthropic, or any OpenAI-compatible API provider
- **Multi-modal image input** - Pass image URLs or base64 data to LLM providers supporting vision
- **Symmetric Registry + Factory architecture** - Register tool/agent metadata with bindings (`LocalToolBinding`, `RemoteToolBinding`, `ExternalScriptToolBinding`); executable instances created lazily by `ToolFactory`
- **Middleware hooks** - Observe and intercept the agent lifecycle (agent start/end, LLM calls, tool execution, tool policy enforcement)
- **AsyncIterator events** - Real-time async iteration for chunks, tool start/end, execution events
- **Conversation memory sessions** - Persistent sessions with identity (user_id, scenario_id), auto-persisted to disk
- **Remote agents & tools** - Execute agents and tools remotely via SSE over HTTP; pluggable driver/executor protocols
- **Batch evaluation** - Built-in `eval` module for running agent evaluation suites and generating reports
- **ESC stop support** - Gracefully stop LLM streaming and tool execution

## Architecture

The framework uses a **three-layer architecture**:

```
Layer 3: Application (TUI client)
Layer 2: Service Abstractions (AgentRuntime, Registry, SessionStore, Factory, Remote drivers)
Layer 1: Core Abstractions (Agent, Tool, Memory, LLMProvider, AgentEvent/ToolEvent)
```

All event types are defined in `src/minimal_harness/types.py`. No separate client event layer exists.

**Event flow:**

```python
async for event in agent.run(
    user_input=[{"type": "text", "text": "..."}],
    memory=memory,
    tools=tools,
):
    if isinstance(event, LLMChunk):
        # handle chunk
    elif isinstance(event, ToolEnd):
        # handle tool result
```

## How to Build an App

### Project Structure

A typical app looks like this:

```
my-app/
├── cli.py          # Entry point
└── tools.py        # Your custom tools
```

### 1a. Layer 1 — Direct Control

```python
import argparse
import asyncio
from openai import AsyncOpenAI

from minimal_harness.agent.simple import SimpleAgent
from minimal_harness.llm.openai import OpenAILLMProvider
from minimal_harness.memory import ConversationMemory
from minimal_harness.tool.built_in.bash import get_tools as get_bash_tools
from minimal_harness.types import (
    AgentStart,
    AgentEnd,
    LLMChunk,
    ToolStart,
    ToolEnd,
)

def main():
    parser = argparse.ArgumentParser(description="My AI agent")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", default="deepseek-v4-flash")
    args = parser.parse_args()

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    llm_provider = OpenAILLMProvider(client=client, model=args.model)
    agent = SimpleAgent(llm_provider=llm_provider, max_iterations=50)
    memory = ConversationMemory()
    tools = list(get_bash_tools().values())

    async def run():
        stop_event = asyncio.Event()
        context = {"user_id": "abc123"}  # passed to middleware hooks
        async for event in agent.run(
            user_input=[{"type": "text", "text": "What files are in the current directory?"}],
            stop_event=stop_event,
            memory=memory,
            tools=tools,
            context=context,
        ):
            if isinstance(event, AgentStart):
                print("Agent starting...")
            elif isinstance(event, LLMChunk):
                delta = event.chunk
                if delta and delta.content:
                    print(delta.content, end="", flush=True)
            elif isinstance(event, ToolStart):
                print(f"\n[Calling tool: {event.tool_call['function']['name']}]")
            elif isinstance(event, ToolEnd):
                print(f"\n[Tool result: {str(event.result)[:100]}...]")
            elif isinstance(event, AgentEnd):
                print(f"\n[Done in {event.time_taken:.2f}s]")
                break

    asyncio.run(run())

if __name__ == "__main__":
    main()
```

### 1b. Layer 2 — Managed Orchestration

```python
from minimal_harness.agent.runtime import AgentRuntime
from minimal_harness.agent.registry import AgentRegistry
from minimal_harness.tool.registry import ToolRegistry, collect_builtin_tools
from minimal_harness.client.built_in.memory_store import DiskSessionStore
from minimal_harness.types import AgentMetadata

tool_registry = ToolRegistry()
await collect_builtin_tools(tool_registry)

agent_registry = AgentRegistry()
await agent_registry.register(AgentMetadata(
    name="assistant", display_name="Assistant",
    description="General assistant",
    system_prompt="You are helpful.", agent_type="simple",
    tool_names=["bash", "local_file_operation"],
))

store = DiskSessionStore()
runtime = AgentRuntime(
    agent_registry=agent_registry,
    session_store=store,
    tool_registry=tool_registry,
    llm_provider_factory=lambda: create_llm_provider(...),
)
await runtime.register_runtime_tools()

session = await store.create_session()
task, stop, queue = runtime.run(
    user_input=[{"type": "text", "text": user_message}],
    agent_metadata_id="assistant",
    memory_id=session.session_id,
)
```

### 2. Add Custom Tools

Tools are defined as async generator functions and registered via **`ToolMetadata` + Binding**:

```python
from minimal_harness.tool.registry import ToolRegistry
from minimal_harness.types import ToolMetadata, LocalToolBinding

registry = ToolRegistry()

async def get_weather(location: str) -> AsyncIterator[dict]:
    yield {"success": True, "result": f"The weather in {location} is sunny."}

await registry.register(ToolMetadata(
    name="get_weather",
    display_name="Get Weather",
    description="Get weather for a location",
    parameters={
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    },
    binding=LocalToolBinding(fn=get_weather),
))
```

Or use the `@register_tool` decorator (recommended pattern — omit `registry` and call `register_decorated_tools()` during async setup):

```python
from minimal_harness.tool.registration import register_tool, register_decorated_tools

@register_tool(
    name="get_weather",
    description="Get weather for a location",
    parameters={
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    },
    # registry=...  # optional — see below
)
async def get_weather(location: str) -> AsyncIterator[dict]:
    yield {"success": True, "result": f"The weather in {location} is sunny."}

# Later, during async setup:
await register_decorated_tools(registry)
```

For **remote tools**, use `RemoteToolBinding`:

```python
from minimal_harness.types import RemoteToolBinding

await registry.register(ToolMetadata(
    name="weather",
    description="Get weather",
    parameters={...},
    binding=RemoteToolBinding(url="https://my-service.com/weather"),
))
```

For **external script tools**, use `ExternalScriptToolBinding`:

```python
from minimal_harness.types import ExternalScriptToolBinding

await registry.register(ToolMetadata(
    name="my_tool",
    description="...",
    parameters={...},
    binding=ExternalScriptToolBinding(script_path="/path/to/tool.py"),
))
```

**Localized tool output**: Tools can detect the user's language at runtime via `get_current_locale()`:

```python
from minimal_harness.agent.runtime import get_current_locale

async def my_tool() -> AsyncIterator[dict]:
    locale = get_current_locale()
    yield {"message": "你好" if locale == "zh" else "Hello"}
```

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

### Middleware Hooks

Subclass `Middleware` to observe or intercept the agent lifecycle:

```python
from minimal_harness.agent.middleware import Middleware
from minimal_harness.types import LLMEnd, ToolCall

class PolicyEnforcer(Middleware):
    async def should_allow_tool(
        self, tool_call: ToolCall, **kwargs
    ) -> bool | str:
        if tool_call["function"]["name"] == "bash":
            return "bash is not permitted in this context"
        return True

    async def on_llm_end(self, event: LLMEnd) -> None:
        if event.usage:
            print(f"Tokens: {event.usage['total_tokens']}")
```

Pass middleware to `SimpleAgent`:

```python
agent = SimpleAgent(
    llm_provider=llm_provider,
    middleware=[PolicyEnforcer()],
    max_iterations=50,
)
```

### Multi-modal Image Input

Pass image URLs or base64-encoded image data as input content parts:

```python
user_input = [
    {"type": "text", "text": "What's in this image?"},
    {
        "type": "image",
        "image_url": {"url": "https://example.com/photo.jpg"},
    },
]
```

For local images, encode as base64:

```python
import base64

with open("photo.jpg", "rb") as f:
    data = base64.b64encode(f.read()).decode()

user_input = [
    {"type": "text", "text": "Describe this image"},
    {
        "type": "image",
        "data": data,
        "media_type": "image/jpeg",
    },
]
```

### Built-in Tools

Register them in bulk via `collect_builtin_tools()`:

```python
from minimal_harness.tool.registry import collect_builtin_tools
await collect_builtin_tools(tool_registry)  # returns set[str] of names
```

| Tool                   | Description                                           |
| ---------------------- | ----------------------------------------------------- |
| `bash`                 | Execute shell commands with timeout and workdir support |
| `local_file_operation` | Read, write, patch, or delete files (4 universal modes) |

### Event Types

All events are defined in `minimal_harness.types` and consumed as a single `AgentEvent` union:

| Event             | Fields                                                 | Description                     |
| ----------------- | ------------------------------------------------------ | ------------------------------- |
| `AgentStart`      | `user_input`, `timestamp`                              | Agent execution started         |
| `AgentEnd`        | `response`, `time_taken`, `exceeded`, `interrupted`    | Agent execution completed       |
| `LLMStart`        | `messages`, `tools`                                    | LLM generation started          |
| `LLMChunk`        | `chunk: LLMChunkDelta \| None`                         | LLM output chunk received       |
| `LLMEnd`          | `content`, `reasoning_content`, `tool_calls`, `usage`  | LLM generation completed        |
| `ExecutionStart`  | `tool_calls`                                           | Tool execution started          |
| `ExecutionEnd`    | `results`                                              | Tool execution completed        |
| `ToolStart`       | `tool_call`                                            | Tool call started               |
| `ToolProgress`    | `tool_call`, `chunk`                                   | Tool intermediate progress      |
| `ToolEnd`         | `tool_call`, `result`                                  | Tool call completed with result |
| `MemoryUpdate`    | `usage`                                                | Memory token usage updated      |
| `MessageEvent`    | `message`                                              | Conversation message added to memory |

`LLMChunkDelta` contains `content`, `reasoning`, and `tool_calls` fields for provider-agnostic partial deltas.

### Batch Evaluation

The `eval` module runs agent evaluation suites and generates metrics reports:

```bash
python -m minimal_harness.eval.runner \
    --eval-suite my_suite.json \
    --results-dir ./eval_results
```

```python
from minimal_harness.eval.runner import EvalRunner
from minimal_harness.eval.types import EvalCase

runner = EvalRunner(registry, runtime)
report = await runner.run([
    EvalCase(input="Sort [3,1,2]", expected="[1,2,3]"),
])
print(report.summary())  # pass_rate, avg_score, etc.
```

See [docs/eval-guide.md](./docs/eval-guide.md) for details.

### Remote Agents

Register agents that execute on a remote service via SSE over HTTP:

```python
from minimal_harness.types import AgentMetadata, RemoteAgentBinding

await agent_registry.register(AgentMetadata(
    name="remote_coder",
    binding=RemoteAgentBinding(
        url="https://my-agent-service.example.com/run",
        headers={"Authorization": "Bearer xxx"},
    ),
))
```

This creates a `RemoteAgent` backed by `SSEAgentDriver`. Implement `RemoteAgentDriver` for custom transports.

### Environment Variables

| Variable             | Description                                 |
| -------------------- | ------------------------------------------- |
| `MH_BASE_URL`        | API base URL (default: https://aihubmix.com/v1) |
| `MH_API_KEY`         | API key                                     |
| `MH_MODEL`           | Model name (default: deepseek-v4-flash)      |
| `MH_MAX_ITERATIONS`  | Max agent loop iterations (default: 100)    |
| `MH_THEME`           | TUI theme name (default: tokyo-night)       |

### Stop Mechanism

Press **ESC** during execution to gracefully stop LLM streaming and tool execution.

# Developer Guide

This guide walks through building agent applications using the Layer 1 (Core Abstractions) and Layer 2 (Service Abstractions) from `minimal-harness`. You can use these layers directly without ever touching the TUI.

---

## Choosing Your Abstraction Level

| Level | When to Use |
|-------|-------------|
| **Layer 1 only** | You want full control — wire up agent, LLM, memory, and tools yourself |
| **Layer 2** | You want higher-level orchestration — registries, runtime, persistent memory |
| **Layer 2 + Runtime** | You want multi-agent handoff, event-driven execution, task-based concurrency |

---

## Layer 1: Core Abstractions

Layer 1 gives you four protocols and an event system. You compose them directly.

### 1. Memory — Conversation History

`Memory` stores messages and tracks token usage.

```python
from minimal_harness.memory import ConversationMemory

memory = ConversationMemory()
memory.add_message({"role": "user", "content": [{"type": "text", "text": "Hello"}]})
memory.add_message({"role": "assistant", "content": "Hi there!", "tool_calls": None})

for msg in memory.get_all_messages():
    print(msg["role"], msg["content"])
```

Helper constructors are available for each message type:

```python
from minimal_harness.memory import (
    user_message, assistant_message, tool_message, system_message
)

memory.add_message(system_message("You are a helpful assistant"))
memory.add_message(user_message([{"type": "text", "text": "Hello"}]))
memory.add_message(assistant_message("Hi!", tool_calls=None))
memory.add_message(tool_message("call_123", "command output"))
```

`get_forward_messages()` excludes `reasoning` messages (used when sending to LLM). `dump_memory()` returns a serializable dict for persistence.

### 2. Tools — LLM-Callable Functions

A tool wraps an async generator function. The generator yields progress events and a final result.

**Define a tool function:**

```python
import asyncio
from typing import Any, AsyncIterator

async def reverse_text(text: str) -> AsyncIterator[dict]:
    yield {"status": "progress", "message": f"Reversing '{text}'..."}
    await asyncio.sleep(0.2)
    yield {"success": True, "result": text[::-1]}
```

**Wrap it into a `StreamingTool`:**

```python
from minimal_harness.tool.base import StreamingTool

reverse_tool = StreamingTool(
    name="reverse",
    description="Reverse a given string",
    parameters={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The string to reverse",
            },
        },
        "required": ["text"],
    },
    fn=reverse_text,
)
```

Or use the convenience factory:

```python
from minimal_harness.tool.base import create_streaming_tool

reverse_tool = create_streaming_tool(
    name="reverse",
    fn=reverse_text,
    description="Reverse a given string",
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string", "description": "Text to reverse"}},
        "required": ["text"],
    },
)
```

The tool is ready. You can test it directly:

```python
call = {"id": "t1", "type": "function", "function": {"name": "reverse", "arguments": '{"text": "hello"}'}}
async for event in reverse_tool.execute({"text": "hello"}, call, None):
    print(event)  # ToolStart → ToolProgress → ToolEnd
```

**Built-in tools** (`bash`, file operations) are available via `tool/built_in/`:

```python
from minimal_harness.tool.built_in.bash import get_tools as get_bash_tools
bash_tools = get_bash_tools()  # {"bash": StreamingTool, ...}
```

### 3. LLM Provider — Talk to an LLM

`LLMProvider` abstracts the chat completion API.

```python
from openai import AsyncOpenAI
from minimal_harness.llm.openai import OpenAILLMProvider

client = AsyncOpenAI(api_key="...", base_url="...")
provider = OpenAILLMProvider(client=client, model="gpt-4o")

response = await provider.chat(
    messages=[{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
    tools=[reverse_tool],
)

async for chunk in response:
    print(chunk)  # LLMChunkDelta(content="...", reasoning=None, tool_calls=None)

# After iteration, access the final response:
llm_response = response.response
print(llm_response.content, llm_response.tool_calls)
```

For Anthropic:

```python
from anthropic import AsyncAnthropic
from minimal_harness.llm.anthropic import AnthropicLLMProvider

client = AsyncAnthropic(api_key="...")
provider = AnthropicLLMProvider(client=client, model="claude-sonnet-4-20250514")
```

### 4. Agent — The Execution Loop

`SimpleAgent` runs the standard agentic loop: user input → LLM → tool calls → LLM → ...

```python
import asyncio
from minimal_harness.agent.simple import SimpleAgent
from minimal_harness.memory import ConversationMemory

agent = SimpleAgent(llm_provider=provider, max_iterations=10)
memory = ConversationMemory()

stop_event = asyncio.Event()
async for event in agent.run(
    user_input=[{"type": "text", "text": "Reverse the word 'hello'"}],
    stop_event=stop_event,
    memory=memory,
    tools=[reverse_tool],
    system_prompt="You are a helpful assistant.",
):
    # Each event is an AgentEvent (Union type)
    match event:
        case AgentStart():       print("Run started")
        case LLMStart():         print("LLM call started")
        case LLMChunk():         print(event.chunk.content or "", end="")
        case LLMEnd():           print("LLM done")
        case ExecutionStart():   print("Tool execution begins")
        case ToolStart():        print(f"Tool: {event.tool_call['function']['name']} started")
        case ToolProgress():     print(f"Progress: {event.chunk}")
        case ToolEnd():          print(f"Tool done: {event.result}")
        case ExecutionEnd():     print("Tool execution done")
        case MemoryUpdate():     print(f"Usage: {event.usage}")
        case AgentEnd():         print(f"Done: {event.response}")
```

### 5. Events Reference

All events are `@dataclass` types, unified under `AgentEvent`:

| Event | Fields |
|-------|--------|
| `AgentStart` | `user_input` |
| `AgentEnd` | `response`, `time_taken`, `exceeded`, `interrupted` |
| `LLMStart` | `messages`, `tools` |
| `LLMChunk` | `chunk: LLMChunkDelta \| None` |
| `LLMEnd` | `content`, `reasoning_content`, `tool_calls`, `usage` |
| `ExecutionStart` | `tool_calls` |
| `ExecutionEnd` | `results: list[(ToolCall, Any)]` |
| `ToolStart` | `tool_call` |
| `ToolProgress` | `tool_call`, `chunk` |
| `ToolEnd` | `tool_call`, `result` |
| `MemoryUpdate` | `usage` |

---

## Layer 2: Service Abstractions

Layer 2 provides registries, persistent memory, and the `AgentRuntime` orchestrator.

### 1. Registries — Discoverable Components

**`ToolRegistry`** — register and look up tools by name:

```python
from minimal_harness.tool.registry import ToolRegistry

tool_registry = ToolRegistry()
tool_registry.register(reverse_tool)
tool_registry.register(bash_tool)

tool_registry.names()           # ["reverse", "bash"]
tool_registry.get("reverse")    # -> StreamingTool
tool_registry.get_all()         # -> list[Tool]
```

Register built-in tools in bulk:

```python
from minimal_harness.tool.registry import collect_builtin_tools
collect_builtin_tools(tool_registry)  # returns set of names registered
```

**`AgentRegistry`** — register agent metadata:

```python
from minimal_harness.types import AgentMetadata
from minimal_harness.agent.registry import AgentRegistry

agent_registry = AgentRegistry()
agent_registry.register(AgentMetadata(
    name="coder",
    description="Writes and debugs code",
    system_prompt="You are a coding expert.",
    agent_type="simple",
    tool_names=["bash", "read_file", "write_file"],
))
```

**`ToolRegistryProtocol`** and **`AgentRegistryProtocol`** are `@runtime_checkable`, so you can substitute custom implementations.

### 2. MemoryStore — Persistent Conversations

```python
from minimal_harness.memory_store import MemoryStore

store = MemoryStore(storage_dir="/path/to/memories")

# Create a new conversation
memory = store.create_memory(memory_id="conv_001", agent_name="coder")

# Retrieve later
memory = store.get_memory("conv_001")
for msg in memory.get_all_messages():
    print(msg)

# Save explicitly (auto-persist is on by default)
store.save_memory(memory, "conv_001")

# List all sessions
store.list_sessions()

# Delete
store.delete_memory("conv_001")
```

`MemoryStoreProtocol` allows you to swap in custom backends (e.g., SQLite, Redis).

### 3. Settings — Configuration from Environment

```python
from minimal_harness.settings import Settings

Settings.model()           # MH_MODEL env or DEFAULT_MODEL
Settings.base_url()        # MH_BASE_URL
Settings.api_key()         # MH_API_KEY
Settings.max_iterations()  # MH_MAX_ITERATIONS
```

### 4. LLM Provider Factory

```python
from minimal_harness.llm import create_llm_provider

provider = create_llm_provider({
    "provider": "openai",
    "model": "gpt-4o",
    "base_url": "https://api.openai.com/v1",
    "api_key": "sk-...",
})
```

### 5. AgentRuntime — The Core Orchestrator

`AgentRuntime` ties everything together. Given metadata IDs, it resolves agents, tools, memory, and runs the agent loop as a background `asyncio.Task`.

```python
from minimal_harness.agent.runtime import AgentRuntime
from minimal_harness.llm import create_llm_provider

# Wire up Layer 2 components
runtime = AgentRuntime(
    agent_registry=agent_registry,
    memory_store=store,
    tool_registry=tool_registry,
    llm_provider_factory=lambda: create_llm_provider({
        "provider": "openai",
        "model": "gpt-4o",
        "base_url": "https://api.openai.com/v1",
        "api_key": "sk-...",
    }),
)

# Register runtime tools (handoff, discover_agents)
runtime.register_runtime_tools()

# Start a run
task, stop_event, event_queue = runtime.run(
    user_input=[{"type": "text", "text": "Write a Python script to sort a list"}],
    agent_metadata_id="coder",
    memory_id="conv_001",
)
```

The return value `(Task, Event, Queue)` gives you full control:

```python
import asyncio

# Consume events
while True:
    event = await event_queue.get()
    if event is None:
        break
    # process event...

# Stop mid-run
stop_event.set()
await task
```

`AgentRuntimeProtocol` is `@runtime_checkable` — you can write your own runtime that satisfies the same interface.

#### Using Custom Agent Factories

You can provide a custom `AgentFactory` to substitute `SimpleAgent` with your own agent implementation:

```python
from minimal_harness.agent.protocol import Agent

class MyCustomAgent:
    def __init__(self, llm_provider, max_iterations):
        self._llm_provider = llm_provider
        self._max_iterations = max_iterations

    def run(self, user_input, stop_event, memory, tools, system_prompt=""):
        # Your custom loop here...
        ...

def my_agent_factory(agent_type: str) -> Agent:
    if agent_type == "custom":
        llm = create_llm_provider({"provider": "openai", "model": "gpt-4o"})
        return MyCustomAgent(llm, max_iterations=20)
    raise ValueError(f"Unknown type: {agent_type}")

runtime = AgentRuntime(
    agent_registry=...,
    memory_store=...,
    tool_registry=...,
    agent_factory=my_agent_factory,
    llm_provider_factory=...,
)
```

## Common Patterns

### Pattern 1: Headless CLI Agent (Layer 1)

```python
import asyncio
from minimal_harness.agent.simple import SimpleAgent
from minimal_harness.memory import ConversationMemory
from minimal_harness.llm.openai import OpenAILLMProvider
from minimal_harness.tool.base import create_streaming_tool
from minimal_harness.types import AgentEnd, LLMChunk

async def my_tool(query: str) -> AsyncIterator[dict]:
    yield {"result": f"processed: {query}"}

tool = create_streaming_tool("my_tool", my_tool)
provider = OpenAILLMProvider(client=..., model="gpt-4o")
agent = SimpleAgent(provider, max_iterations=5)
memory = ConversationMemory()

async for event in agent.run(
    user_input=[{"type": "text", "text": "Process this data"}],
    stop_event=None,
    memory=memory,
    tools=[tool],
):
    if isinstance(event, LLMChunk) and event.chunk:
        print(event.chunk.content or "", end="")
    elif isinstance(event, AgentEnd):
        print(f"\nDone: {event.response[:200]}")
```

### Pattern 2: Server with Registries (Layer 2)

```python
from minimal_harness.agent.runtime import AgentRuntime
from minimal_harness.agent.registry import AgentRegistry
from minimal_harness.tool.registry import ToolRegistry, collect_builtin_tools
from minimal_harness.memory_store import MemoryStore
from minimal_harness.types import AgentMetadata

# Setup
tool_registry = ToolRegistry()
collect_builtin_tools(tool_registry)

agent_registry = AgentRegistry()
agent_registry.register(AgentMetadata(
    name="assistant", description="General assistant",
    system_prompt="You are helpful.", agent_type="simple",
    tool_names=["bash", "read", "write"],
))

store = MemoryStore()
runtime = AgentRuntime(agent_registry, store, tool_registry,
    llm_provider_factory=lambda: create_llm_provider(...))
runtime.register_runtime_tools()

# Each user request:
memory = store.create_memory()
task, stop, queue = runtime.run(
    user_input=[{"type": "text", "text": user_message}],
    agent_metadata_id="assistant",
    memory_id=memory.memory_id,
)
```

### Pattern 3: Multi-Agent Handoff (Layer 2)

Runtime automatically injects `handoff` and `discover_agents` tools (call `register_runtime_tools()` first). Your system prompt should reference them:

```python
agent_registry.register(AgentMetadata(
    name="triage",
    description="Routes tasks to specialist agents",
    system_prompt=(
        "You are a triage agent. Use discover_agents to find "
        "available specialists, then handoff tasks to them."
    ),
    agent_type="simple",
    tool_names=[],  # handoff/discover_agents injected by runtime
))
agent_registry.register(AgentMetadata(
    name="coder",
    description="Writes and debugs code",
    system_prompt="You are a coding expert.",
    agent_type="simple",
    tool_names=["bash"],
))
```

When LLM calls `handoff(target_agent_name="coder", ...)`, Runtime spawns a child task with the coder agent — events from the child are tunneled back through the parent's queue.

### Pattern 4: Custom Event Consumer

The event-driven design makes it easy to build custom UIs, bots, or logging:

```python
async def consume_events(queue: asyncio.Queue[AgentEvent | None]):
    buffer = ""
    while True:
        event = await queue.get()
        if event is None:
            break
        if isinstance(event, AgentStart):
            print(f"╔══ Agent Run ══╗")
        elif isinstance(event, LLMChunk) and event.chunk:
            if event.chunk.content:
                buffer += event.chunk.content
                print(event.chunk.content, end="", flush=True)
        elif isinstance(event, ToolStart):
            print(f"\n  ┌─ {event.tool_call['function']['name']}")
        elif isinstance(event, ToolProgress):
            print(f"  │ {event.chunk}")
        elif isinstance(event, ToolEnd):
            print(f"  └─ Result: {str(event.result)[:100]}")
        elif isinstance(event, AgentEnd):
            print(f"\n╚══ {event.time_taken:.2f}s ══╝")
```

---

## What Not to Do

1. **Don't import Layer 2 from Layer 1 code** — Layer 1 (`agent/`, `llm/`, `memory.py`, `tool/base.py`) must not import `Settings`, registries, runtime, or memory store. Pass dependencies explicitly via `__init__`.

2. **Don't hardcode concrete implementations** — Accept protocols (`Agent`, `LLMProvider`, `Memory`, `Tool`, `AgentRuntimeProtocol`, etc.) rather than constructing specific classes. This keeps your code testable and swappable.

3. **Don't call `Queue.get()` without timeout in production** — Use `asyncio.wait_for(queue.get(), timeout=...)` to avoid blocking indefinitely if the task stalls.

4. **Don't forget to call `register_runtime_tools()`** — Without it, `handoff` and `discover_agents` will not be available to agents.

5. **Don't rely on `asyncio.Task` reference escaping** — The runtime returns the task; ensure you `await task` after setting `stop_event` to clean up properly.

---

## Summary

```
Layer 1 (direct control)      Layer 2 (managed orchestration)
───────────────────────       ──────────────────────────────
Memory                         MemoryStore (persistence)
Tool / StreamingTool           ToolRegistry (discovery)
LLMProvider                    AgentRegistry (metadata)
SimpleAgent                    AgentRuntime (task + queue)
AgentEvent hierarchy           Settings, Factory types
```

- Use **Layer 1** for simple scripts, custom agents, or embedding in existing frameworks.
- Use **Layer 2** for multi-conversation apps, server backends, any scenario needing persistence and discovery.
- Use **AgentRuntime** when you want concurrent runs with cancellation, event queues, and multi-agent handoff.

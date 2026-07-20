# Developer Guide

This guide walks through building agent applications using the Layer 1 (Core Abstractions) and Layer 2 (Service Abstractions) from `minimal-harness`. You can use these layers directly without ever touching the TUI.

---

## Choosing Your Abstraction Level

| Level | When to Use |
|-------|-------------|
| **Layer 1 only** | You want full control �?wire up agent, LLM, memory, and tools yourself |
| **Layer 2** | You want higher-level orchestration �?registries, runtime, persistent memory |
| **Layer 2 + Runtime** | You want multi-agent handoff, event-driven execution, task-based concurrency |

---

## Layer 1: Core Abstractions

Layer 1 gives you four protocols and an event system. You compose them directly.

### 1. Memory �?Conversation History

`Memory` stores messages and tracks token usage.

```python
from minimal_harness.memory import ConversationMemory

memory = ConversationMemory()
await memory.add_message({"role": "user", "content": [{"type": "text", "text": "Hello"}]})
await memory.add_message({"role": "assistant", "content": "Hi there!", "tool_calls": None})

for msg in memory.get_all_messages():
    print(msg["role"], msg["content"])
```

Helper constructors are available for each message type:

```python
from minimal_harness.memory import (
    user_message, assistant_message, tool_message, system_message
)

await memory.add_message(system_message("You are a helpful assistant"))
await memory.add_message(user_message([{"type": "text", "text": "Hello"}]))
await memory.add_message(assistant_message("Hi!", tool_calls=None))
await memory.add_message(tool_message("call_123", "command output"))
```

`get_forward_messages()` excludes `reasoning` messages (used when sending to LLM). `dump_memory()` returns a serializable dict for persistence.

### 2. Tools �?LLM-Callable Functions

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

**Detect the current language inside a tool:**

Tools can use `get_current_locale()` to produce localized output:

```python
from minimal_harness.agent.runtime import get_current_locale

async def my_tool() -> AsyncIterator[dict]:
    locale = get_current_locale()
    if locale == "zh":
        yield {"message": "处理完成"}
    else:
        yield {"message": "Processing complete"}
```

The locale is set by the application (e.g. from `Accept-Language` header) and propagated through the runtime context to all tools, including sub-tasks spawned by `handoff`. See [Language Detection in user_tool_writing.md](user_tool_writing.md#language-detection) for more detail.

**Wrap it into a `StreamingTool`:**

```python
from minimal_harness.tool.base import StreamingTool

reverse_tool = StreamingTool(
    name="reverse",
    display_name="Reverse",
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
    display_name="Reverse",
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
    print(event)  # ToolStart �?ToolProgress �?ToolEnd
```

**Built-in tools** (`bash`, file operations) live in
[`mh-tui`](https://github.com/J0ey1iu/mh-tui) as `mh_tui.built_in`:

```python
from mh_tui.built_in import get_tools as get_builtin_tools
all_tools = get_builtin_tools()  # {"bash": ..., "local_file_operation": ...}
```

The SDK has no tools of its own. To use built-in tools outside the TUI,
import them from `mh_tui.built_in` (or copy the ~400-line module �?
it depends only on `minimal_harness.tool.base` / `.types`).

### 3. LLM Provider �?Talk to an LLM

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

### 4. Agent �?The Execution Loop

`SimpleAgent` runs the standard agentic loop: user input �?LLM �?tool calls �?LLM �?...

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

Pass extra parameters (e.g., disable thinking/reasoning on supported models) via `llm_kwargs`:

```python
async for event in agent.run(
    user_input=[{"type": "text", "text": "Hello"}],
    stop_event=stop_event,
    memory=memory,
    tools=[],
    # OpenAI o-series models: disable reasoning
    llm_kwargs={"reasoning_effort": None},
    # Anthropic Claude 3.7+: disable extended thinking
    # llm_kwargs={"thinking": {"type": "disabled"}},
    # DeepSeek: disable thinking (if supported)
    # llm_kwargs={"no_think": True},
):
    ...
```

The same `llm_kwargs` parameter is also accepted by `AgentRuntime.run()`.

### 5. Middleware �?Lifecycle Hooks

`Middleware` (`agent/middleware.py`) lets you inject logic into the agent lifecycle:

```python
from minimal_harness.agent.middleware import Middleware
from minimal_harness.types import LLMEnd

class MyMiddleware(Middleware):
    async def on_llm_end(self, event: LLMEnd) -> None:
        print(f"LLM call used {event.usage.total_tokens} tokens")

    async def should_allow_tool(self, tool_call, **kwargs) -> bool | str:
        if tool_call["function"]["name"] == "bash":
            reason = "Bash execution is not allowed"
            print(reason)
            return reason  # returning a str vetoes; bool(True) allows
        return True
```

Pass middleware to `SimpleAgent`:

```python
agent = SimpleAgent(
    llm_provider=provider,
    max_iterations=10,
    middleware=[MyMiddleware()],
)
```

Or to `AgentRuntime`:

```python
runtime = AgentRuntime(
    agent_registry=agent_registry,
    session_store=store,
    tool_registry=tool_registry,
    middleware=[MyMiddleware()],
    llm_provider_resolver=lambda _: provider,
)
```

### 5.1. Auto-Compacting Agents

`CompactionAgent` (`agent_type="compacting"`) runs the same loop as
`SimpleAgent` (they share `BaseAgent` �?see `agent/base.py`) but
auto-folds older messages into a streaming summary whenever the
cumulative `prompt_tokens` from the LLM exceeds a configured
threshold. Use it for long-running multi-turn conversations that would
otherwise run out of context.

#### Wiring

```python
from minimal_harness import CompactionSettings, CompactionSummarizer

async def my_summarizer(messages, existing_summary):
    """Streaming summarizer. Yield summary text chunk by chunk."""
    async for chunk in call_llm_to_summarize(messages, existing_summary):
        yield chunk

# summarizer_factory: callable that takes an LLMProvider and returns
# a CompactionSummarizer. The runtime closes over the same LLM
# provider the agent loop uses.
runtime = AgentRuntime(
    ...,
    compaction_summarizer_factory=lambda llm: my_summarizer,  # type: ignore[arg-type]
    default_compaction_settings=CompactionSettings(
        prompt_token_threshold=8000,
        keep_recent=6,
    ),
)

await agent_registry.register(AgentMetadata(
    name="long_runner",
    agent_type="compacting",       # �?enables CompactionAgent
    system_prompt="...",
    tool_names=[...],
    # Per-agent overrides �?leave unset to fall back to
    # ``default_compaction_settings`` on the runtime.
    compaction=CompactionSettings(
        prompt_token_threshold=12000,
        keep_recent=4,
    ),
))
```

`CompactionSummarizer` is a callable that returns an
`AsyncIterator[str]`. Pass any function whose `yield` produces the
summary text �?typically an OpenAI / Anthropic streaming chat call.
`prompt_token_threshold` is checked against the *cumulative*
`prompt_tokens` tracked by `Memory.get_message_usage()` after every
LLM call; crossing it triggers `Memory.compact()` (after the assistant
turn has been recorded in the buffer). `keep_recent` is the number of
tail messages kept verbatim (default 6).

Compaction is **soft-fail**: if the summarizer raises, the agent logs
a warning, surfaces the failure through `CompactionEnd(error=...)`,
and continues the run. The LLM's reply is preserved in memory and
visible to the user; the next iteration will retry compaction on the
unchanged buffer.

#### Manual compaction (`/compact`)

Call `runtime.compact_session(memory_id)` to fold an existing session
outside the agent loop. This yields the same
`CompactionStart / CompactionChunk / CompactionEnd` event stream as
the auto-compaction path, so any consumer wired to the agent events
(sessions controller, display layer, replay) works without changes.

The summarizer is built from the runtime's
`compaction_summarizer_factory`; the threshold and `keep_recent` come
from the session's owning agent's `CompactionSettings`, falling back
to the runtime's `default_compaction_settings`. This is the same path
the TUI's `/compact` slash command drives �?there is no longer a
separate "submit a prompt" hack.

#### Per-turn event order

For a single LLM turn that crosses the compaction threshold, events
stream out in this order:

```
AgentStart
LLMStart
LLMChunk...
LLMEnd
MessageEvent(reasoning)   ─�?
MessageEvent(assistant)    ├─ PRIMARY content (the LLM's actual reply)
CompactionStart           ─�?
CompactionChunk...         ├─ HOUSEKEEPING (the fold)
CompactionEnd              �?
MessageEvent(compaction)  ─�?
AgentEnd
```

The two layers are **decoupled**: frontends see the raw LLM reply via
`MessageEvent(assistant)`, while the next LLM call only sees the
compacted buffer via `get_forward_messages()`. If the buffer is so
large that the just-added assistant falls inside the `keep_recent`
fold region, it gets summarised too �?the frontend is still informed
via the raw `MessageEvent(assistant)` it already received.

#### Observing compaction

Three new event types stream out of the agent run while a compaction is
in progress:

```python
from minimal_harness import CompactionStart, CompactionChunk, CompactionEnd

async for event in agent.run(...):
    match event:
        case CompactionStart():
            print(f"compacting {event.dropped_message_count} msgs "
                  f"(prior summary {len(event.existing_summary or '')} chars)")
        case CompactionChunk():
            update_preview(event.accumulated)
        case CompactionEnd():
            print(f"done in {event.duration:.2f}s, summary {len(event.summary)} chars")
```

`CompactionEnd.error` is set when the summarizer raises mid-stream; in
that case `event.summary == ""` (the partial streamed text is not
reported as a valid fold) and the memory buffer is left unchanged.
The LLM's assistant turn is still recorded in memory and emitted as
`MessageEvent(assistant)`, and the agent loop continues �?compaction
is a soft-fail. The run ends normally with `AgentEnd.error=None` and
`response` set to the assistant text.

#### Middleware hooks

```python
class CompressionLogger(Middleware):
    async def on_compaction_start(self, event: CompactionStart) -> None:
        metrics.increment("compaction.started", tags={"trigger": str(event.prompt_tokens)})

    async def on_compaction_end(self, event: CompactionEnd) -> None:
        metrics.histogram("compaction.duration", event.duration)
        metrics.histogram("compaction.summary_chars", len(event.summary))
        if event.error:
            metrics.increment("compaction.failed")
```

### 6. Events Reference

All events are `@dataclass` types, unified under `AgentEvent`:

| Event | Fields |
|-------|--------|
| `AgentStart` | `user_input`, `timestamp` |
| `AgentEnd` | `response`, `time_taken`, `exceeded`, `interrupted`, `error` |
| `LLMStart` | `messages`, `tools` |
| `LLMChunk` | `chunk: LLMChunkDelta \| None` |
| `LLMEnd` | `content`, `reasoning_content`, `tool_calls`, `usage`, `error` |
| `CompactionStart` | `dropped_message_count`, `existing_summary`, `keep_recent`, `prompt_tokens`, `timestamp` |
| `CompactionChunk` | `delta`, `accumulated`, `timestamp` |
| `CompactionEnd` | `summary` ("" on failure), `dropped_message_count` (0 on failure), `new_offset`, `duration`, `error?`, `timestamp` |
| `MessageEvent(role=compaction)` | emitted only on successful compaction; `meta` carries `dropped_count` / `keep_recent` / `previous_summary_chars` / `timestamp` |
| `ExecutionStart` | `tool_calls` |
| `ExecutionEnd` | `results: list[(ToolCall, Any)]`, `error`, `should_stop`, `response_text` |
| `ToolStart` | `tool_call` |
| `ToolProgress` | `tool_call`, `chunk` |
| `ToolEnd` | `tool_call`, `result` |
| `MemoryUpdate` | `usage` |
| `MessageEvent` | `message` |

---

## Layer 2: Service Abstractions

Layer 2 provides registries, persistent memory, and the `AgentRuntime` orchestrator.

### 1. Registries �?Discoverable Components

**`ToolRegistry`** �?register and look up tool metadata by name:

```python
from minimal_harness.tool.registry import ToolRegistry
from minimal_harness.types import ToolMetadata, LocalToolBinding, ExternalScriptToolBinding

tool_registry = ToolRegistry()

# Register a local tool (async generator function)
await tool_registry.register(ToolMetadata(
    name="reverse",
    display_name="Reverse",
    description="Reverse a given string",
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string", "description": "Text to reverse"}},
        "required": ["text"],
    },
    binding=LocalToolBinding(fn=reverse_text),
))

# Or use the convenience shortcut:
await tool_registry.register_from_binding(
    name="reverse",
    description="Reverse a given string",
    parameters={"type": "object", "properties": {...}},
    binding=LocalToolBinding(fn=reverse_text),
)

# Register an external script tool (runs as subprocess):
await tool_registry.register_from_binding(
    name="my_external_tool",
    description="A tool implemented in an external .py file",
    parameters={"type": "object", "properties": {}},
    binding=ExternalScriptToolBinding(script_path="/path/to/tool.py"),
)

await tool_registry.names()           # ["reverse", "my_external_tool"]
await tool_registry.get("reverse")    # -> ToolMetadata
await tool_registry.get_all()         # -> list[ToolMetadata]
```

#### RemoteTool �?Execute Tools via HTTP

```python
from minimal_harness.tool.registry import ToolRegistry
from minimal_harness.types import ToolMetadata, RemoteToolBinding
from minimal_harness.tool.remote import SSEToolExecutor, RemoteTool

# Register a remote tool (executed via HTTP)
await tool_registry.register(ToolMetadata(
    name="weather",
    description="Get weather for a city",
    parameters={"type": "object", "properties": {"city": ...}},
    binding=RemoteToolBinding(
        url="https://my-tool-service.example.com/weather",
        driver="default",  # uses SSEToolExecutor by default
        headers={"Authorization": "Bearer xxx"},
    ),
))

# Or with a custom executor for a non-SSE protocol:
class MyCustomExecutor:
    async def execute(self, args, tool_call, stop_event):
        # custom protocol logic here...
        yield ToolStart(tool_call)
        yield ToolProgress(tool_call, {"status": "processing"})
        yield ToolEnd(tool_call, {"result": "done"})

await tool_registry.register(ToolMetadata(
    name="custom_tool",
    description="...",
    parameters={...},
    binding=RemoteToolBinding(url="...", driver="my_driver"),
))

# Register the custom executor factory via AgentRuntime:
runtime = AgentRuntime(
    ...
    tool_executor_factories={
        "my_driver": ToolExecutorFactory(lambda binding: MyCustomExecutor()),
    },
)
```

Register built-in tools in bulk:

```python
from minimal_harness.tool.registry import collect_builtin_tools
await collect_builtin_tools(tool_registry)  # returns set of names registered
```

#### Convenience: `@register_tool` decorator

For quick tool registration without manually constructing `ToolMetadata`:

```python
from minimal_harness.tool.registration import register_tool, register_decorated_tools

@register_tool(
    name="reverse",
    display_name="Reverse",
    description="Reverse a string",
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
    display_name_locale={"zh": "反转"},
    description_locale={"zh": "反转字符�?},
)
async def reverse_text(text: str) -> AsyncIterator[dict]:
    yield {"success": True, "result": text[::-1]}

# In your async setup, register all decorated tools:
await register_decorated_tools(tool_registry)
```

If `registry` is passed to `@register_tool(registry=tool_registry)`, it registers
immediately (synchronously via `asyncio.create_task`). The recommended pattern is
to omit `registry` and call `register_decorated_tools()` during async setup.

**`AgentRegistry`** �?register agent metadata:

```python
from minimal_harness.types import AgentMetadata
from minimal_harness.agent.registry import AgentRegistry

agent_registry = AgentRegistry()
await agent_registry.register(AgentMetadata(
    name="coder",
    display_name="Coder",
    description="Writes and debugs code",
    system_prompt="You are a coding expert.",
    agent_type="simple",
    tool_names=["bash", "read_file", "write_file"],
))
```

**`ToolRegistryProtocol`** and **`AgentRegistryProtocol`** are `@runtime_checkable`, so you can substitute custom implementations.

### 2. MemoryStore �?Persistent Conversations

> **0.7.0 调整**：`Session` / `SimpleSession` / `SessionStoreProtocol`
> 整体迁出 SDK。具�?Session 实现位于�?
> - `mh-tui` �?`JsonlSessionStore`（`~/.minimal_harness/sessions/`）�?�?[`mh-tui` 源码](https://github.com/J0ey1iu/mh-tui)
> - `mh-gateway` �?`BuiltinSessionStore`（在 `mh_gateway.database`）�?�?[`mh-gateway` 源码](https://github.com/J0ey1iu/mh-gateway)
>
> SDK 现在只暴露一个最小的 `MemoryStoreProtocol` —�?只需要实�?`get_session(id) -> Memory | None` 即可。`AgentRuntime` 不再关心 `user_id` / `scenario_id` 等身份字段。本节示例改�?`ConversationMemory` 直接做内�?store�?

```python
from minimal_harness.memory import ConversationMemory, MemoryStoreProtocol


class InMemoryMemoryStore:
    """Minimal in-memory store for the SDK's AgentRuntime."""

    def __init__(self) -> None:
        self._cache: dict[str, ConversationMemory] = {}

    async def get_session(self, memory_id: str) -> ConversationMemory | None:
        return self._cache.get(memory_id)

    def add(self, memory_id: str, memory: ConversationMemory) -> None:
        self._cache[memory_id] = memory


store = InMemoryMemoryStore()
        return sess

    async def get_session(self, session_id: str) -> SimpleSession | None:
        return self._cache.get(session_id)

    async def save_memory(self, memory, session_id, extra=None) -> None:
        pass  # in-memory only

    async def delete_session(self, session_id: str) -> bool:
        return self._cache.pop(session_id, None) is not None

    async def list_sessions(self) -> list[dict]:
        return []

    async def list_user_sessions(self, user_id, scenario_id=None) -> list[dict]:
        return []

    async def get_session_messages(self, session_id):
        sess = await self.get_session(session_id)
        return [dict(m) for m in sess.get_all_messages()] if sess else []

    def get_messages_as_items(self, session):
        return [dict(m) for m in session.get_all_messages()]


store = InMemorySessionStore()

# Create a new conversation
session = await store.create_session(session_id="conv_001", agent_name="coder")

# Retrieve later
session = await store.get_session("conv_001")
for msg in session.get_all_messages():
    print(msg)

# Persist
await store.save_memory(session.memory, "conv_001")

# Delete
await store.delete_session("conv_001")
```

`MemoryStoreProtocol` (in `memory.py`) is intentionally minimal. The
richer `Session` contract (with `user_id`, `scenario_id`,
`display_name_locale`, `title`) lives in
`mh_gateway.database._session`. mh-tui ships its own
copy in `mh_tui._session_types`. Use one of those if you need
identity-aware persistence; the SDK only needs `get_session()` to
run an agent.

### 3. Configuration from Environment

> **0.7.0 change:** `Settings` has been removed from the SDK. Each
> consumer reads `MH_*` env vars directly:
>
> - `mh-tui.config.defaults` �?`MH_BASE_URL`, `MH_API_KEY`, `MH_MODEL`, `MH_THEME`, `MH_MAX_ITERATIONS`
> - `mh-service_kit.logging_setup.setup_service_logging` �?`MH_LOG_LEVEL`, `MH_LOG_DIR`
>
> In your own code, prefer reading env vars with `os.environ.get()`.

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

### 5. AgentRuntime �?The Core Orchestrator

`AgentRuntime` ties everything together. Given metadata IDs, it resolves agents, tools, memory, and runs the agent loop as a background `asyncio.Task`.

```python
from minimal_harness.agent.runtime import AgentRuntime
from minimal_harness.llm import create_llm_provider

# Wire up Layer 2 components
runtime = AgentRuntime(
    agent_registry=agent_registry,
    session_store=store,
    tool_registry=tool_registry,
    llm_provider_resolver=lambda _: create_llm_provider({
        "provider": "openai",
        "model": "gpt-4o",
        "base_url": "https://api.openai.org/v1",
        "api_key": "sk-...",
    }),
)

# Register runtime tools (handoff, discover_agents) �?moved to mh-tui
from mh_tui.runtime_tools import register_runtime_tools
await register_runtime_tools(
    agent_registry=agent_registry,
    session_store=store,
    tool_registry=tool_registry,
    run_fn=runtime.run,
)

# Start a run
task, stop_event, event_queue = runtime.run(
    user_input=[{"type": "text", "text": "Write a Python script to sort a list"}],
    agent_metadata_id="coder",
    memory_id="conv_001",
    context={"locale": "zh"},  # tools can read this via get_current_locale()
)

# Disable thinking/reasoning on supported models:
task, stop_event, event_queue = runtime.run(
    user_input=[{"type": "text", "text": "Write a Python script to sort a list"}],
    agent_metadata_id="coder",
    memory_id="conv_001",
    llm_kwargs={"reasoning_effort": None},  # OpenAI o-series
    # llm_kwargs={"thinking": {"type": "disabled"}},  # Anthropic Claude 3.7+
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

`AgentRuntimeProtocol` is `@runtime_checkable` �?you can write your own runtime that satisfies the same interface.

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
    session_store=...,
    tool_registry=...,
    agent_factory=my_agent_factory,
    llm_provider_resolver=...,
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

tool = create_streaming_tool("my_tool", my_tool, display_name="My Tool")
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

> **0.7.0 调整**：`register_runtime_tools` �?`JsonlSessionStore` 已迁�?SDK�?
> 本节示例不再调用 `register_runtime_tools`（运行时工具是应用层概念�?
> 详见 [Pattern 3](#pattern-3-multi-agent-handoff-layer-2)）�?
> SessionStore 使用内联 `InMemorySessionStore`；生产请�?[`mh-gateway`](https://github.com/J0ey1iu/mh-gateway)
> �?SQLite 实现或自实现 `SessionStoreProtocol`�?

```python
from minimal_harness.agent.runtime import AgentRuntime
from minimal_harness.agent.registry import AgentRegistry
from minimal_harness.tool.registry import ToolRegistry, collect_builtin_tools
from minimal_harness.types import AgentMetadata

# (InMemorySessionStore omitted for brevity �?see Section 2 above)

# Setup
tool_registry = ToolRegistry()
await collect_builtin_tools(tool_registry)

agent_registry = AgentRegistry()
await agent_registry.register(AgentMetadata(
    name="assistant", display_name="Assistant",
    description="General assistant",
    system_prompt="You are helpful.", agent_type="simple",
    tool_names=["bash", "read", "write"],
))

store = InMemorySessionStore()
runtime = AgentRuntime(
    agent_registry=agent_registry,
    session_store=store,
    tool_registry=tool_registry,
    llm_provider_resolver=lambda _: create_llm_provider(...),
)

# Each user request:
session = await store.create_session()
task, stop, queue = await runtime.run(
    user_input=[{"type": "text", "text": user_message}],
    agent_metadata_id="assistant",
    memory_id=session.session_id,
)
```

### Pattern 3: Multi-Agent Handoff (Layer 2)

> **0.7.0 调整**：`handoff` / `discover_agents` 运行时工具及 `register_runtime_tools()`
> 现托管于 [`mh-tui`](https://github.com/J0ey1iu/mh-tui) 包：
> `from mh_tui.runtime_tools import register_runtime_tools`�?
> 它们是多 Agent 应用层概念（同一进程内的子任务委托）�?
> 因此跟随 TUI 走而非留在 SDK。若你正在构建一个不使用 mh-tui 的服务，
> 可在自建服务中重新实现等价工具�?

Runtime tools (`handoff` and `discover_agents`) must be registered via `register_runtime_tools()`. Your system prompt should reference them:

```python
await agent_registry.register(AgentMetadata(
    name="triage",
    display_name="Triage",
    description="Routes tasks to specialist agents",
    system_prompt=(
        "You are a triage agent. Use discover_agents to find "
        "available specialists, then handoff tasks to them."
    ),
    agent_type="simple",
    tool_names=[],  # handoff/discover_agents injected by register_runtime_tools
))
await agent_registry.register(AgentMetadata(
    name="coder",
    display_name="Coder",
    description="Writes and debugs code",
    system_prompt="You are a coding expert.",
    agent_type="simple",
    tool_names=["bash"],
))
```

When LLM calls `handoff(target_agent_name="coder", ...)`, Runtime spawns a child task with the coder agent �?events from the child are tunneled back through the parent's queue.

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
            print(f"╔═�?Agent Run ══�?)
        elif isinstance(event, LLMChunk) and event.chunk:
            if event.chunk.content:
                buffer += event.chunk.content
                print(event.chunk.content, end="", flush=True)
        elif isinstance(event, ToolStart):
            print(f"\n  ┌─ {event.tool_call['function']['name']}")
        elif isinstance(event, ToolProgress):
            print(f"  �?{event.chunk}")
        elif isinstance(event, ToolEnd):
            print(f"  └─ Result: {str(event.result)[:100]}")
        elif isinstance(event, AgentEnd):
            print(f"\n╚═�?{event.time_taken:.2f}s ══�?)
```

---

## What Not to Do

1. **Don't import Layer 2 from Layer 1 code** �?Layer 1 (`agent/`, `llm/`, `memory.py`, `tool/base.py`) must not import `Settings`, registries, runtime, or memory store. Pass dependencies explicitly via `__init__`.

2. **Don't hardcode concrete implementations** �?Accept protocols (`Agent`, `LLMProvider`, `Memory`, `Tool`, `AgentRuntimeProtocol`, etc.) rather than constructing specific classes. This keeps your code testable and swappable.

3. **Don't call `Queue.get()` without timeout in production** �?Use `asyncio.wait_for(queue.get(), timeout=...)` to avoid blocking indefinitely if the task stalls.

4. **Don't forget to call `register_runtime_tools()`** �?Without it, `handoff` and `discover_agents` will not be available to agents. Note: `register_runtime_tools` is a standalone function in `tool.built_in.runtime_tools`, not a method of `AgentRuntime`.

5. **Don't rely on `asyncio.Task` reference escaping** �?The runtime returns the task; ensure you `await task` after setting `stop_event` to clean up properly.

---

## Summary

```
Layer 1 (direct control)      Layer 2 (managed orchestration)
───────────────────────       ──────────────────────────────
Memory                         SessionStore (persistence)
Tool / StreamingTool           ToolRegistry (discovery)
LLMProvider                    AgentRegistry (metadata)
SimpleAgent                    AgentRuntime (task + queue)
AgentEvent hierarchy           Settings, Factory types
```

- Use **Layer 1** for simple scripts, custom agents, or embedding in existing frameworks.
- Use **Layer 2** for multi-conversation apps, server backends, any scenario needing persistence and discovery.
- Use **AgentRuntime** when you want concurrent runs with cancellation, event queues, and multi-agent handoff.

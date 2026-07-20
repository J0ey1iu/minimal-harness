# minimal-harness

**Documentation: [/docs](./docs/)**

A lightweight Python agent SDK for building LLM-powered agents with tool-calling support.

Latest version: **0.7.0a2**

> **Heads up �?TUI moved out (0.7.0):** The Textual-based TUI client that
> previously shipped as `minimal_harness.client.built_in` now lives in
> its own package: [`mh-tui`](https://github.com/J0ey1iu/mh-tui).
> Install it separately with `pip install mh-tui`. The `mhc` CLI command
> is preserved.

> **Umbrella:** This SDK is one of five packages wired together in the
> [`mh-incubator`](https://github.com/J0ey1iu/mh-incubator) workspace.
> For the full picture (services, gateway, TUI, frontend) see the
> umbrella README.

## What This Project Is For

Minimal-harness is a lean SDK for building agents that can call tools. It provides:

- **OpenAI/Anthropic-compatible API** - Works with OpenAI, Anthropic, or any OpenAI-compatible API provider
- **Multi-modal image input** - Pass image URLs or base64 data to LLM providers supporting vision
- **Symmetric Registry + Factory architecture** - Register tool/agent metadata with bindings (`LocalToolBinding`, `RemoteToolBinding`, `ExternalScriptToolBinding`); executable instances created lazily by `ToolFactory`
- **Middleware hooks** - Observe and intercept the agent lifecycle (agent start/end, LLM calls, tool execution, tool policy enforcement, compaction start/end)
- **AsyncIterator events** - Real-time async iteration for chunks, tool start/end, execution events, compaction progress
- **Conversation memory sessions** - Persistent sessions with identity (user_id, scenario_id), auto-persisted to disk
- **Auto-compaction** - `CompactionAgent` (`agent_type="compacting"`) folds older messages into a streaming summary whenever the LLM's `prompt_tokens` exceeds a configured threshold, enabling arbitrarily long conversations
- **Remote tools** - Pluggable `RemoteToolExecutor` Protocol; default SSE-over-HTTP executor lives in `mh-service-kit`
- **ESC stop support** - Gracefully stop LLM streaming and tool execution

## Reference applications

`minimal-harness` is the SDK. There are four sibling packages that build
on it:

| Layer | Repo | Shape |
|---|---|---|
| Service SDK | [J0ey1iu/mh-service-kit](https://github.com/J0ey1iu/mh-service-kit) | FastAPI helpers, SSE engine, service logger |
| Local TUI | [J0ey1iu/mh-tui](https://github.com/J0ey1iu/mh-tui) | Local-running, single-user Textual TUI (includes `bash` / `local_file_operation` built-in tools as `mh_tui.built_in`) |
| Cloud gateway | [J0ey1iu/mh-gateway](https://github.com/J0ey1iu/mh-gateway) | Multi-tenant FastAPI gateway with sessions, eval, M2M auth |

## Architecture

The SDK is a **single-layer framework**:

```
┌──────────────────────────────────────────�?
�? Framework (this package)                �?
�? Protocols, types, in-memory primitives  �?
�? Agent loop · Registries · Memory        �?
�? LLM providers · Event types             �?
└──────────────────────────────────────────�?
       �?             �?          �?
       �?             �?          �?
   mh-tui     mh-service-kit    mh-gateway
   (TUI)      (FastAPI service)  (multi-tenant gateway)
```

Everything above this layer �?sessions, persistence, executors,
logging, the TUI, the gateway �?lives in the sibling packages.

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

### 1a. Layer 1 �?Direct Control

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

### 1b. Layer 2 �?Managed Orchestration

```python
from minimal_harness.agent.runtime import AgentRuntime
from minimal_harness.agent.registry import AgentRegistry
from minimal_harness.tool.registry import ToolRegistry, collect_builtin_tools
from minimal_harness.types import AgentMetadata
from minimal_harness.session import SimpleSession


class InMemorySessionStore:
    """Minimal in-memory session store �?replace with your own backend."""

    def __init__(self) -> None:
        self._cache: dict[str, SimpleSession] = {}

    async def create_session(
        self,
        session_id: str | None = None,
        agent_name: str = "",
        user_id: str = "",
        scenario_id: str | None = None,
        transient: bool = False,
        display_name_locale: str | None = None,
    ) -> SimpleSession:
        from uuid import uuid4

        sid = session_id or uuid4().hex
        sess = SimpleSession(
            session_id=sid,
            agent_name=agent_name,
            user_id=user_id,
            scenario_id=scenario_id,
            display_name_locale=display_name_locale,
        )
        self._cache[sid] = sess
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


tool_registry = ToolRegistry()
await collect_builtin_tools(tool_registry)

agent_registry = AgentRegistry()
await agent_registry.register(AgentMetadata(
    name="assistant", display_name="Assistant",
    description="General assistant",
    system_prompt="You are helpful.", agent_type="simple",
    tool_names=["bash", "local_file_operation"],
))

store = InMemorySessionStore()

runtime = AgentRuntime(
    agent_registry=agent_registry,
    session_store=store,
    tool_registry=tool_registry,
    llm_provider_resolver=lambda _: create_llm_provider(...),
)

session = await store.create_session()
task, stop, queue = await runtime.run(
    user_input=[{"type": "text", "text": user_message}],
    agent_metadata_id="assistant",
    memory_id=session.session_id,
)
```

> **Note:** If you need the `handoff` and `discover_agents` runtime
> tools, they now ship in the `mh-tui` package as
> `mh_tui.runtime_tools.register_runtime_tools()`. They are application
> glue (multi-agent coordination) rather than core SDK functionality, so
> they live alongside the TUI that uses them.

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

Or use the `@register_tool` decorator (recommended pattern �?omit `registry` and call `register_decorated_tools()` during async setup):

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
    # registry=...  # optional �?see below
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

The SDK ships no tools of its own. The `bash` and `local_file_operation`
tools live in [`mh-tui`](https://github.com/J0ey1iu/mh-tui) as
`mh_tui.built_in` (they're application-level concerns that the TUI
happens to ship). To use them outside the TUI, copy the module �?it's
about 400 lines and depends only on `minimal_harness.tool.base` and
`minimal_harness.types`.

```python
from mh_tui.built_in import collect_builtin_tools, get_tools

# Register them into a ToolRegistry in one call
await collect_builtin_tools(tool_registry)  # �?set[str] of names

# Or use the Tool instances directly
for name, tool in get_tools().items():
    print(name, tool.display_name)
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
| `AgentEnd`        | `response`, `time_taken`, `exceeded`, `interrupted`, `error` | Agent execution completed   |
| `LLMStart`        | `messages`, `tools`                                    | LLM generation started          |
| `LLMChunk`        | `chunk: LLMChunkDelta \| None`                         | LLM output chunk received       |
| `LLMEnd`          | `content`, `reasoning_content`, `tool_calls`, `usage`, `error` | LLM generation completed |
| `CompactionStart` | `dropped_message_count`, `existing_summary`, `keep_recent`, `prompt_tokens` | `Memory.compact()` triggered (CompactionAgent only) |
| `CompactionChunk` | `delta`, `accumulated`                                 | Streaming summary delta (CompactionAgent only) |
| `CompactionEnd`   | `summary`, `dropped_message_count`, `new_offset`, `duration`, `error?` | Compaction completed (CompactionAgent only) |
| `ExecutionStart`  | `tool_calls`                                           | Tool execution started          |
| `ExecutionEnd`    | `results`, `error`, `should_stop`, `response_text`     | Tool execution completed        |
| `ToolStart`       | `tool_call`                                            | Tool call started               |
| `ToolProgress`    | `tool_call`, `chunk`                                   | Tool intermediate progress      |
| `ToolEnd`         | `tool_call`, `result`                                  | Tool call completed with result |
| `MemoryUpdate`    | `usage`                                                | Memory token usage updated      |
| `MessageEvent`    | `message`                                              | Conversation message added to memory |

`LLMChunkDelta` contains `content`, `reasoning`, and `tool_calls` fields for provider-agnostic partial deltas.

### Batch Evaluation

The `minimal_harness.eval` module has been removed. Use the
`mh-gateway`'s eval API or
[`POST /api/v1/eval/batch`](https://github.com/J0ey1iu/mh-gateway)
to run agent evaluation campaigns.

### Environment Variables

The SDK no longer reads environment variables. The `MH_*` env vars are
read by their respective consumers:

| Variable             | Read by |
| -------------------- | --- |
| `MH_BASE_URL`, `MH_API_KEY`, `MH_MODEL` | `mh-tui.config.defaults` |
| `MH_MAX_ITERATIONS` | `mh-tui.config.defaults` |
| `MH_LOG_LEVEL`, `MH_LOG_DIR` | `mh-service-kit.setup_service_logging` |
| `MH_THEME` | `mh-tui.config.defaults` |

### Stop Mechanism

Pass an `asyncio.Event` to `agent.run(..., stop_event=event)` and
`event.set()` it from any concurrent task (e.g. an HTTP handler, a key
press handler) to gracefully stop LLM streaming and tool execution.
The TUI (`mh-tui`) wires this to the `Esc` key.

# Event-Driven Mechanism

This document describes the event-driven architecture in minimal-harness, which enables real-time observation and control of agent execution.

## Overview

The system uses a single-layer event model. All event types are defined in `src/minimal_harness/types.py` and consumed directly by the consumer:

```
┌──────────────────────────────────────────────────────────────────┐
│                        SimpleAgent                               │
│  (yields AgentEvent: AgentStart, AgentEnd, LLMChunk,             │
│   ExecutionStart, LLMEnd, LLMStart, MemoryUpdate,                │
│   ToolStart, ToolProgress, ToolEnd)                              │
└─────────────────────────┬────────────────────────────────────────┘
                          │ async generator
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Consumer                                    │
│  (receives AgentEvent directly from types.py)                    │
└──────────────────────────────────────────────────────────────────┘
```

No separate client-event layer exists. The `minimal_harness.client.events` module is a backward-compat shim that simply re-exports from `types.py`.

## Event Types

All events are defined in `src/minimal_harness/types.py`:

| Event | Fields | Description |
|-------|--------|-------------|
| `AgentStart` | `user_input: Iterable[ExtendedInputContentPart]`, `timestamp: float` | Emitted when agent begins execution |
| `AgentEnd` | `response: str`, `time_taken: float \| None`, `exceeded: bool`, `interrupted: bool` | Emitted when agent finishes execution |
| `LLMChunk` | `chunk: LLMChunkDelta \| None` | Streaming chunk from LLM |
| `ExecutionStart` | `tool_calls: list[ToolCall]` | Emitted before tool execution |
| `ExecutionEnd` | `results: list[tuple[ToolCall, Any]]` | Emitted after tool execution completes |
| `LLMStart` | `messages: list[Message]`, `tools: Any` | Emitted when LLM starts processing |
| `LLMEnd` | `content: str \| None`, `reasoning_content: str \| None`, `tool_calls: list[ToolCall]`, `usage: TokenUsage \| None` | Emitted when LLM finishes with complete result and usage |
| `MemoryUpdate` | `usage: TokenUsage` | Emitted when memory usage is updated |
| `ToolStart` | `tool_call: ToolCall` | Emitted when a tool starts |
| `ToolProgress` | `tool_call: ToolCall`, `chunk: Any` | Progress update during streaming tool |
| `ToolEnd` | `tool_call: ToolCall`, `result: Any` | Emitted when a tool finishes |

### Provider-Agnostic Delta Types

`LLMChunkDelta` is the building block for streaming chunks:

| Field | Type | Description |
|-------|------|-------------|
| `content` | `str \| None` | Text content delta |
| `reasoning` | `str \| None` | Reasoning/thinking content delta |
| `tool_calls` | `list[ToolCallDelta] \| None` | Tool call deltas |

`ToolCallDelta` contains `index`, `id`, `name`, and `arguments` fields for partial tool call accumulation.

## Event Flow

### 1. Agent Execution Flow

```
1. SimpleAgent.run() is called
         │
         ▼
2. Yields AgentStart(user_input)
         │
         ▼
3. LLM processes user input
          │
          ├──► Yields LLMStart(messages, tools)
          │
          ├──► Yields LLMChunk(chunk) for each streaming token
          │
          ├──► Yields LLMEnd(content, reasoning_content, tool_calls, usage)
          │
          ▼
4. If tool_calls exist:
         │
         ▼
5. Yields ExecutionStart(tool_calls)
         │
         ▼
6. For each tool_call:
         │
         ├──► StreamingTool.execute() yields ToolStart
         │         │
         │         ├──► Yields ToolProgress for each chunk
         │         │
         │         └──► Yields ToolEnd with result
         │
         ▼
7. Yields ExecutionEnd(results)
         │
         ▼
8. Yields AgentEnd(response_text, time_taken, exceeded)
```

## Usage

```python
from minimal_harness.types import (
    AgentEnd,
    ToolStart,
    ToolProgress,
    ToolEnd,
)

async def main():
    async for event in agent.run(
        user_input=[{"type": "text", "text": "..."}],
        memory=memory,
        tools=tools,
        context={"locale": "zh"},                       # optional: runtime context
        llm_kwargs={"reasoning_effort": None},           # optional: LLM SDK params
    ):
        if isinstance(event, ToolStart):
            print(f"Tool started: {event.tool_call['function']['name']}")
        elif isinstance(event, ToolProgress):
            print(f"Progress: {event.chunk}")
        elif isinstance(event, ToolEnd):
            print(f"Tool ended: {event.result}")
        elif isinstance(event, AgentEnd):
            print(f"Agent finished: {event.response}")
```

## Iterator Pattern

The `Agent.run()` method returns an `AsyncIterator[AgentEvent]` that yields events as they occur. Use `async for` to consume events:

```python
async for event in agent.run(
    user_input=[{"type": "text", "text": "..."}],
    memory=memory,
    tools=tools,
    context={"locale": "en"},
    llm_kwargs={"max_tokens": 4096},
):
    if isinstance(event, ToolStart):
        print(f"Tool started: {event.tool_call['function']['name']}")
    elif isinstance(event, ToolEnd):
        print(f"Tool ended: {event.result}")
    elif isinstance(event, AgentEnd):
        print(f"Agent finished: {event.response}")
```

All events are yielded in real-time during agent execution. No callbacks are used — the iterator pattern provides a cleaner, more Pythonic way to observe agent behavior.

## Streaming Tools

Tools implement the `Tool` protocol and yield events during execution:

```python
class StreamingTool(Tool):
    def __init__(self, name, description, parameters, fn, display_name=None):
        self.display_name = display_name or name
        ...

    async def execute(self, args, tool_call, stop_event) -> AsyncIterator[ToolEvent]:
        yield ToolStart(tool_call)
        async for chunk in self.fn(**args):
            if stop_event and stop_event.is_set():
                break
            yield ToolProgress(tool_call, chunk)
        yield ToolEnd(tool_call, final_result)
```

A streaming tool function has signature:
```python
StreamingToolFunction = Callable[..., AsyncIterator[Any]]
```

## Stop Mechanism

The `stop_event: asyncio.Event` parameter allows external cancellation:

1. Pass `stop_event` to `Agent.run()`
2. Set `stop_event.set()` to request cancellation
3. The agent/tool checks `stop_event.is_set()` at yield points and stops gracefully

## Type Hierarchy

```
AgentEvent (Union)
├── AgentStart
├── AgentEnd
├── ExecutionEnd
├── ExecutionStart
├── LLMChunk
├── LLMEnd
├── LLMStart
├── MemoryUpdate
├── ToolEnd
├── ToolProgress
└── ToolStart

ToolEvent (Union)
├── ToolStart
├── ToolProgress
└── ToolEnd
```

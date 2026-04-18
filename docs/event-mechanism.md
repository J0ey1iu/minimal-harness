# Event-Driven Mechanism

This document describes the event-driven architecture in minimal-harness, which enables real-time observation and control of agent execution.

## Overview

The system uses a two-layer event model:

```
┌─────────────────────────────────────────────────────────────────┐
│                        OpenAIAgent                              │
│  (yields AgentEvent: AgentStart, AgentEnd, Chunk,               │
│   ExecutionStart, LLMEnd, LLMStart, ToolStart, ToolProgress, ToolEnd)             │
└─────────────────────────┬───────────────────────────────────────┘
                          │ async generator
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              _agent_event_to_client_event()                    │
│  (converts AgentEvent → Event)                                  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FrameworkClient                            │
│  (yields Event: AgentStartEvent, AgentEndEvent, ChunkEvent,     │
│   ExecutionStartEvent, LLMEndEvent, LLMStartEvent, ToolStartEvent, ToolProgressEvent,       │
│   ToolEndEvent)                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Event Types

### Internal Events (`src/minimal_harness/types.py`)

Base events used internally by the agent and tools:

| Event | Fields | Description |
|-------|--------|-------------|
| `AgentStart` | `user_input: Iterable[ExtendedInputContentPart]` | Emitted when agent begins execution |
| `AgentEnd` | `response: str` | Emitted when agent finishes execution |
| `Chunk` | `chunk: Any \| None`, `is_done: bool` | Streaming chunk from LLM |
| `ExecutionStart` | `tool_calls: list[ToolCall]` | Emitted before tool execution |
| `LLMStart` | - | Emitted when LLM starts processing |
| `LLMEnd` | `content: str \| None`, `tool_calls: list[ToolCall]`, `usage: TokenUsage \| None` | Emitted when LLM finishes with complete result and usage |
| `ToolStart` | `tool_call: ToolCall` | Emitted when a tool starts |
| `ToolProgress` | `tool_call: ToolCall`, `chunk: Any` | Progress update during streaming tool |
| `ToolEnd` | `tool_call: ToolCall`, `result: Any` | Emitted when a tool finishes |

### Client Events (`src/minimal_harness/client/events.py`)

Public-facing events for framework consumers:

| Event | Fields | Description |
|-------|--------|-------------|
| `AgentStartEvent` | `user_input: Iterable[ExtendedInputContentPart]` | Agent started |
| `AgentEndEvent` | `response: str` | Agent finished |
| `ChunkEvent` | `chunk: Any \| None`, `is_done: bool` | LLM streaming chunk |
| `ExecutionStartEvent` | `tool_calls: list[ToolCall]` | Tool execution about to begin |
| `LLMStartEvent` | - | LLM started processing |
| `LLMEndEvent` | `content: str \| None`, `tool_calls: list[ToolCall]`, `usage: TokenUsage \| None` | LLM finished with complete result and usage |
| `ToolStartEvent` | `tool_call: ToolCall`, `_` (deprecated) | Tool started |
| `ToolProgressEvent` | `tool_call: ToolCall`, `chunk: Any` | Tool streaming progress |
| `ToolEndEvent` | `tool_call: ToolCall`, `result: Any` | Tool finished |

## Event Flow

### 1. Agent Execution Flow

```
1. OpenAIAgent.run() is called
         │
         ▼
2. Yields AgentStart(user_input)
         │
         ▼
3. LLM processes user input
          │
          ├──► Yields LLMStart()
          │
          ├──► Yields Chunk(chunk, False) for each streaming token
          │
          ├──► Yields LLMEnd(content, tool_calls, usage)
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
7. Yields AgentEnd(response_text)
```

### 2. Event Conversion

The `_agent_event_to_client_event()` function (`client/client.py:34-51`) maps internal events to client events:

```python
AgentStart          → AgentStartEvent
AgentEnd            → AgentEndEvent
Chunk               → ChunkEvent
ExecutionStart      → ExecutionStartEvent
LLMStart            → LLMStartEvent
LLMEnd              → LLMEndEvent
ToolStart           → ToolStartEvent (with None for deprecated field)
ToolProgress        → ToolProgressEvent
ToolEnd             → ToolEndEvent
```

## Usage

```python
from minimal_harness.client import FrameworkClient
from minimal_harness.client.events import (
    AgentEndEvent,
    ToolStartEvent,
    ToolProgressEvent,
    ToolEndEvent,
)

async def main():
    async for event in framework_client.run(user_input=[{"type": "text", "text": "..."}]):
        if isinstance(event, ToolStartEvent):
            print(f"Tool started: {event.tool_call['function']['name']}")
        elif isinstance(event, ToolProgressEvent):
            print(f"Progress: {event.chunk}")
        elif isinstance(event, ToolEndEvent):
            print(f"Tool ended: {event.result}")
        elif isinstance(event, AgentEndEvent):
            print(f"Agent finished: {event.response}")
```

## Iterator Pattern

The `Agent.run()` method returns an `AsyncIterator[AgentEvent]` that yields events as they occur. Use `async for` to consume events:

```python
from minimal_harness.client import FrameworkClient
from minimal_harness.client.events import AgentEndEvent, ToolStartEvent, ToolEndEvent

async def main():
    async for event in framework_client.run(user_input=[{"type": "text", "text": "..."}]):
        if isinstance(event, ToolStartEvent):
            print(f"Tool started: {event.tool_call['function']['name']}")
        elif isinstance(event, ToolEndEvent):
            print(f"Tool ended: {event.result}")
        elif isinstance(event, AgentEndEvent):
            print(f"Agent finished: {event.response}")
```

All events are yielded in real-time during agent execution. No callbacks are used — the iterator pattern provides a cleaner, more Pythonic way to observe agent behavior.

## Streaming Tools

Tools implement the `StreamingTool` interface (`tool/base.py`) and yield events during execution:

```python
class StreamingTool:
    def __init__(self, name, description, parameters, fn):
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

1. Pass `stop_event` to `FrameworkClient.run()` or `OpenAIAgent.run()`
2. Set `stop_event.set()` to request cancellation
3. The agent/tool checks `stop_event.is_set()` at yield points and stops gracefully

## Type Hierarchy

```
AgentEvent (Union)
├── AgentStart
├── AgentEnd
├── Chunk
├── ExecutionStart
├── LLMEnd
├── LLMStart
├── ToolStart
├── ToolProgress
└── ToolEnd

ToolEvent (Union)
├── ToolStart
├── ToolProgress
└── ToolEnd

Event (Union) [Client-facing]
├── AgentStartEvent
├── AgentEndEvent
├── ChunkEvent
├── ExecutionStartEvent
├── LLMEndEvent
├── LLMStartEvent
├── ToolStartEvent
├── ToolProgressEvent
└── ToolEndEvent
```

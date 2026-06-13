# minimal-harness Framework Evaluation Report

This report evaluates `minimal-harness` v0.6.0 based on a thorough review of its
source code, documentation, tests, and real-world usage in the `mh-application`
project (a full-stack web chat app built atop the framework).

## What It Does Well

### 1. Clean, Well-Enforced Layered Architecture

Three strict layers (L1→Core Abstractions, L2→Service Abstractions,
L3→Application) with enforced import discipline. All concrete classes sit behind
Protocols, making the design testable and extensible. The architecture docs
explicitly document every cross-layer violation and how each was resolved.

### 2. Event-Driven AsyncIterator Design

The entire framework is built around `AsyncIterator[AgentEvent]`. No callbacks,
no pub/sub boilerplate — just `async for event in agent.run(...)`. This is the
right pattern for LLM streaming and fits naturally into any async Python
application.

### 3. Provider-Agnostic LLM Abstraction

OpenAI and Anthropic are unified behind `LLMChunkDelta` (content, reasoning,
tool_calls). The `LLMProvider` Protocol makes adding new providers
straightforward. The `Stream[T]` primitive captures the full response
(`LLMResponse`) after exhaustion, solving the streaming-vs-final-response
tension elegantly.

### 4. Parallel Tool Execution

`SimpleAgent._execute_tools()` spawns tool calls as `asyncio.Task` pool with an
`asyncio.Queue` for interleaved progress events. Multiple tools run concurrently
and their events stream interleaved — a critical feature for real-world agent
usage.

### 5. Graceful Interruption Propagation

`stop_event` is threaded through every layer: LLM streaming
(`await_with_interrupt`), tool execution (bash kill-on-cancel, file operation
cancellation), and the agent loop itself (`asyncio.CancelledError` handling).
Thoroughly designed and battle-tested.

### 6. Rich Conversation Memory Model

`ConversationMemory` supports 5 message types (System, User, Assistant, Tool,
Reasoning), content parts (text/image/file), token usage tracking, JSON
serialization, and auto-persistence. More comprehensive than most agent
frameworks.

### 7. External Tool Loading System

`ExternalToolWrapper` with subprocess execution, shebang detection, and inline
runner code generation is sophisticated. Tools defined in arbitrary `.py` files
can be loaded at runtime and executed in isolated subprocesses. A
differentiating feature.

### 8. Multi-Agent Handoff

The `handoff` and `discover_agents` tools enable agent-to-agent delegation.
Child agent runs tunnel their events back as tool progress events. Combined
with the agent registry, this enables multi-agent orchestration patterns.

### 9. Comprehensive Documentation

7 doc files covering README, dev guide, architecture, event mechanism,
stop/interrupt design, tool writing guide, and external loading. The docs are
well-structured and include diagrams, tables, and concrete examples.

### 10. High-Quality TUI Client

> 0.7.0 抽离至独立包 [`mh-tui`](https://github.com/J0ey1iu/mh-tui)。

28-module Textual client with streaming markdown rendering, session management,
config UI, tool selection, SVG export, and theme support. Validates the
framework APIs from a real consumer perspective.

### 11. Middleware System Proven in Production

The `Middleware` hook system (added in v0.5.1) is used by `mh-application` to
enforce role-based access control via `PermissionMiddleware`. The
`should_allow_tool()` hook intercepts every tool execution and returns a denial
reason string visible to the LLM. This validates the middleware design works
for real-world authorization, observability, and safety gating.

## What It Lacks

### 1. No Structured Output Support

The LLM providers only pass `tool_choice="auto"` — there is no
`response_format` parameter for forcing JSON schemas or structured outputs. This
is a table-stakes feature for agent frameworks.

### 2. No Built-In HTTP/SSE Transport Layer

The framework has no web server, no REST API, and no standard event
serialization format. The `mh-application` had to build 60+ lines of
`_serialize_event()` manually converting every event dataclass to JSON. Each
application must reinvent this wheel.

**What should exist**: A `minimal_harness.server` package providing
`EventSerializer` + `SSEFormatter` + optional FastAPI router.

### 3. No LLM Error Recovery (Retry/Backoff)

LLM calls have zero retry logic, no rate-limit handling, no exponential
backoff. If the provider returns a 429 or the connection drops, the agent loop
terminates. Real production usage requires this.

### 4. No Tool Approval / Human-in-the-Loop

Tools execute immediately with no confirmation gate. There's no mechanism to
pause the agent loop and wait for user approval before executing a tool. This
is a safety concern for shell execution and file operations.

### 5. No Memory Summarization / Context Compaction

`ConversationMemory` accumulates messages indefinitely. There's no
summarization, sliding window, or token-based truncation. Long conversations
will hit context limits with no recourse. A `summarize()` or `compact()` method
on `Memory` is essential.

### ~~6. No Middleware / Hook System~~

**Status**: ✅ Implemented (`agent/middleware.py`)

`Middleware` base class provides hooks: `on_agent_start`, `on_agent_end`,
`on_llm_start`, `on_llm_end`, `on_tool_start`, `on_tool_end`, `on_tool_error`,
`should_allow_tool`, and `on_error`. Used by `EvalCollector` and production
applications for authorization (e.g. `PermissionMiddleware`).

### 7. No Observability (Tracing / Logging / Metrics)

No OpenTelemetry integration, no structured logging, no token/cost tracking
abstraction. Building production agents requires knowing token usage, latency
breakdowns, and error rates. The framework provides none of this.

### 8. No Pagination or Truncation for Tool Outputs

File reads and bash executions can produce massive outputs with no built-in
limits. Large tool results can flood the LLM context. A `max_output_length`
parameter on tools is needed.

### 9. Synchronous Disk Persistence on Every Mutation

> 0.7.0：`JsonlSessionStore` 已迁出至 [`mh-tui`](https://github.com/J0ey1iu/mh-tui)。
> 本节描述的"逐消息写盘"行为依然是 mh-tui 的实现选择。

`JsonlSessionStore` persists to disk on every message mutation. For chat
applications with token-by-token streaming, this means potentially hundreds of
disk writes per message. A debounced/batched persistence strategy is needed
(partially addressed: `set_message_usage` now uses debounced fire-and-forget writes).

### 10. No Model Routing

`AgentRuntime` uses a single `llm_provider_factory`. There's no way to route
different agents to different models or switch models mid-session. Multi-model
orchestration requires this.

### 11. No Conversation Branching/Forking

No support for creating conversation branches from a checkpoint. This limits
use cases like A/B testing prompts or exploring alternative conversation paths.

### 12. No Built-In Caching

LLM responses are never cached. Repeated queries within a session or across
sessions re-execute identically. A TTL-based cache on `LLMProvider` would
reduce costs and latency.

### ~~13. Multi-Modal Input Is Defined But Not Functional~~

**Status**: ✅ Implemented

`ImageContentPart` supports optional `data`/`media_type` fields for base64 image
data. Anthropic provider sends native image blocks; OpenAI provider converts to
`image_url` blocks. Multi-modal image input is fully functional.

### 14. No Sandbox for Bash Tool

Shell execution runs on the host OS with no container/sandbox isolation. For a
framework used to build coding agents, this is a significant security gap.

### 15. No Production Deployment Artifacts

No Dockerfile, no docker-compose, no Kubernetes manifests, no production
deployment documentation. The framework is development-only by default.

### 16. Tool Output Not Typed

Tool results are `Any`. There's no `ToolResult` type or schema validation for
tool outputs. The `mh-application` had to write custom `_serialize_result()`
logic with type-checking gymnastics.

### 17. No Event Serialization Standard

Agent events are plain `@dataclass` objects with no `to_dict()` or
`from_dict()` methods. The `mh-application` wrote a **60-line `match/case`
serializer** (`agent_service.py:132-192`) that manually extracts every field
from 9 event types. There is no deserialization support at all — events cannot
be reconstructed from JSON. Adding a new event type to the framework silently
returns `{}` with no type error.

### ~~18. No Session Abstraction~~

**Status**: ✅ Implemented (`session.py`)

Session was promoted from L3 to L2 in v0.6.0. `Session` Protocol enriches
Memory with identity fields (`session_id`, `user_id`, `scenario_id`, `title`,
`agent_name`). `SessionSummary` provides listing metadata (with `user_id`
filtering). `SessionStoreProtocol` manages CRUD at the session level.

### ~~19. Event Queue Lacks Completion Signal~~

**Status**: ✅ Fixed

`AgentRuntime.run()` now attaches a `done_event` to the returned `Task` object
(`task.done_event`), set in the producer's `finally` block so consumers can use
`asyncio.wait()` instead of polling.

### ~~20. Memory Extra Metadata Not Preserved~~

**Status**: ✅ Fixed

`_ManagedMemory.dump_memory()` now overrides to inject metadata (`memory_id`,
`title`, `created_at`, `agent_name`) into the `extra` dict, and `_persist()`
passes the managed memory (not the inner) so the override is used.

### ~~21. No Multi-Tenant / User Support in MemoryStore~~

**Status**: ✅ Partially fixed

`SessionStoreProtocol` now includes `list_user_sessions(user_id, scenario_id=None)`
for session listing with user filtering. The protocol supports multi-tenant
isolation at the interface level.

### 22. Config Duplication Between Framework and App

> **Resolved in 0.7.0:** the framework's `Settings` class has been
> removed. Each consumer (TUI, service) reads env vars directly. The
> `MH_*` env prefix is now documented as part of the
> `mh-tui.config.defaults` module.

## Summary Table

| Area | Status | Priority |
|---|---|---|
| Architecture & Protocols | Strong | — |
| Event streaming | Strong | — |
| Parallel tool execution | Strong | — |
| Interruption handling | Strong | — |
| External tools | Strong | — |
| Multi-agent handoff | Strong | — |
| Documentation | Strong | — |
| Eval module | **Implemented** | **Done** |
| Middleware/hooks | **Implemented** | **Done** |
| Session abstraction | **Implemented** | **Done** |
| Remote Agent/Tool (SSE) | **Implemented** | **Done** |
| Locale/i18n support | **Implemented** | **Done** |
| Multi-modal (functional) | **Implemented** | **Done** |
| Queue completion signal | **Fixed** | **Done** |
| Memory extra metadata | **Fixed** | **Done** |
| **HTTP/SSE transport** | **Partially done** | **Medium** |
| **Event serialization standard** | **Partially done** | **Medium** |
| **LLM retry/backoff** | **Missing** | **High** |
| **Tool approval (human-in-loop)** | **Missing** | **High** |
| **Memory summarization** | **Missing** | **High** |
| **Observability/tracing** | **Missing** | **Medium** |
| **Multi-tenant support** | **Partially fixed** | **Done** |
| Structured output (`response_format`) | Missing | Medium |
| Model routing | Missing | Medium |
| Tool output limits | Missing | Medium |
| Async/cached disk persistence | Missing | Medium |
| Conversation branching | Missing | Low |
| LLM response caching | Missing | Low |
| Bash sandbox/container isolation | Missing | Low |
| Production deployment (Docker, etc.) | Missing | Low |
| Typed tool output schema | Missing | Low |
| Config deduplication | Missing | Low |

> **Bottom line**: `minimal-harness` v0.6.0 has an excellent core agent loop with
> strong architectural foundations. Major gaps from v0.5 have been addressed:
> middleware hooks, session abstraction, multi-modal input, eval module, remote
> agent/tool support via SSE, and locale/i18n support. Still missing is the
> operational infrastructure for production applications: HTTP/SSE transport
> (the SSE primitives exist at the tool/agent level but there's no server
> package), a standard event serialization format, LLM retry/backoff, tool
> approval gates, and memory summarization. The highest-impact improvements
> would be: a standard event serialization format, LLM retry/backoff, tool
> approval gates, and memory summarization.

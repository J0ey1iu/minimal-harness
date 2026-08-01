# Change log

## 0.8.0a1

- feat(controller): new Controller layer wrapping Agent runs —
  `DefaultController` (passthrough, agent events untouched),
  `GoalController` (judge-driven
  DONE/NEXT loop, `max_goal_rounds`), `TimerController` (duration-bounded
  loop with forced continuation). New `ControllerStart / Continue / End`
  events, `ControllerRegistry` on `AgentRuntime` with per-request
  `controller_type` / `controller_config`. Judge calls forward
  `stop_event` so user stops interrupt internal LLM calls.

## 0.7.0a8

- chore: lockstep pre-release bump for the `mh-local` ship chain.
  No SDK code change.

## 0.7.0a7

- chore: lockstep pre-release bump. No SDK code change.

## 0.7.0a6

- chore: aligned pre-release bump with `mh-gateway==0.1.0a5`
  (no SDK code change; release only).
- feat(compaction): drop `compaction_summarizer_factory`; rewrite
  the summariser as a chat-payload composition (issue #23).
- feat(agent): `settings_schema` added to agent factories for
  dynamic type discovery (PR #22).
- feat(llm): forward `llm_config` params (temperature, max_tokens,
  etc.) to LLM API calls (PR #21).
- feat(tool): add `ToolCompactionAgent` with tool-result
  compression and full-conversation compaction (PR #20,
  PR #17).
- feat(context): surface per-message usage to drive a context
  percentage visualisation in the UI (PR #19).
- fix: pass `trust_env=False` to the OpenAI / Anthropic SDKs'
  `httpx` clients so system proxy env vars no longer hijack
  outbound calls in proxy-enabled environments (PR #16).

## 0.7.0a5

- chore: pre-release bump for `mh-gateway==0.1.0a4`
  no SDK code change; release only.

## 0.7.0a4

- chore: pre-release bump.

## 0.7.0a3

- chore: pre-release bump.

## 0.7.0a2

- chore: pre-release bump for the round-2 SDK-decoupling
  distribution; no SDK code change.

## 0.7.0a1

- chore: pre-release for the upcoming 0.7.0 work (round-2 SDK decoupling + compaction refactor)

## 0.7.0

> **BREAKING (round 2)**: The SDK has been further decoupled from
> application/service concerns. The following modules have been
> extracted to dedicated packages. The SDK is now strictly a
> framework — Protocols, types, in-memory primitives, and the
> agent loop.
>
> - **Built-in tools** (`bash`, `local_file_operation`,
>   `collect_builtin_tools`, `get_builtin_tool_names`, `collect_tools`)
>   → [`mh-tui`](https://github.com/J0ey1iu/mh-tui) as
>   `mh_tui.built_in`. They are application-level concerns that the
>   TUI happens to ship; the SDK has no tools of its own. Consumers
>   that need these tools outside the TUI can import them from
>   `mh_tui.built_in` or copy the ~400-line module (it depends only
>   on `minimal_harness.tool.base` / `.types`).
> - **SSE-over-HTTP wire protocol** (`SSEAgentDriver`,
>   `DefaultAgentDriverFactory`, `SSEAgentRunner`, `SSEAgentDriver`,
>   `SSEToolExecutor`, `ToolServiceExecutor`, `serialize_event`,
>   `deserialize_event`) and the **service-mode logger**
>   (`setup_service_logging`) → [`mh-service-kit`](https://github.com/J0ey1iu/mh-service-kit).
>   The Protocol abstractions (`RemoteAgentDriver`,
>   `RemoteAgentDriverFactory`, `RemoteToolExecutor`) stay in the SDK.
> - **Session persistence** (`Session`, `SimpleSession`, `SessionSummary`,
>   `SessionStoreProtocol`, `generate_bigint_id`) →
>   `mh_gateway.database`. The SDK now exposes a
>   thinner `MemoryStoreProtocol` whose only method is
>   `get_session(id) -> Memory | None`.
> - **Customer-deployment adapters** (`RegistryProvider`,
>   `MetadataManager`, `ToolProvider`) →
>   `mh_gateway.adapters`.
> - **Evaluation campaigns** (`EvalCollector`, `EvalPersistence`,
>   `generate_html_report`, `run_evaluation`, `run_evaluation_simple`)
>   → **removed entirely**. The orchestration service's
>   `mh_gateway.eval` package is now the only eval
>   entry point.
> - **`Settings` class** (`MH_*` env helpers) → **removed entirely**.
>   Each consumer (TUI, service) reads env vars directly.
> - **Examples** (`examples/`) → **removed entirely**. TUI-related
>   examples moved to `mh-tui/examples/`.
>
> Migration sketch:
>
> ```diff
> - from minimal_harness.tool.built_in.bash import bash_tool
> + from mh_tui.built_in import bash_tool
>
> - from minimal_harness.adapters import RegistryProvider
> + from mh_gateway.adapters import RegistryProvider
>
> - from minimal_harness.session import Session, SimpleSession
> - from minimal_harness.memory_store import SessionStoreProtocol
> + from mh_gateway.database._session import Session, SimpleSession
> + from mh_gateway.database._memory_store import SessionStoreProtocol
>
> - from minimal_harness.sse_serialization import serialize_event
> - from minimal_harness.agent.runner import SSEAgentRunner
> + from mh_service_kit.sse import serialize_event, SSEAgentRunner
>
> - from minimal_harness.client.logging_setup import setup_service_logging
> + from mh_service_kit import setup_service_logging
>
> - from minimal_harness.settings import Settings
> - Settings.base_url()  # was a class-method that read MH_BASE_URL
> + os.environ["MH_BASE_URL"]  # read env var directly
> ```
>
> `AgentRuntime` and `AgentFactory` no longer register a default
> SSE driver / executor. Consumers that use them in the default
> configuration must explicitly register:
>
> ```python
> from mh_service_kit.sse import (
>     DefaultAgentDriverFactory,
>     SSEToolExecutor,
> )
> from mh_gateway.database._protocol import ToolExecutorFactory  # type: ignore
>
> agent_factory = DefaultAgentFactory(
>     llm_provider_resolver=...,
>     driver_factories={"default": DefaultAgentDriverFactory()},
> )
> tool_factory = DefaultToolFactory(
>     executor_factories={"default": _SSEToolExecutorFactory()},
> )
> ```
>
> (The orchestration service wires this up for you in
> `create_app` / `runtime_service`.)

- **BREAKING** feat: built-in tools moved into `mh-tui` as
  `mh_tui.built_in` (the standalone `mh-builtin-tools` package was
  abandoned; only mh-tui used it). Use
  `from mh_tui.built_in import bash_tool, local_file_operation_tool,
  collect_builtin_tools, get_tools, collect_tools`.
- **BREAKING** refactor: remove `minimal_harness.tool.built_in`
  (use `mh_tui.built_in`)
- **BREAKING** refactor: remove `minimal_harness.collect_tools`
  (use `mh_tui.built_in.collect_tools`)
- **BREAKING** refactor: remove `minimal_harness.eval` (entire module
  gone; use `mh_gateway.eval` for HTTP-based eval)
- **BREAKING** refactor: remove `minimal_harness.adapters` (use
  `mh_gateway.adapters`)
- **BREAKING** refactor: remove `minimal_harness.session`,
  `minimal_harness.database`, `minimal_harness.memory_store`
  (use `mh_gateway.database._session`,
  `mh_gateway.database._ids`,
  `mh_gateway.database._memory_store`)
- **BREAKING** refactor: replace `SessionStoreProtocol` with the
  SDK's `MemoryStoreProtocol` (only `get_session(id) -> Memory | None`).
  Session identity fields now live in orchestration / mh-tui.
- **BREAKING** refactor: remove `minimal_harness.sse_serialization`,
  `minimal_harness.agent.runner` (use `mh_service_kit.sse`)
- **BREAKING** refactor: remove `minimal_harness.client.*` (use
  `mh_service_kit.logging_setup` for service logging; TUI has its
  own `mh_tui.logging_setup`)
- **BREAKING** refactor: remove `minimal_harness.settings.Settings`
- **BREAKING** chore: remove `DefaultAgentDriverFactory` /
  `DefaultToolExecutorFactory` from the SDK defaults. Consumers
  must register their own driver/executor factory when creating
  `DefaultAgentFactory` / `DefaultToolFactory` for remote agents /
  remote tools.
- **BREAKING** chore: remove `examples/` (TUI examples moved to
  `mh-tui/examples/`; the rest were tied to deleted modules)

## 0.7.0 (unreleased)

- refactor(core): extract `BaseAgent` (`agent/base.py`) from the
  `SimpleAgent` / `CompactionAgent` classes. Both agents now share a
  single agentic-loop implementation; `CompactionAgent` overrides a
  single `_post_llm_response` hook to inject the compaction step.
  This removes the ~40% duplication that lived between the two
  classes and ensures any future bug fix to the loop or tool
  execution flows to both at once.
- refactor(core): add `BaseMemory` (`memory.py`) as the default
  implementation base for the `Memory` Protocol. Downstream
  implementors (e.g. `JsonlManagedSession`, `SimpleSession`) that
  subclass `BaseMemory` get a clear, typed contract: missing
  `compact()` (or any other abstract method) raises
  `NotImplementedError` at instantiation, not buried in a streaming
  generator at runtime. Closes the regression where adding
  `compact()` to the protocol silently broke downstream stores
  (commit 766f13c).
- feat(core): add `AgentRuntime.compact_session(memory_id) -> AsyncIterator[CompactionEvent]`
  for manual compaction outside the agent loop. Yields the same
  `CompactionStart / CompactionChunk / CompactionEnd` event stream the
  auto-compaction produces; the summarizer is built from the runtime's
  `compaction_summarizer_factory`, with per-agent `CompactionSettings`
  on `AgentMetadata` taking precedence over the runtime's
  `default_compaction_settings`. This is the public API that the
  TUI's `/compact` slash command now drives — the legacy
  "submit-a-prompt-that-asks-the-LLM-to-summarise" hack is gone.
- design(compact): compaction is now **soft-fail**. When the
  summarizer raises, the agent logs a warning, surfaces the failure
  through `CompactionEnd(error=...)`, and continues the run. The
  LLM's assistant turn is still recorded in memory and emitted as
  `MessageEvent(assistant)`; the run ends normally with
  `AgentEnd.error=None` and `response` set to the assistant text. The
  next iteration will retry compaction on the unchanged buffer. (The
  previous behaviour of raising and terminating the run with
  `AgentEnd.error="Compaction failed: ..."` is gone — it silently
  dropped the LLM's reply.)
- design(memory): `_forward_offset` is now a regular in-memory
  attribute on `ConversationMemory` (always 0 after a fold), not
  stored in `_extra["compact_offset"]`. The implementation detail
  no longer leaks to persisted `MemoryData` dumps.
- feat(core): `AgentMetadata.compaction` is now a typed
  `CompactionSettings` (TypedDict with `prompt_token_threshold` and
  `keep_recent` keys, both optional). The runtime's
  `compact_summarizer_factory` is the new way to inject the
  summarizer; the previous `AgentRuntime(compaction_config=...)`
  singleton is replaced by per-agent settings that override the
  runtime defaults.
- feat(core): add `CompactionAgent` (`agent_type="compacting"`) — runs
  the same loop as `SimpleAgent` but auto-folds older messages into a
  streaming summary whenever `LLMEnd.usage["prompt_tokens"]` exceeds a
  configured `prompt_token_threshold`. The user supplies a streaming
  `summarizer: Callable[[list[Message], str | None], AsyncIterator[str]]`
  via the new `CompactionConfig`, injected through
  `AgentRuntime(compaction_config=...)`.
- feat(core): add `Memory.compact()` to the `Memory` Protocol — yields
  `CompactionStart` once, zero or more `CompactionChunk`s, and exactly
  one `CompactionEnd` (with `error` set on failure). The folded messages
  are replaced by a single synthetic `CompactionMessage` (role="compaction",
  distinct from any real assistant turn) at index 0; `_forward_offset`
  stays at 0 since the summary is the natural start of the compacted
  conversation. `get_forward_messages()` re-projects the summary to
  `role="assistant"` for the LLM. `dump_memory`/`load_memory` round-trip
  the summary and offset transparently.
- feat(core): add three new agent events — `CompactionStart`,
  `CompactionChunk`, `CompactionEnd` — and a `CompactionEvent` union.
  `CompactionChunk.delta` / `.accumulated` mirror the streaming summary
  text so frontends and eval collectors can render progress.
- feat(core): add `MessageEvent` carrying a `message` dict (role ∈
  user/assistant/reasoning/compaction) so frontends can replay the raw
  turn-by-turn stream. The compaction summary surfaces as
  `role="compaction"`; LLM-bound views re-project it to `role="assistant"`.
- feat(core): add `Middleware.on_compaction_start` /
  `Middleware.on_compaction_end` hooks (no-op by default) for
  observability, metering, and data collection.
- feat(core): add `DefaultAgentFactory.register_local_agent_factory`
  pre-registration for `agent_type="compacting"` (via the new
  `CompactingAgentFactory`).
- design(compact): record the LLM's assistant turn in memory and emit
  `MessageEvent(assistant)` **before** running `Memory.compact()`. This
  decouples "what the user sees" (raw MessageEvent stream) from "what
  the next LLM call sees" (compacted buffer / `get_forward_messages()`).
  If the buffer is so large that the just-added assistant falls inside
  the `keep_recent` fold region, it is summarised along with the older
  turns; the frontend is still informed via the raw event it already
  received. `CompactionEnd.summary == ""` on summarizer failure (no
  partial streamed text is reported as a valid fold). On compact
  failure the LLM's assistant turn is still recorded in memory and
  emitted as a `MessageEvent` before the agent loop terminates with
  `AgentEnd.error` wrapping `Compaction failed: ...`.

## 0.7.0 (round 1)

> **BREAKING**: The TUI client (`minimal_harness.client.built_in` and the
> `mhc` CLI entry point) has been extracted into a separate package
> [`mh-tui`](https://github.com/J0ey1iu/mh-tui). The SDK is now
> application-agnostic and no longer depends on `textual` or `rich`.
>
> Migration:
>
> ```diff
> - from minimal_harness.client.built_in import TUIApp
> + from mh_tui import TUIApp
> ```
>
> ```bash
> pip install mh-tui  # the TUI is no longer installed with minimal-harness
> ```
>
> Notes:
> - The `mhc` CLI command is preserved (now provided by `mh-tui`).
> - The `~/.minimal_harness/` config directory is preserved verbatim —
>   no user data migration needed.
> - The `handoff` and `discover_agents` runtime tools now live in
>   `mh-tui.runtime_tools`. `register_runtime_tools()` is still callable
>   from the same path within `mh-tui` (no service-side replacements).

- **BREAKING** feat: extract TUI into standalone `mh-tui` package
- **BREAKING** chore: remove `textual` and `rich` runtime dependencies
- **BREAKING** refactor: remove `minimal_harness.client.built_in` (use `mh_tui`)
- **BREAKING** refactor: remove `minimal_harness.tool.built_in.runtime_tools`
  (use `mh_tui.runtime_tools`)
- **BREAKING** refactor: remove `minimal_harness.client.logging_setup.setup_logging`
  (use `mh_tui.logging_setup.setup_logging`); `setup_service_logging` is
  retained for service-mode use
- chore: remove `aiosqlite` runtime dependency (was only used by the
  now-removed `SqliteSessionStore`)

## 0.6.2

- feat(core): add `stop` flag to `ToolResult` for early agent loop termination
- feat(core): add `verify_ssl` to `RemoteToolBinding`/`RemoteAgentBinding` for SSL verification control
- fix(core): add `index_lock` to prevent concurrent `os.replace()` race on `_index.json`
- refactor(core): remove `adopt_logger()` — logging uses root logger only
- docs: add programmatic stop mechanism docs for `ToolResult.stop`
- docs: sync docs with current codebase

## 0.6.2a12

- chore: bump alpha version for mh-gateway compatibility

## 0.6.1

- feat(core): add ToolResult mechanism to separate LLM content from UI metadata; optimize tool output display
- feat(core): add reasoning_effort config with Off/Default/Low/Medium/High options
- feat(core): add llm_extra_headers_provider callback for injecting custom headers into LLM API requests
- feat(core): add per-working-directory config via resolve_config_dir()
- feat(core): add extra_headers_provider lazy auth injection for remote tool/agent bindings
- feat(core): add file logging to ~/.minimal_harness/log/ with daily rotation
- feat(core): ring terminal bell on session completion
- feat(core): add correlation_id injection and unified log format for observability
- feat(core): add DEBUG inbound/outbound logging for all packages
- feat(core): move logging_setup out of built_in, add service-mode logging
- feat(core): complete zh/en locale for handoff and discover_agents runtime tools
- feat(core): add ConfigProvider protocol for external config center integration
- feat(core): merge AgentMetadataProvider/ToolMetadataProvider/ScenarioProvider into RegistryProvider
- feat(core): add OpenGaussDatabase, DatabaseBackend registry, and generate_bigint_id
- feat(core): add get_scenario to RegistryProvider protocol
- fix(core): return None for assistant message content instead of string 'null'
- fix(core): upgrade Memory.add_message to async with tiered persistence
- fix(core): prevent tool execution deadlock and lost ToolEnd events
- fix(core): fix exception handling in agent runner, tool remote, memory, and simple agent
- fix(core): fix SSE protocol envelope, error handling, and LLM provider robustness
- fix(core): don't mutate input schema dict in make_remote_tool()
- fix(core): always pass enable_thinking in extra_body
- fix(core): add defensive SSE event field access for SSEToolExecutor and ToolServiceExecutor
- fix(core): x-reasoning-format header value should be reasoning_content
- fix(core): ensure Event Start/End pairing and propagate error info
- fix(core): incremental append storage with sort_order
- fix(core): gracefully skip failed tool creation instead of crashing AgentRuntime
- fix(tui): prevent TUI freeze on large bash output with truncation and throttling
- fix(tui): platform-native notification sound and background session bell detection
- fix(tui): Windows-safe atomic write, lazy orphaned tmp cleanup, robust list_sessions
- fix(tui): ErrorHandler singleton __init__ reset bug, add stale tmp cleanup
- fix(tui): resolve WinError 183, error display, @ file picker issues
- fix(core): add transaction support (begin/commit/rollback/executemany) to DatabaseProtocol
- fix(core): support transient in database
- refactor(core): rename TokenVerifier → UserAuthProvider
- refactor(core): move tool_service_url from runner into per-tool endpoint_url
- refactor(core): replace DiskSessionStore with SqliteSessionStore
- refactor(core): extract AgentFactory from AgentRuntime._create_agent
- refactor(core): remove AgentRuntime.register_runtime_tools, add standalone function for TUI
- refactor(core): move ConfigProvider/SecretResolver to mh-gateway
- refactor(auth): TokenVerifier.verify receives full request, add UserIdentity.extra_data
- refactor(core): move session store SQL into each database backend
- refactor(core): add adapter protocols for customer deployment
- refactor(tui): improve error handling, startup perf, and UX
- docs: update docs for new features

## 0.6.0.post1

- fix(tui): fix @ command trigger conditions — only activate when preceded by whitespace/start-of-string and followed by non-whitespace
- fix(tui): fix @ keyword extraction — stop at whitespace to prevent space-as-keyword freeze on Windows
- fix(tui): add 2s timeout to rglob fallback in @ file picker to prevent TUI freeze
- feat(tui): append trailing space after inserting path via @ file picker

## 0.6.0

- feat(core): add symmetric Registry/ToolMetadata/Binding architecture (LocalToolBinding, RemoteToolBinding, ExternalScriptToolBinding)
- feat(core): add ToolFactory / DefaultToolFactory and ToolExecutorFactory for lazy tool instantiation from metadata
- feat(core): add tool.remote module — RemoteTool, RemoteToolExecutor (Protocol), SSEToolExecutor
- feat(core): add agent.remote module — RemoteAgent backed by RemoteAgentDriver Protocol
- feat(core): add agent.driver module — RemoteAgentDriver, RemoteAgentDriverFactory, SSEAgentDriver (SSE-over-HTTP driver)
- feat(core): promote Session from L3 to L2 (session.py), rename MemoryStoreProtocol → SessionStoreProtocol, add SessionSummary
- feat(core): add RegistryChangeEvent to Registry listeners (action/name/item payload)
- feat(core): register_tool decorator now captures ToolMetadata; support deferred registration via register_decorated_tools()
- feat(core): add AgentMetadata.binding field for LocalAgentBinding / RemoteAgentBinding
- feat(core): add metadata_id validation in AgentMetadata/ToolMetadata __post_init__
- feat(core): add MessageEvent protocol + SSEAgentRunner for remote agent streaming
- feat(core): delegate SSEAgentRunner to SimpleAgent + ToolServiceExecutor
- feat(eval): add eval module — EvalRunner, EvalCase, EvalReport, persistence, and report generation
- feat(eval): enrich eval report page with per-run detail pages and timeline visualization
- feat(tui): add error reporting panel (Error Notifications + ErrorScreen modal with code viewer)
- feat(tui): add @ file/directory picker command (Ctrl+P style) with git ls-files for performance
- feat(tui): top-bar turns green with theme-aware $success color during streaming
- feat(tui): Ctrl+Y copies assistant response (popup list to pick when multiple responses exist)
- fix(tui): restore @ command functionality after async refactor
- feat(tui): streaming content always updates even when user scrolls up (no silent freeze)
- perf(tui): add MarkdownRenderCache throttle to avoid full re-parse on every streaming tick
- perf(tui): replace widget remove/mount jitter with display=False during streaming transitions
- perf(tui): add 80ms debounce for slash/at command input detection
- perf(tui): early return in poll_background_completions when no background sessions
- refactor(core): collect_builtin_tools now registers ToolMetadata with LocalToolBinding instead of Tool objects
- refactor(core): ToolRegistry.register_from_binding replaces register_external_tool
- refactor(core): registry listeners receive RegistryChangeEvent instead of bare notification
- refactor(core): ToolRegistry stores ToolMetadata (not Tool) — execution deferred to ToolFactory
- refactor(core): AgentMetadata and ToolMetadata resolve_display_name/resolve_description locale methods
- fix(core): fix input history navigation — only move on first/last line boundary
- fix(core): correct LLMStart event ordering and eliminate _current_context cross-task leak
- fix(registry): fix metadata defects (3,4,7,8,11) — name validation, binding propagation, locale defaults
- docs: add architecture.md, dev-guide.md, eval-guide.md with Binding/Factory architecture docs
- docs: add usage examples for disabling model thinking via llm_kwargs
- docs: update example_use_tui.py and add eval_demo.py example

## 0.5.7

- refactor(core): convert entire framework from sync to fully async APIs (RegistryProtocol, memory stores, runtime, session controller, TUI)
- feat(core): add llm_kwargs pass-through from AgentRuntime to SDK API calls (temperature, max_tokens, extra_headers, etc.)
- fix(core): add per-memory asyncio.Lock and debounced auto-save to prevent concurrent file-write races in DiskMemoryStore
- fix(core): cancel sub-agent task in handoff_fn finally block so Escape/interrupt stops sub-agents
- fix(core): mark handoff memories as transient and filter from list_sessions to fix orphaned sessions in session selector
- fix(core): cancel all active runs on TUI exit
- fix(core): prevent TUI freeze from bash subprocess stealing stdin; ensure subprocess killed on Escape
- fix(test): update outdated UTs for sync-to-async refactor; remove redundant tests

## 0.5.6

- feat(core): add system_prompt_locale support to AgentMetadata for i18n system prompts

## 0.5.5

- feat(core): add display_name to Tool and AgentMetadata for human-readable UI labels
- feat(core): add locale support to tool/agent metadata (display_name_locale, description_locale, resolve_display_name, resolve_description)
- feat(core): add get_current_locale() for runtime locale detection in tools
- fix(core): add description_locale to local_file_operation and handoff; make list_tools and ToolStart locale-aware
- fix(core): align tool result storage with SSE serialization (filter private keys, consistent error prefix, safe default)
- fix(core): resolve pyright type errors in simple.py and test_context.py
- fix(core): pass agent_name to context in start_run so discover_agents excludes caller

## 0.5.4

- feat(core): add exclude parameter to Registry.get_all for filtering out the calling agent
- feat(core): discover_agents tool now excludes the calling agent via runtime context

## 0.5.3

- fix(core): add done_event to AgentRuntime and fix _ManagedMemory metadata persistence

## 0.5.2

- feat(core): add context propagation via ContextVar and should_allow_tool middleware hook
- feat(core): allow should_allow_tool to return a reason string for richer denial messages
- feat(llm): implement multi-modal image input for Anthropic and OpenAI providers
- feat(memory): add ImageContentPart with optional data/media_type fields for base64 image data
- fix(core): emit ToolStart before ToolEnd when tool is vetoed by middleware

## 0.5.1

- feat(core): add middleware hooks system for agent lifecycle observability
- feat(core): capture tool progress chunks during execution and persist in memory
- refactor(core): move tool progress storage into _execute_tools, define ToolMessage.progress
- docs: add developer guide for building apps on Layer 1 and Layer 2 abstractions

## 0.5.0

- refactor(core): enforce two-layer abstraction boundary between core and client services
- refactor(core): enforce stateless Agent protocol, remove ToolManager, delete PersistentMemory
- refactor(core): consolidate event system, fix functional bugs, clean up protocols
- refactor(core): resolve all P0/P1 architecture violations and add missing Protocols
- refactor(core): move system_prompt ownership from Memory to AgentMetadata
- refactor(core): move DiskMemoryStore into client layer, keep MemoryStoreProtocol in core
- refactor(core): unify AgentRegistry and ToolRegistry via generic Registry[T]
- refactor(core): unify runtime tools into standard tool pipeline
- refactor(core): remove dead on_chunk callback from LLM providers
- refactor(core): internalize agent creation in AgentRuntime, clean up parameter leaks
- refactor(core): add flush() to Memory protocol, add LLMProviderFactory and create_llm_provider
- refactor(core): simplify handoff tool event forwarding (truncated payloads)
- feat(core): parallel tool execution with grouped rendering
- fix(core): make interrupt signals respond immediately via task cancellation
- fix(core): update event type imports after types consolidation
- refactor(session): decouple handoff from session system, allow concurrent session runs
- feat(session): add session status management and live TUI visualization via listener pattern
- refactor(session): create fresh session per handoff, defer persistence until first user message
- feat(tui): show version number, live session status (● Running / ○ Idle), and current agent name in top-bar
- feat(tui): auto-start default agent on boot, keep input active during streaming
- feat(tui): session-completion notification and live session-select status update
- refactor(tui): extract StreamingController, ExportTracker, AgentManager, RunManager from display.py
- refactor(tui): extract action handlers into actions/ modules, ChatInput messages to messages.py
- refactor(tui): rename SessionManager to SessionReplayer, extract HandoffCoordinator and SessionFactory
- fix(tui): pass user-selected tools from session to runtime, record tool call/reasoning in SVG export
- fix(tui): move notification to top of screen, adjust modal dimensions and session list borders
- docs: add architecture design document with deep layering analysis and 21 identified issues
- docs: remove outdated refactor plans, sync docs with current code

## 0.4.5

- refactor(core): eliminate reverse dependency in event system and decouple tool schema from OpenAI SDK
- refactor(tui): extract StreamPresenter and ExportPresenter from TUIApp
- refactor(tui): address 10 more maintainability issues across built_in module
- refactor(tui): reduce SessionManager callbacks via TUIAppInterface protocol
- fix(tui): dynamic SVG export height and eliminate expensive markdown StringIO round-trip
- fix(tui): address 6 functional issues from Section 3 audit
- fix(memory): write session to disk immediately on every mutation (remove SAVE_THRESHOLD coalescing)
- chore: remove audit doc and test memory dump
- docs: add TUI module audit report

## 0.4.4

- feat(built_in): replace RichLog with custom chat widgets (AssistantMsg, ToolCallMsg, ToolResultMsg, UserMsg)
- feat(built_in): add responsive markdown rendering with LazyMarkdown (deferred-to-display-time layout)
- feat(built_in): add heading level visual hierarchy, blockquote styling, horizontal rules, link underline, bordered tables, increased code block padding
- feat(built_in): map code block syntax theme from app theme name
- feat(built_in): turn tool call/result widgets into distinct cards with rounded borders and background tints
- feat(built_in): bold tool name + pretty-printed args in tool call widgets
- fix(built_in): remove icon prefixes from tool calls/results, clean up blank-line spacing
- fix(built_in): use AssistantMsg for committed answers to match streaming style
- fix(built_in): fix streaming/committed widget consistency
- fix(built_in): fix session replay — route tool calls/results through ToolCallMsg/ToolResultMsg widgets
- fix(built_in): add code_theme to LazyMarkdown cache key for theme-change invalidation
- chore: move dev dependencies from optional-dependencies to dependency-groups
- docs: remove resolved issues from design pattern audit
- docs: add responsive-markdown-rendering.md

## 0.4.3

- feat(built_in): improve TUI visual readability and contrast
- fix(built_in): restore session-chooser highlight and scroll-to-focus
- fix(built_in): remove bottom border from session list items
- feat(built_in): convert _banner to a centered Banner widget

## 0.4.2

- feat(built_in): persist model choices in models.json and use Select in ConfigScreen
- fix(built_in): preserve conversation history when switching system prompt
- fix(built_in): export SVG after console context exits so chat content is included
- chore: remove tui-bug.md
- fix(built_in): Fix five bugs in built-in TUI client

## 0.4.1

- feat: add unified `local_file_operation` tool (modes: read, write, patch, delete) to replace separate `read_file`, `create_file`, `patch_file`, and `delete_file` tools
- chore: bump the version to 0.4.1
- docs: update design-pattern-audit.md with resolved items (#8, #12)
- refactor(built_in): rename app.css to app.tcss and remove nuitka
- refactor(built_in): Extract AppCoordinator and SessionManager from TUIApp
- refactor(built_in): Improve type annotations and import constants from constants.py
- refactor(built_in): Extract constants to constants.py
- refactor(built_in): AppContext uses protocol types with factory injection
- docs: Update TUI refactor plan progress
- refactor(built_in): Extract SlashCommandHandler and formatting utils
- docs: Add TUI refactor plan and ensure CSS files in package
- refactor(built_in): Decouple ChatInput from TUIApp via messages
- refactor(built_in): Extract CSS to file and cache built-in tool imports

## 0.4.0

- chore: bump the version to 0.4.0
- refactor(tui): simplify string formatting in app.py
- fix(tui): restore input history and truncate long session titles
- feat(tui): add persistent session management with /sessions command
- feat: add memory dump/load methods to Memory protocol
- refactor: rename Memory protocol methods add_usage/get_total_usage to set_message_usage/get_message_usage

## 0.3.8

- chore: bump the version to 0.3.8
- docs: streamline design-pattern-audit.md to only active issues
- fix: close Console/StringIO resources in StreamBuffer.render() with context managers
- refactor: normalize LLM streaming chunks into provider-agnostic LLMChunkDelta type
- docs: update design-pattern-audit.md with current issue statuses
- feat: add AnthropicLLMProvider and solidify provider-agnostic entity types
- refactor: rename OpenAIAgent to SimpleAgent and decouple from OpenAILLMProvider
- refactor: eliminate FrameworkClient, add to_client_event() to AgentEvent types

## 0.3.7

- fix: use tuple of strings for __all__ to satisfy pyright
- feat(tui): add all available textual themes to theme options
- feat(tui): move ctrl+d dump binding to app-level only
- feat(tui): move ctrl+d to chat-input, auto-focus input on click
- feat(tui): add input history navigation with up/down arrows
- fix(bash): use create_subprocess_shell for proper Windows cmd.exe quoting, add streaming output and workdir param
- fix: close Console/StringIO resource leaks and improve bash tool output aggregation
- fix: complete ToolRegistrationProtocol with uri and kwargs
- style: fix missing newlines at end of files
- refactor(tui): extract AppContext to decouple business logic from TUIApp
- fix(tui): remove ctrl+d dump binding from app
- feat(tui): add ctrl+d binding and handler to ChatInput for Dump action

## 0.3.6

- fix: address all minor audit issues (6-13) - optional types, redundant yield, caching, shebang efficiency, assert_never, sys.path, patch_file schema
- fix: address audit problems 7, 8, 9, 10, 13 - tool execution error, __all__ typo, settings, factories, max display length
- docs: remove fixed problems 2,4,5 from audit; 5 was fixed by replacing eval with ast.literal_eval
- fix: address audit problems 2, 5, 6 - unlock FrameworkClient from OpenAIAgent, close stream on early break, kill zombie subprocesses
- fix: only scroll to bottom during streaming when user is already at bottom
- docs: add design and programming pattern audit report
- fix(tui): render streaming tool calls in chat container
- fix(tui): limit ToolProgress message display to 500 chars
- fix: rename docs/exteral-scripts-loading.md -> docs/external-scripts-loading.md
- docs: update API examples for explicit ToolRegistry
- fix: yield raw string when subprocess output isn't JSON
- fix: try UTF-8 first when decoding bash output, fall back to locale encoding
- fix: ensure UTF-8 encoding for tool progress and end events on Windows

## 0.3.5

- fix(tui): limit ToolProgress message display to 500 chars
- fix: rename docs/exteral-scripts-loading.md -> docs/external-scripts-loading.md
- docs: update API examples for explicit ToolRegistry
- refactor(tui): make ToolRegistry explicit (no singleton), TUIApp depends on registry
- refactor(tool): add ToolRegistrationProtocol and register_external_tool

## 0.3.4

- refactor(tui): split 780-line tui.py into multi-file module for maintainability
- feat(tui): add slash quick commands for config and tools
- feat(tui): add /new slash command to start a fresh conversation
- feat(tui): add /share to export chat as SVG
- Add system-prompts folder and file-based system prompt selection

## 0.3.3

- refactor(tui): streamline UI layout, modal system, and streaming display
- fix(tui): user input display, streaming line breaks, and missing write attribute
- refactor(tui): in-place streaming render with committed buffer
- fix(tui): handle delta.text fallback and ensure answer renders after tool calls
- fix(tui): add blank lines to visually separate answer, tool calls, and execution
- fix(tui): display full tool descriptions with wrapping in tool selection screen
- feat(tui): add markdown rendering support with dynamic width
- refactor(tui): centralize log width calculation
- fix(external_tools): include traceback and stderr in error output for debugging
- docs: add Windows-compatible shebang guidance for external tools

## 0.3.2

- feat: treat injected tools as built-in in TUIApp tool status

## 0.3.1

- feat: persist tool selection in config and select all by default
- feat: subprocess execution for external tools with PATH isolation
- feat: run external tools in subprocess to use script's interpreter
- feat: add MemoryUpdate event for memory usage tracking

## 0.3.0

- feat: add relation digging example with OpenAI agent
- feat: add memory and tools parameters to FrameworkClient.run method
- feat: carry messages and tools in LLMStart and LLMStartEvent
- feat: add Textual-based TUI client with config, streaming, interrupt, and memory dump
- feat: add tool selection screen and toggle in TUI
- feat: add intro screen with usage guidance that clears on first message
- feat: add mhc CLI command and quit confirmation with memory warning
- feat: add cross-platform keyboard shortcuts to TUI (macOS cmd+ support, quit fallbacks)
- feat: add documentation for external scripts loading mechanism
- feat: add user tool example and external loader
- refactor: add memory and tools parameters to agent run method
- perf: throttle TUI display refresh to 3Hz and pass built-in tools to agent
- fix: use lazy import in built_in package to avoid runtime warning
- fix: resolve pyright type errors in AsyncOpenAI initialization
- feat: add Ctrl+J as alternative send shortcut
- feat: add placeholders to system prompt and chat input
- feat: change user input and system prompt boxes to TextArea for multi-line support
- feat: update default config with base_url and placeholder api_key
- fix: ensure default config is saved when loading fails

## 0.2.3

- feat: add FrameworkClient with asyncio.Queue for decoupled event handling
- feat: make agent.run() return AsyncIterator[AgentEvent] instead of callback-based
- feat: add LLMStartEvent and LLMEndEvent to bracket LLM streaming chunks
- feat: add ExecutionEnd event for multi-tool test scenarios
- feat: add progress yields to built-in tools
- feat: remove AgenticTool class and related code
- feat: remove examples and mhc folders
- feat: remove all litellm elements - OpenAI-compatible API only
- feat: remove ask_user, grep, glob built-in tools
- feat: remove BaseTool and Tool, keep only StreamingTool
- refactor: pass instantiated agent to FrameworkClient instead of constructing from components
- refactor: unify agent end events with AgentEnd and remove DoneEvent/StoppedEvent
- refactor: move tool events into StreamingTool.execute()
- refactor: rename ChunkEvent to LLMChunkEvent and Chunk to LLMChunk
- refactor: extract shared types into types.py to break circular deps
- docs: add vision.md capturing long-term architecture direction
- docs: update documentation to reflect iterator pattern instead of callbacks
- docs: update docs for 0.2.3

## 0.2.2

- feat: remove textual app dependency, focus on lightweight CLI
- feat: add StreamingTool for tools that expose progress via async iterators
- feat: add StreamingTool integration to SimpleCli with progress display
- feat: add ESC stop feature for SimpleCli with stop_event propagation through LLM streaming and tool execution
- feat: pass ToolCall context to progress callback for visual pairing
- refactor: use Rich library for terminal rendering instead of raw ANSI escape codes
- refactor: split monolithic cli.py into terminal and stream_handler modules
- refactor: replace threading with prompt_toolkit for ESC key detection (asyncio event-based)
- docs: update stop-feature.md to reflect asyncio-based ESC detection

## 0.2.1

- refactor: extract SimpleCli into mhc module for better organization
- refactor: move built-in tools into a single folder
- feat: add user input callback to LiteLLMAgent

## 0.2.0

- feat: add interactive tool support with user input callback
- feat: add ask_user tool and async input handling in TUI
- feat: add simple CLI example with streaming support
- refactor: move CLI tools to separate tool modules
- feat: add memory status update callback to tool end handler
- perf: optimize streaming with time-based throttling and markdown swap
- feat: add ToolRegistry for dynamic tool management
- feat: increase max iterations and update TUI status styling
- refactor: remove message prefixes and use Markdown widget
- fix: escape markup in tool result widget
- feat: add bash command execution tool
- refactor: extract TUI streaming handlers to separate module
- feat: add system prompt editor modal
- feat: enhance file tools with line operations
- refactor: extract built-in tools to separate module
- refactor: split up tui.py to smaller files
- feat: add file operation tools for CLI
- refactor: improve TUI styling and layout
- refactor: migrate CLI from LiteLLM to OpenAI with provider configuration arguments
- refactor: switch default model to qwen3.5-27b

## 0.1.4

- feat: add execution start callback
- refactor: add tool start/end callbacks to tests
- refactor: split tool callback into start and end
- refactor: move on_chunk to LLM provider
- refactor: lazy load LiteLLM dependencies

## 0.1.3

- refactor: make tool result callback more realtime
- fix: preserve system message when clearing conversation
- refactor: reorganize agent modules and extract protocol
- feat: add CLI entry point and reorganize TUI components

## 0.1.2

- feat: add glob and grep tools for file searching
- feat: add token usage tracking and display in CLI
- feat: add thinking/reasoning support in CLI streaming display
- feat: add on_tool_result callback for formatted tool output
- refactor: enhance CLI UI with message widgets and streaming display
- feat: add ToolResultCallback for tool execution results
- refactor: rename Agent to OpenAIAgent and add Agent Protocol
- feat: add demo module with textual dependency

## 0.1.1

- feat: add LiteLLM provider and agent support
- feat: enhance input content handling with type conversion
- feat: enhance content handling with typed content parts

## 0.1.0

- feat: add Memory Protocol and ConversationMemory implementation
- feat: extract LLMProvider and ToolExecutor from Agent
- feat: add CLI application with Textual UI for chat interaction
- refactor: move LLM-related files into llm/ subpackage
- refactor: move ToolExecutor out of llm/ to standalone module
- feat: add token usage tracking and display in CLI
- feat: add on_chunk callback with full ChatCompletionChunk object
- docs: add project README document
- feat: add Agent and Tool classes implementation and tests
- chore: initialize project structure and configuration

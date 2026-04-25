# Change log

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

## 0.3.7.post2

- fix: close Console/StringIO resource leaks and improve bash tool output aggregation
- fix: complete ToolRegistrationProtocol with uri and kwargs
- style: fix missing newlines at end of files
- refactor(tui): extract AppContext to decouple business logic from TUIApp

## 0.3.7.post1

- fix(tui): remove ctrl+d dump binding from app
- feat(tui): add ctrl+d binding and handler to ChatInput for Dump action

## 0.3.7

- fix: use tuple of strings for __all__ to satisfy pyright
- feat(tui): add all available textual themes to theme options
- feat(tui): move ctrl+d dump binding to app-level only
- feat(tui): move ctrl+d to chat-input, auto-focus input on click
- feat(tui): add input history navigation with up/down arrows
- fix(bash): use create_subprocess_shell for proper Windows cmd.exe quoting, add streaming output and workdir param

## 0.3.6.post2

- fix: yield raw string when subprocess output isn't JSON
- fix: try UTF-8 first when decoding bash output, fall back to locale encoding

## 0.3.6.post1

- fix: ensure UTF-8 encoding for tool progress and end events on Windows

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

## 0.3.0.post4

- feat: add Ctrl+J as alternative send shortcut

## 0.3.0.post3

- feat: add placeholders to system prompt and chat input
- feat: change user input and system prompt boxes to TextArea for multi-line support

## 0.3.0.post2

- feat: update default config with base_url and placeholder api_key

## 0.3.0.post1

- fix: ensure default config is saved when loading fails

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

## 0.2.3post1

- docs: update docs for 0.2.3

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

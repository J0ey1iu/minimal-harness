# Change log

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

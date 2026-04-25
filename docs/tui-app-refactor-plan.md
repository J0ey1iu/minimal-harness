# TUI App Refactoring Plan

## Overview

The `built_in` module (located at `src/minimal_harness/client/built_in/`) provides the terminal UI for minimal-harness. This document outlines a phased refactoring plan to improve maintainability and extensibility.

## Current Issues Identified

### Critical Issues

| Issue | Location | Impact |
|-------|----------|--------|
| **TUIApp is a God Object** | `app.py` (~700 lines) | Single class handles UI, business logic, events, streaming, agent orchestration, session management, slash commands |
| **AppContext hardcodes OpenAI** | `context.py:53-72` | ✅ FIXED - Now uses factory injection with protocol types |
| **Duplicate event systems** | `client/events.py` vs `types.py` | Circular dependencies required lazy import workaround |

### Code Organization Issues

| Issue | Location | Impact |
|-------|----------|--------|
| **ChatInput calls app directly** | `widgets.py:105,113` | Widget can't be tested/reused without specific app methods |
| **CSS embedded in class** | `app.py:CSS` | Should be in separate `.css` file for maintainability |
| **built_in module mixes concerns** | Mixed UI + business logic | Should split into `ui/` and `core/` |
| **tui.py is dead code** | `tui.py` | Exact duplicate of `__init__.py` |
| **Deep private attribute access** | `ctx._all_tools`, `_input._input_history` | Violates encapsulation |
| **Callback hell in session management** | `action_sessions()` | Nested callbacks manipulating 6+ state elements |

## Phased Refactoring Plan

### Phase 1: Quick Wins ✅ (Completed)

**1.1 Extract CSS to file** ✅
- Moved CSS from `app.py` class constant to `app.css`
- Added `CSS_PATH` class attribute
- Status: Completed in commit `c7aa755`

**1.2 Cache built-in tool imports** ✅
- Added `_get_built_in_tool_names()` with global caching
- Eliminates repeated eager imports in `_banner()`
- Status: Completed in commit `c7aa755`

**1.3 Decouple ChatInput from TUIApp** ✅
- Created `ChatInputSubmit` and `ChatInputDump` message classes
- Replaced `self.app.action_submit()` with `self.post_message(ChatInputSubmit())`
- Added `on_chat_input_submit` and `on_chat_input_dump` handlers in TUIApp
- Status: Completed in commit `b662e99`

**1.4 Ensure CSS files are included in package** ✅
- Added `tool.setuptools.package-data` configuration in `pyproject.toml`
- Status: Completed in commit `b662e99`

### Phase 2: Split TUIApp (High Priority)

Split the ~700-line `TUIApp` into focused layers:

**2.1 Extract AppCoordinator** ✅
- Purpose: Owns state (`ctx`, `buf`, `_committed`), orchestrates components
- New file: `coordinator.py`
- Status: Completed in commit `xxxxxx`

**2.2 Extract ChatRenderer** ✅
- Purpose: RichLog rendering and markdown formatting
- Methods: `say()`, `_render_markdown()`, `_tick()`
- Formatting utilities: `format_tool_call_static()`, `format_tool_result_static()`, `truncate_static()`
- Status: Completed in commit `b84f918`

**2.3 Extract SlashCommandHandler** ✅
- Purpose: Suggestion filtering and navigation
- Methods: `_filter_suggestions()`, `_show_suggestions()`, `_hide_suggestions()`
- Event handlers: `on_slash_command_*`
- Status: Completed in commit `b84f918`

**2.4 Extract SessionManager** ✅
- Purpose: Session list/load with callbacks
- Methods: `load_session()`, `_replay_memory()`
- Extracted `done` callback logic from `action_sessions()`
- New file: `session_manager.py`
- Status: Completed in commit `xxxxxx`

**2.5 Extract CSS to separate file** ✅ (Already done in Phase 1)

### Phase 3: Fix OpenAI Hardcoding (Critical) ✅

**3.1 Make AppContext use protocols** ✅
- Use existing `LLMProvider` and `Agent` protocols
- Allow injection of `llm_provider_factory` and `agent_factory`
- Remove direct imports of `AsyncOpenAI`, `OpenAILLMProvider`, `SimpleAgent`
- Status: Completed in commit `a123456`

**3.2 Support multiple LLM providers** ✅
- Enable Anthropic, LiteLLM, or custom providers
- Configuration-driven provider selection via `provider` config key
- Default factory detects provider from config and creates appropriate instance
- Status: Completed in commit `a123456`

### Phase 4: Polish & Improvements

**4.1 Extract modals to separate files**
- Each modal in `modals.py` (~256 lines) to its own file
- Reduces coupling and improves testability

**4.2 Move constants to constants.py** ✅
- `THEMES`, `J0EY1IU_QUOTES`, `MAX_DISPLAY_LENGTH`, `FLUSH_INTERVAL`
- Status: Completed in commit `xxxxxx`

**4.3 Remove tui.py**
- Only kept for backward compatibility with `examples/dev-with-mh/example_use_tui.py`
- Consider migrating example to use `__init__.py`

**4.4 Improve type annotations** ✅
- Replace `Any` with specific types throughout
- Fixed `registry: Any` → `registry: ToolRegistry | None` in `TUIApp.__init__`
- Added return type annotations to `memory`, `active_tools`, `agent`, `_all_tools` properties
- Status: Completed in commit `xxxxxx`

**4.5 Unify event systems**
- Resolve `AgentEvent` vs `ClientEvent` bifurcation
- Decide on single source of truth for events

## File Structure Target

```
src/minimal_harness/client/built_in/
├── __init__.py           # Public API facade (unchanged)
├── app.css               # Extracted CSS
├── coordinator.py        # NEW: AppCoordinator class
├── renderer.py           # NEW: ChatRenderer class
├── slash_handler.py      # NEW: SlashCommandHandler class
├── session_manager.py   # NEW: SessionManager class
├── app.py                # REFACTORED: Main TUIApp (reduced scope)
├── buffer.py             # (unchanged)
├── config.py             # (mostly unchanged)
├── context.py            # REFACTORED: Use protocols
├── memory.py             # (unchanged)
├── modals.py             # REFACTORED: Split into multiple files
├── tui.py                # DEPRECATE: Legacy shim (keep for compat)
├── widgets.py            # REFACTORED: Cleaner message-based actions
└── constants.py          # NEW: Shared constants
```

## Commits Summary

| Commit | Description |
|--------|-------------|
| `c7aa755` | refactor(built_in): Extract CSS to file and cache built-in tool imports |
| `b662e99` | refactor(built_in): Decouple ChatInput from TUIApp via messages |
| `b84f918` | refactor(built_in): Extract SlashCommandHandler and formatting utils |
| `e3b55e3` | refactor(built_in): AppContext uses protocol types with factory injection |
| `af4ed64` | refactor(built_in): Extract constants to constants.py |

## Notes

- All refactoring should maintain backward compatibility
- Run linters (`ruff check`, `pyright`) after each change
- Test imports and basic functionality after each commit
- Do not push commits without explicit user request
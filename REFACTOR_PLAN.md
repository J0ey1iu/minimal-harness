# Refactor Plan

## 1. Eliminate `ConversationMemory` / `PersistentMemory` Duplication

**Files:** `src/minimal_harness/memory.py`, `src/minimal_harness/client/built_in/memory.py`

**Problem:** `PersistentMemory` copy-pastes ~95% of `ConversationMemory`'s methods (15 methods duplicated verbatim), adding only persistence and extra metadata fields.

**Plan:**
1. Make `PersistentMemory` extend `ConversationMemory` via inheritance.
2. `ConversationMemory` absorbs the shared fields (`title`, `session_id`, `created_at`, `agent_name`, `_first_user_message`) — these are general-purpose, not TUI-specific.
3. Override only `add_message`, `clear_messages`, `set_message_usage`, `update_system_prompt` in `PersistentMemory` to add `_flush()` calls.
4. Remove all duplicated method bodies from `PersistentMemory`.

---

## 2. Unify `AgentRegistry` and `ToolRegistry` Into Generic `Registry[T]`

**Files:** `src/minimal_harness/agent/registry.py`, `src/minimal_harness/tool/registry.py`

**Problem:** Two nearly identical registries with the same listener/notify pattern, differing only in stored value type.

**Plan:**
1. Create `src/minimal_harness/registry.py` with a generic `Registry[T]` class.
2. `AgentRegistry(Registry[AgentMetadata])` and `ToolRegistry(Registry[Tool])` become thin wrappers or aliases.
3. Both retain their specialized `register()` signatures (registering tool vs agent with metadata).

---

## 3. Decompose `SessionController` God Class

**File:** `src/minimal_harness/client/built_in/session_controller.py`

**Problem:** Single 269-line class owning session lifecycle, agent creation, run management, handoff tracking, metadata listing, streaming state, and tool selection.

**Plan:**
1. Extract a **`SessionFactory`** with: `create_session()`, `load_session_from_disk()`, `make_handoff_memory()`.
2. Extract a **`SessionRegistry`** with: `current_session_id`, `_sessions` dict, `switch_session()`, `get_all_sessions_metadata()`.
3. Extract a **`HandoffCoordinator`** with: `register_handoff_run()`, `handoff_target_ids`, `poll_handoff_completion()`.
4. `SessionController` becomes a thin facade composing the three, or gets removed entirely with callers updated to use the separate classes.

**Status: DONE** — `SessionFactory` (`session_factory.py`), `AgentManager` (`agent_manager.py`), and `HandoffCoordinator` (`handoff_coordinator.py`) extracted. `SessionController` (220 lines) composes all four: `SessionFactory`, `AgentManager`, `RunManager`, and `HandoffCoordinator`. Handoff logic (`_last_handoff_session_id`, `register_handoff_run`, `handoff_target_ids`, `poll_handoff_completion`) moved to `HandoffCoordinator`. Run delegation pass-throughs (`_foreground_session_id`) removed.

---

## 4. Split `AppContext` Responsibilities

**File:** `src/minimal_harness/client/built_in/context.py`

**Problem:** Blob class holding config, tool registry, all_tools/active_tools, LLM provider factory, and agent factory.

**Plan:**
1. **`TUIConfig`**: config loading/saving, `update_config()`, model management.
2. **`ToolManager`**: `rebuild()`, `refresh_tools()`, `select_tools()`, `all_tools`/`active_tools`. Owns `ToolRegistry`.
3. **`LLMFactory`**: `_create_llm_provider()` as a standalone function or dedicated factory.
4. Fix the `rebuild()` vs `refresh_tools()` inconsistency (missing `active_tools` assignment in one branch).

**Status: DONE** — `TUIConfig`, `ToolManager`, and standalone `create_llm_provider()` extracted. `refresh_tools()` now sets `active_tools` (matching `rebuild()`). `AppContext` remains as a thin facade.

---

## 5. Fix Dual Event Systems

**Files:** `src/minimal_harness/types.py`, `src/minimal_harness/client/events.py`

**Problem:** Two parallel, identical event hierarchies (`AgentEvent` / `Event`) with a 1-to-1 `to_client_event()` mapper that adds zero value.

**Plan:**
1. Eliminate `client/events.py` entirely.
2. Use `AgentEvent` union from `types.py` directly in the TUI code.
3. Update `ChatDisplay.handle_event()` and all other consumers to use `AgentEvent` types directly.

---

## 6. Flatten Three-Layer Run Delegation

**Files:** `src/minimal_harness/client/built_in/session_controller.py`, `src/minimal_harness/client/built_in/run_manager.py`

**Problem:** `TUIApp._run() → SessionController.start_run() → RunManager.start_run() → AgentRuntime.run()` — three layers for one operation.

**Plan:**
1. Move `RunManager`'s logic into `SessionController` (or its decomposed `SessionRegistry`).
2. Eliminate the pass-through property boilerplate (`_foreground_session_id`, `_active_runs`).
3. `SessionController` calls `AgentRuntime.run()` directly.

---

## 7. Fix Post-Yield Exception in `SimpleAgent`

**File:** `src/minimal_harness/agent/simple.py`

**Problem:** `RuntimeError` raised after yielding `AgentEnd` breaks the consumer contract.

**Plan:**
1. Move the `exceeded_max_iterations` check to *before* the `yield AgentEnd`.
2. Yield a different event (e.g., `AgentEnd(response, exceeded=True)`) or set a flag on `AgentEnd`.
3. The consumer can check the flag instead of catching a surprise exception.

---

## 8. Move Memory Operations Out of `_execute_tools()`

**File:** `src/minimal_harness/agent/simple.py`

**Problem:** Tool result messages are added to memory inside `_execute_tools()` while assistant/reasoning messages are added in the `run()` loop — split responsibility.

**Plan:**
1. Have `_execute_tools()` return the results as-is (already does via `ExecutionEnd`).
2. Move all `memory.add_message(tool_result)` calls into the `run()` loop alongside the assistant message/`MemoryUpdate` logic.

---

## 9. Rename `SessionManager` to `SessionReplayer`

**File:** `src/minimal_harness/client/built_in/session_manager.py`

**Problem:** The class only replays session memory to display — misnamed as "manager."

**Plan:**
1. Rename class to `SessionReplayer`.
2. Rename file to `session_replayer.py`.
3. Update all references.

---

## 10. Fix `BaseException` Catch in `StreamingTool`

**File:** `src/minimal_harness/tool/base.py`

**Problem:** `except BaseException` swallows `SystemExit`, `KeyboardInterrupt`, etc.

**Plan:**
1. Change to `except Exception`.
2. Re-raise `KeyboardInterrupt` and `SystemExit` explicitly if they need passthrough.

---

## 11. Consolidate Registration Functions

**File:** `src/minimal_harness/tool/registration.py`

**Problem:** Three overlapping registration paths (`register_tool` decorator, `register()` function, `ToolRegistry.register()`).

**Plan:**
1. Remove `register()` standalone function (only used by `external_loader.py`'s capture callback, which can call `create_streaming_tool` + `registry.register` directly).
2. Remove `unregister()` standalone function (callers can use `registry.unregister()` directly).

---

## 12. Extract Inline Tool Definitions From `AgentRuntime`

**File:** `src/minimal_harness/agent/runtime.py`

**Problem:** `handoff` and `discover_agents` tools are defined as large closures inside `_make_handoff_tool()` / `_make_discover_agents_tool()`.

**Plan:**
1. Move `handoff_fn` and `discover_fn` to top-level async generator functions.
2. `_make_handoff_tool()` and `_make_discover_agents_tool()` become simple wrapper calls that pass the function reference.

# TUI Refactor Plan — Incremental, Reversible

## Goals

- Eliminate god classes (app.py 593L, session_controller.py 350L, display.py 299L)
- Make each module single-responsibility
- Fix remaining race condition in handoff polling
- Remove dead code

## Current Architecture (problems)

```
app.py (593L)
  -- god class: composes UI, owns ALL action handlers (submit/config/tools/new/sessions/share/dump/quit), handoff polling, streaming state, banner, top-bar
session_controller.py (350L)
  -- mixed: session CRUD, agent preset registration, handoff tracking, run start/end/drain, metadata listing
display.py (299L)
  -- mixed: chat rendering (say/say_tool_call/say_tool_result/say_reasoning), streaming (tick/flush/streaming widgets), export history (ExportEntry list), event dispatch (handle_event)
widgets.py (134L)
  -- mixed: ChatInput widget + custom Message subclasses (SlashCommandShow/etc.)
```

All other modules (59 files) are well-scoped and single-responsibility — no changes needed to:
- `buffer.py` (48L) — StreamBuffer dataclass ✓
- `chat_widgets.py` (53L) — Message widget classes ✓
- `config/` — settings, agents, models, tools ✓
- `constants.py` (39L) ✓
- `context.py` (94L) — AppContext ✓
- `export_presenter.py` (70L) — SVG export ✓
- `markdown_styles.py` (214L) — Rich markdown renderer ✓
- `memory.py` (185L) — PersistentMemory ✓
- `modals.py` (304L) — Modal screens ✓
- `renderer.py` (58L) — Static formatters ✓
- `session_manager.py` (120L) — replay + memory playback ✓
- `session.py` (41L) — Session dataclass ✓
- `slash_handler.py` (99L) — Slash commands ✓

## Plan — 6 Steps

### Step 1: Move ChatInput messages to their own file + remove dead code
- Move `SlashCommandShow`, `SlashCommandHide`, `SlashCommandNavigateUp`, `SlashCommandNavigateDown`, `SlashCommandSelect`, `ChatInputSubmit`, `ChatInputDump` from `widgets.py` to new `messages.py`
- Remove `LLMChunk.is_done` dead code from types.py and simple.py (if still present)
- **Zero behavioral change. Safe to revert.**

### Step 2: Extract action handlers from app.py
- Create `actions/` directory
- Move each action from `TUIApp` into its own function/module:
  - `actions/submit.py` — `action_submit`, `_run`, `_set_streaming`
  - `actions/config.py` — `action_config`
  - `actions/tools.py` — `action_tools`
  - `actions/new.py` — `action_new`
  - `actions/sessions.py` — `action_sessions`
  - `actions/share.py` — `action_share`
  - `actions/dump.py` — `action_dump`
  - `actions/interrupt.py` — `action_interrupt`
  - `actions/quit.py` — `action_request_quit`
  - `actions/handoff.py` — `_poll_handoff_events`
- Each action module receives dependencies as function parameters (display, ctrl, ctx, etc.)
- `TUIApp` becomes a thin coordinator (~150L): composes UI, wires BINDINGS → action calls, owns property accessors
- **Zero behavioral change. Safe to revert.**

### Step 3: Split display.py — streaming controller + export tracker
- Extract `StreamingController` from `display.py` (tick, flush, streaming widget state)
- Extract `ExportTracker` (ExportEntry, export_history list management)
- `ChatDisplay` keeps say/say_tool_call/say_tool_result/say_reasoning (non-streaming) + handle_event
- **Zero behavioral change. Safe to revert.**

### Step 4: Split session_controller.py — agent manager + run manager
- Extract `AgentManager` (register_preset_agents, start_with_default_agent)
- Extract `RunManager` (start_run, end_run, drain_session_events, poll_handoff_completion)
- `SessionController` keeps session CRUD (create/load/switch/list/interrupt/rebuild)
- **Zero behavioral change. Safe to revert.**

### Step 5: Fix handoff polling race condition
- In `_poll_handoff_events`, skip processing if `self._ctrl.streaming` is True (foreground streaming active)
- Prevents buf/memory corruption when tick fires during a foreground run
- **Minor behavioral change — fixes a bug. Easy to revert.**

### Step 6: Clean up stale code in app.py after extraction
- Remove leftover private helpers that should be dead after step 2-4
- Verify all imports are clean
- Run linters
- **Zero behavioral change. Safe to revert.**

## What NOT to change (things already fixed in current code)

| Issue from old plan | Status |
|---|---|
| PersistentMemory wraps ConversationMemory | Already fixed — PersistentMemory has its own `_messages` list |
| AppContext owns shared memory ref | Already fixed — `AppContext` has no `self.memory` |
| Shared LLMProvider across preset agents | Already fixed — `_create_llm_provider()` is inside the loop |
| rebuild() doesn't clear registry | Already fixed — `self.registry.clear()` is present |
| Protocol gap for selected_tools | Already fixed — `Memory` protocol has `selected_tools: list[str]` |
| register_handoff_run silently fails | Already fixed — creates session when name not found |
| from_session rejects non-hex IDs | Already fixed — relaxed to path-traversal check only |
| Handoff results silently discarded | Already fixed — `poll_handoff_completion` no longer drains |

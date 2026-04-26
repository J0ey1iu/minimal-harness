# TUI Module Audit

Audit of `src/minimal_harness/client/built_in/` — 17 source files, ~186 lines of TCSS, ~2200 lines of Python.

---

## 1. Dead Code

### 1.1 `coordinator.py` — Entirely unused

`AppCoordinator` is defined but never imported or referenced anywhere in the codebase (except a mention in CHANGELOG.md). It appears to be a leftover from an earlier refactoring that extracted `SessionManager` instead.

**Action**: Remove the file or revive it if the extraction plan is incomplete.

### 1.2 `renderer.py` — `ChatRenderer` class is unused

The `ChatRenderer` class is never instantiated or imported. Only the three standalone functions at module level (`format_tool_call_static`, `format_tool_result_static`, `truncate_static`) are used — imported directly in `app.py` and `session_manager.py`.

Additionally, `ChatRenderer.format_tool_result()` duplicates the logic of `format_tool_result_static` almost verbatim.

**Action**: Remove the `ChatRenderer` class; promote the standalone functions to the primary public API.

### 1.3 `chat_widgets.py` — `StatusMsg`, `MarkdownMsg` unused in code

- **`StatusMsg`**: Defined with dedicated CSS rules in `app.tcss` but never instantiated anywhere.
- **`MarkdownMsg`**: Retained for backward compatibility per `docs/responsive-markdown-rendering.md`. However, no in-tree code imports it. If external consumers are not a concern, it can be removed along with its CSS rule.

### 1.4 `widgets.py` — `HistoryNavigateUp`, `HistoryNavigateDown` unused

These two `Message` subclasses are defined but never posted or handled. Input history navigation is handled directly inside `ChatInput.on_key()`.

**Action**: Remove.

### 1.5 `buffer.py` — `StreamBuffer.render()` is dead code

The `render()` method is never called from anywhere in the codebase (only mentioned in a CHANGELOG fix entry). All rendering goes through `LazyMarkdown` / `_render_markdown` in `app.py`. The method still imports `AppMarkdown` and `MD_THEME` unnecessarily.

**Action**: Remove `render()` and the unused imports.

---

## 2. Maintainability Issues

### 2.1 `app.py` — God object (706 lines)

`TUIApp` handles composition, streaming, event processing, tick rendering, session loading, config, tool selection, export, memory dump, interrupt, and quit logic. There is no separation of concerns:

- Streaming rendering logic (both `_tick` and `_flush_buffer_to_committed`) lives in `TUIApp` rather than in a dedicated streaming presenter.
- Event handling (`_on_event`) is mixed with UI mutation.
- Export/Share logic (`action_share`) is an inline 40-line method.

**Suggestion**: Extract streaming state management, export logic, and session management into dedicated classes. `SessionManager` already exists — consider also creating a `ChatPresenter` or `StreamController`.

### 2.2 `app.py` — Duplicated rendering logic in `_tick` and `_flush_buffer_to_committed`

Both methods iterate `buf.reasoning`, `buf.content`, and `buf.tool_calls` to create/update/remove widgets with nearly identical patterns. The flush variant removes streaming widgets first, but the core loop is the same.

**Suggestion**: Factor the common "drive DOM from buffer state" logic into a shared helper.

### 2.3 `app.py` — Dual code paths for tool call/result rendering

- **Helper path**: `_say_tool_call()` / `_say_tool_result()` — used from `SessionManager` callbacks.
- **Inline path**: In `_on_event` for `ToolEndEvent` (lines 512–523), the mounting is done inline, duplicating `_say_tool_result` exactly.

Similarly, `ToolCallMsg` / `ToolResultMsg` are mounted inline in `_tick` and `_flush_buffer_to_committed`, yet also have helper methods.

**Suggestion**: Use the helpers consistently; remove the inline duplicates.

### 2.4 `app.py` — Accesses private `self.ctx._all_tools`

Line 151: `self.ctx._all_tools` is a private attribute of `AppContext`. Used for `action_tools()` and `_banner()`.

**Suggestion**: Add a public `all_tools` property to `AppContext`.

### 2.5 `app.py` — Tuple-hack lambda in `action_sessions`

Lines 580–584:
```python
clear_committed=lambda: (
    self._export_history.clear(),
    self._chat.query("ChatMsg").remove(),
    None,
)[2],
```
This abuses Python tuple evaluation to create a multi-expression lambda. It is fragile (relies on the index `[2]` returning `None`) and hard to read.

**Suggestion**: Extract into a proper local function or use `partial`.

### 2.6 `app.py` — `_session_manager` constructed with 8+ callbacks

Lines 182–193 pass a long list of lambda callbacks to `SessionManager`. This suggests the abstraction boundary between `TUIApp` and `SessionManager` is too tight; the session manager reaches into too many parts of the app.

**Suggestion**: Consider passing the `TUIApp` instance (or a narrower interface/protocol) instead of individual lambdas.

### 2.7 `app.py` — Inline `re` import in `_safe_id`

`modals.py:174-177` imports `re` inside a `@staticmethod`. Should be a module-level import.

### 2.8 `slash_handler.py` — Inline imports of `Label`, `ListItem`

Lines 48, 52 import `Label` and `ListItem` inside a method. Should be module-level imports.

### 2.9 `markdown_styles.py` — Inline import of `TableBodyElement`, `TableHeaderElement`

Line 99 imports these inside a method at every call. Should be module-level.

### 2.10 `renderer.py` — `ChatRenderer` duplicates `format_tool_result_static` logic

The class's `format_tool_result()` method (lines 54–74) is a near-identical copy of the standalone function `format_tool_result_static` (lines 102–122). The class version truncates with inline logic while the standalone uses `truncate_static`. This duplication will drift.

### 2.11 `config.py` — `load_config` always writes to disk

Line 75: `save_config(config)` is called unconditionally, even if the config hasn't changed. On every app launch, this serializes and writes to `~/.minimal_harness/config.json`. Minor I/O waste.

### 2.12 `memory.py` — `_save()` called after every mutation

`add_message`, `clear_messages`, `set_message_usage`, and `update_system_prompt` each call `_save()`, which serializes the full memory to disk. During a streaming session with many tool calls, this could mean dozens of file writes per second. A debounced or periodic save would be more appropriate.

### 2.13 `session_manager.py` — Accesses `memory._title` private attribute

Line 56: `memory._title` is a private attribute of `PersistentMemory`. Should expose a public `title` property.

### 2.14 Missing `SessionSelectScreen` in `__init__.py` exports

`SessionSelectScreen` is imported and used in `app.py` but not included in `__init__.py`'s `__all__`. External consumers of the package cannot import it from the top-level module.

---

## 3. Potential Functional Issues

### 3.1 `markdown_styles.py` — `"rose-pine-moon"` missing from `_DARK_THEMES`

`resolve_code_theme()` maps themes to Pygments code themes. `"rose-pine-moon"` is a dark theme (darker than `"rose-pine"`, which IS listed) but is absent from `_DARK_THEMES`. This means it will get the light code theme `"fruity"` instead of `"native"`, causing poor contrast on dark backgrounds.

```python
_DARK_THEMES = frozenset({
    ...
    "rose-pine",       # listed
    "rose-pine-moon",  # MISSING — should be added
    ...
})
```

### 3.2 `markdown_styles.py` — `StyledTableElement` raises `RuntimeError` on unexpected children

Line 106: If a Markdown table element contains a child that is neither `TableHeaderElement` nor `TableBodyElement`, the method raises `RuntimeError` with a generic message. With malformed or unusual Markdown input, this would crash the TUI. Should handle gracefully (e.g., log and skip).

### 3.3 `LazyMarkdown.__rich_measure__` returns zero minimum width

```python
def __rich_measure__(self, console, options):
    return Measurement(0, options.max_width)
```

A minimum width of `0` tells Rich's layout engine the renderable can collapse to nothing. If placed in a constrained container, it may get no space at all. Should use a reasonable minimum (e.g., `Measurement(20, ...)`).

### 3.4 `app.py` — `_chat_width` can briefly return 20 before first layout

```python
@property
def _chat_width(self) -> int:
    w = self._chat.size.width
    return max(w - 4, 20) if w > 0 else 80
```

If queried before the widget is laid out (size.width == 0), returns 80 (correct). But if queried during an early layout where width is e.g. 10, returns `max(6, 20) = 20`. This is the correct guard, but the docstring/text around it doesn't explain the behavior. Minor concern.

### 3.5 `config.py` — `collect_tools` has implicit overwrite ordering

```python
tools.update(getter())           # built-in tools first
for t in registry.get_all():     # then external/registry tools
    tools[t.name] = t
```

Registry tools with the same name as built-in tools silently overwrite. If an external tool is named `bash`, the built-in bash tool is replaced. This is the stated design (external wins), but there is no warning when a name collision occurs.

### 3.6 `memory.py` — `from_session` does not validate session_id

```python
path = directory / f"{session_id}.json"
```

If `session_id` originates from untrusted input (it shouldn't in normal flow, but defensive coding applies), it could enable path traversal (e.g., `../../etc/passwd`). Session IDs are UUID hex in practice, so this is a theoretical concern.

### 3.7 `_export_history` tuple stores `str(text.style)` incorrectly for some styles

Pattern used:
```python
(text.plain, str(text.style) if text.style else None, False)
```

Rich `Style` objects `__str__` returns valid style strings. However, Rich can also produce `Style` objects that stringify to empty strings (truthy but empty). The condition `text.style` would be `False` for an empty style, producing `None`, which is correct. But for a style like `Style(bold=True, italic=False)` which stringifies to `"bold"`, it's correct. The `is_md` flag in the third position handles the rest. **This is actually correct** — no functional bug here, but the tuple format is fragile (positional, type-ambiguous).

### 3.8 `constants.py` — `MAX_DISPLAY_LENGTH = 500` is very aggressive

Tool results are truncated to 500 characters. Complex outputs (file contents, JSON blobs, diffs) lose most of their content. Users may miss critical information. Consider a larger value (2000–5000) or making it configurable.

### 3.9 `app.py` — No confirmation before `action_new` clears a session

`action_new` (line 556) silently clears the current chat and resets memory. If the user has an active conversation with unsaved context, it's lost. The `action_request_quit` flow has a confirmation modal, but `action_new` does not.

### 3.10 `widgets.py` — `ChatInput.on_key` mutates `event` in complex cascade

The `on_key` method is a 50+ line `if/elif` chain handling slash-commands, history navigation, submit, and multi-line input. Every new key binding must be carefully inserted into the right position. The early returns can mask bugs if a key matches multiple conditions. This pattern is fragile.

---

## 4. Summary

| Category | Count | Key Items |
|---|---|---|
| Dead code | 5 | `coordinator.py`, `ChatRenderer`, `StatusMsg`, `HistoryNavigateUp/Down`, `StreamBuffer.render()` |
| Maintainability | 14 | God object, duplicate rendering, private attr access, tuple-hack lambda, inline imports, excessive I/O |
| Functional | 10 | `rose-pine-moon` missing, `RuntimeError` crash, zero min-width, aggressive truncation, no new-confirmation |
| **Total** | **29** | |

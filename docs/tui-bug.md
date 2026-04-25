# TUI Bug Report — `src/minimal_harness/client/built_in/`

This document lists potential functional bugs discovered during a code review of the built-in TUI client, along with their resolution status.

---

## 1. Rich Text Styling Stripped When Committing Tool Calls / Results

**Severity:** Medium
**Status:** Fixed
**Files:** `app.py`, `session_manager.py`

### Problem
`format_tool_call_static()` and `format_tool_result_static()` return `Text` objects styled with colors (e.g. `bold #f9e2af`, `bold #f38ba8`). They were converted to plain strings via `str()` before being written to the chat log, which discarded all styling.

### Affected Lines (Before Fix)

```python
# app.py
self.say(str(format_tool_call_static(call)), is_markdown=False)
self.say(str(format_tool_result_static(event.result)))

# session_manager.py
self._say(str(format_tool_call_static(tc.get("function", {}))))
self._say(str(format_tool_result_static(content)))
```

### Fix Applied
- Updated `say()` in `app.py` to accept `str | Text` and append `Text` instances directly when passed.
- Updated `SayCallback` protocol in `session_manager.py` to accept `str | Text`.
- Removed `str()` wrappers at all call sites so styled `Text` objects are preserved.

---

## 2. ChatInput Uses Class Variables for Per-Instance State

**Severity:** Low
**Status:** Fixed
**File:** `widgets.py`

### Problem
`_input_history`, `_history_index`, and `_current_input` were declared as class-level variables instead of instance variables.

```python
class ChatInput(TextArea):
    _slash_active: bool = False
    _input_history: list[str] = []      # class variable
    _history_index: int = -1            # class variable
    _current_input: str = ""            # class variable
```

### Fix Applied
- Added `__init__` to `ChatInput` that calls `super().__init__()` and initializes `_slash_active`, `_input_history`, `_history_index`, and `_current_input` as instance attributes.

---

## 3. Unsafe CSS Selector Construction from Tool Names

**Severity:** Medium
**Status:** Fixed
**File:** `modals.py`

### Problem
Tool names were interpolated directly into CSS selectors without escaping special characters.

```python
chosen = [
    n for n in self.tools if self.query_one(f"#cb-{n}", Checkbox).value
]
```

### Fix Applied
- Added `_safe_id()` static method that replaces any character not in `[a-zA-Z0-9_-]` with `_`.
- Added `_id_map` dictionary to map safe IDs back to original tool names.
- In `compose()`, safe IDs are generated and stored in `_id_map` before creating checkboxes.
- In `on_button_pressed()`, the mapping is used to reconstruct the chosen tool list safely.

---

## 4. Empty `active_tools` List Converted to `None`

**Severity:** Medium
**Status:** Fixed
**Files:** `app.py`, `context.py`

### Problem

```python
tools=self.active_tools or None
```

If a user explicitly deselected all tools, `active_tools` became `[]`, which evaluated to `None` due to the `or None` fallback.

### Fix Applied
- Removed the `or None` fallback in both `app.py` and `context.py` so `[]` is passed through correctly.

---

## 5. Binding References Non-Existent `action_dump` in ChatInput

**Severity:** Low
**Status:** Fixed
**File:** `widgets.py`

### Problem

```python
BINDINGS = [Binding("ctrl+d", "dump", "Dump", show=True)]
```

`ChatInput` had no `action_dump()` method. The key only worked because `on_key()` intercepted it before the binding fired.

### Fix Applied
- Implemented `action_dump()` on `ChatInput` that posts a `ChatInputDump` message.
- Removed the manual `ctrl+d` handling from `on_key()`, letting Textual's binding system dispatch the action cleanly.

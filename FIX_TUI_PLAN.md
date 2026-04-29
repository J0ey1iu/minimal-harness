# Code Analysis: Bad Practices & Function Bugs

## Bad Practices

### 1. PersistentMemory wraps ConversationMemory (you identified)

**File:** `src/minimal_harness/client/built_in/memory.py:27`

`PersistentMemory` holds `self._inner = ConversationMemory(...)` and delegates every method. This adds an unnecessary indirection layer. `PersistentMemory` should implement the `Memory` protocol directly with its own storage, eliminating the double-dispatch.

**Fix plan:**
- Make `PersistentMemory` implement the `Memory` protocol independently
- Inline the message list, usage tracking, and system prompt handling currently delegated to `ConversationMemory._inner`
- Remove the composition pattern entirely

---

### 2. `AppContext.memory` is a shared mutable reference

**File:** `src/minimal_harness/client/built_in/context.py:38`
**File:** `src/minimal_harness/client/built_in/session_controller.py:111`

`AppContext` stores `self.memory: PersistentMemory | None`. In `create_session()`, `ctx.memory` is set to `None` then rebuilt — the resulting `PersistentMemory` is ALSO assigned as `session.memory`. Both `ctx` and the session point to the same object. Calling `ctx.reset_memory()` would silently orphan the session's memory. This creates temporal coupling between `AppContext` and `SessionController`.

**Fix plan:**
- Remove `self.memory` from `AppContext` entirely
- `AppContext` should not own a memory reference; that's a session concern
- `rebuild()` should accept and modify a memory instance passed in, or create one and return it
- `SessionController.create_session()` should own the `PersistentMemory` lifecycle exclusively

---

### 3. Preset agents all share one LLMProvider instance

**File:** `src/minimal_harness/client/built_in/session_controller.py:167`

```python
llm = self._ctx._create_llm_provider(self._ctx.config)
for a in agents:
    agent = SimpleAgent(llm_provider=llm, ...)
```

Every preset agent gets the same `llm` object. If the provider has any state (connection pools, rate-limit tracking), they're all coupled.

**Fix plan:**
- Move the `_create_llm_provider()` call inside the loop so each agent gets a fresh instance

---

### 4. AppContext.rebuild() accumulates tools in the registry without clearing

**File:** `src/minimal_harness/client/built_in/context.py:62-66`

`rebuild()` calls `self.registry.register(t)` for every tool but never calls `self.registry.clear()`. In contrast, `refresh_tools()` correctly calls `self.registry.clear()` first. This inconsistency means repeated calls to `rebuild()` will cause the registry to grow, and listener notifications fire more than necessary.

**Fix plan:**
- Add `self.registry.clear()` at the start of `rebuild()`, matching `refresh_tools()` behavior

---

### 5. `type: ignore` for `selected_tools` exposes a protocol gap

**File:** `src/minimal_harness/client/built_in/session_controller.py:135,543`

```python
session.memory.selected_tools = default_tools  # type: ignore[reportAttributeAccessIssue]
```

`s`ession.memory` is typed as `Memory` (the protocol), but `selected_tools` is a `PersistentMemory`-only attribute.

**Fix plan:**
- Add `selected_tools: list[str]` to the `Memory` protocol in `src/minimal_harness/memory.py`
- Implement it in `ConversationMemory` (as a no-op or actual field)
- This removes the need for all `type: ignore` comments

---

### 6. `LLMChunk.is_done` is dead code

**File:** `src/minimal_harness/agent/simple.py:99`
**File:** `src/minimal_harness/types.py:99-101`

`simple.py` always yields `LLMChunk(chunk, False)` — `is_done` is never `True`. The display uses `LLMEndEvent` to signal completion. Both `LLMChunk.is_done` and `LLMChunkEvent.is_done` are unused.

**Fix plan:**
- Remove the `is_done` field from `LLMChunk` and `LLMChunkEvent`

---

### 7. Handoff results silently discarded (function bug)

**File:** `src/minimal_harness/client/built_in/app.py:354-389`
**File:** `src/minimal_harness/client/built_in/session_controller.py:248-270`

The handoff tool starts a background run for the target agent, but the target's streaming events are never displayed:

1. `_poll_handoff_events()` drains events for `current_session_id` — but `drain_session_events()` immediately returns `[], False` because the current session IS the foreground session (`session_controller.py:225-226`).
2. Handoff target sessions are only checked for **completion** — `poll_handoff_completion()` silently drains all remaining events and discards them.
3. The user only sees `"→ Delegated to {name}"` and `"✓ Handoff completed"` — the actual handoff output (reasoning, response text, tool calls) is lost.

**Fix plan:**
- Remove the silent drain in `poll_handoff_completion()` — this was the only actual bug
- `_poll_handoff_events()` already correctly drains only the current (viewed) session; handoff events should only be seen when the user switches to that session
- Handoff output is NOT lost — the agent writes to `memory.add_message()` directly, so the handoff session's persistent memory is always correct

---

### 8. `register_handoff_run` silently loses handoff tracking (function bug)

**File:** `src/minimal_harness/client/built_in/session_controller.py:96-100`

When a handoff occurs, `register_handoff_run` searches `self._sessions` by `.name`. If no session matches the agent name, the function returns without registering — the run is completely untracked, events are never drained, and the task leaks.

**Fix plan:**
- If no existing session is found by name, create a new `ConversationSession` (with a fresh `PersistentMemory`) for the handoff target
- Register it in `self._sessions` and `self._active_runs`

---

### 9. `PersistentMemory.from_session` rejects non-hex session IDs (function bug)

**File:** `src/minimal_harness/client/built_in/persistent_memory.py:154`

```python
if not re.fullmatch(r"[a-f0-9]{32}", session_id):
    raise ValueError(f"Invalid session_id: {session_id!r}")
```

This is overly strict. Session IDs might be UUIDs with hyphens, or arbitrary strings from other sources.

**Fix plan:**
- Relax validation to only check for path-traversal characters (`../` or `/`)
- Or use a more inclusive regex

---

### 10. Race condition in `_poll_handoff_events` / `_run` (function bug)

**File:** `src/minimal_harness/client/built_in/app.py:306-343` vs `354-389`

Both `_run` and `_poll_handoff_events` call `d.handle_event()` on the same `buf` and `memory`. Since `_poll_handoff_events` runs on a timer (`_tick`, every 250ms), it can interleave with the main `_run` task — both mutating `buf` and `memory` concurrently, causing corrupted display state or lost events.

**Fix plan:**
- Make `_poll_handoff_events` skip processing if `_run` (streaming) is active
- Or use a lock/mutex around `buf` and `memory` mutations

---

### 11. Session created via handoff reuses preset agent's `created_at` (function bug, you identified)

When handoff creates a background run (`runtime.py:152-158`), `memory=None` is passed to `self.run()`. `SimpleAgent.run()` then uses `self._memory` — which for preset agents was created during `register_preset_agents()` at app startup. The handoff's session carries the original `created_at`, not the handoff time.

**Fix plan:**
- Fix plan for bug #8 already addresses this — creating a fresh `PersistentMemory` for each handoff run gives it the correct `created_at`

---

## Migration Summary

| # | Issue | Impact | Fix Complexity |
|---|-------|--------|----------------|
| 1 | PersistentMemory wraps ConversationMemory | Maintainability | Medium |
| 2 | AppContext owns shared memory ref | Correctness (stale ref risk) | Medium |
| 3 | Shared LLMProvider across preset agents | Correctness (state coupling) | Low |
| 4 | rebuild() doesn't clear registry | Correctness (registry bloat) | Low |
| 5 | Protocol gap for selected_tools | Maintainability (type ignores) | Low |
| 6 | LLMChunk.is_done dead code | Cleanliness | Low |
| 7 | Handoff results silently discarded | **Feature bug** (lost output) | Medium |
| 8 | register_handoff_run silently fails | **Bug** (untracked tasks) | Medium |
| 9 | from_session rejects non-hex IDs | **Bug** (false rejection) | Low |
| 10 | Race in _poll_handoff_events / _run | **Bug** (corrupted display) | Low |
| 11 | Handoff uses stale created_at | **Bug** (wrong timestamp) | Fixed by #8 |

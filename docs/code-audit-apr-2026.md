# Code Audit Report — April 2026

## 1. Protocol Layer Quality

### Good: Protocol-based interfaces

`Agent`, `LLMProvider`, `Memory`, `ToolRegistrationProtocol` are all `Protocol` classes — correct decoupling at the interface level. `SimpleAgent`, `ConversationMemory`, `OpenAILLMProvider`, `ToolRegistry` are the concrete implementations. The two-layer event model (internal `AgentEvent` → conversion → client `Event`) is a well-thought-out design.

#### ✅ Resolved — April 2026

`to_client_event()` was removed from all internal event dataclasses in `types.py`. The lazy `_client_events()` import hack and the `Event` TYPE_CHECKING import were both deleted. A standalone `to_client_event()` function was added to `client/events.py`. All callers (`app.py`, `test_client.py`, `relation_digging/main.py`) were updated to use `to_client_event(event)` instead of `event.to_client_event()`. `to_client_event()` is also re-exported from `client/__init__.py`. `types.py` no longer has any dependency on `client.events`.

### Critical issue (previously): Reverse dependency in event system

Internal events in `types.py` used to have `to_client_event()` methods importing from `client/events.py` via a lazy `_client_events()` workaround — a classic dependency inversion violation.

#### ✅ Resolved — April 2026

`StreamingTool.to_schema()` now returns `dict`. The `from openai.types.chat import ChatCompletionToolUnionParam` import was removed from `tool/base.py`. The OpenAI provider (`llm/openai.py`) uses a `# type: ignore[arg-type]` at the call site since the dict shape matches what OpenAI's API expects. The Anthropic provider already used `to_anthropic_schema()` which was unaffected.

### Minor dependency leak (previously): OpenAI SDK type in tool schema

`StreamingTool.to_schema()` (`tool/base.py:90`) used to return `ChatCompletionToolUnionParam` from `openai.types.chat`, coupling the core tool abstraction to the OpenAI SDK.

### Settings is a static class

`settings.py` uses classmethods. This is untestable — you cannot inject a different config source. Should be a proper dependency (dataclass) passed to constructors rather than read from env vars globally.

---

## 2. TUI Module Issues

### Refactoring direction is right but incomplete

`StreamPresenter`, `ExportPresenter`, and `SessionManager` were extracted from the monolithic `tui.py`, but they still carry tight coupling to `TUIApp`:

- `StreamPresenter` takes a `TUIApp` and accesses `_app._chat`, `_app.streaming`, `_app._chat_width`, `_app.theme`, `_app._next_msg_id()`, `_app._export_history` — private implementation details of the app.
- `ExportPresenter` similarly accesses `_app._chat_width`, `_app._export_history`, `_app.theme`.
- `SessionManager` goes three levels deep: `_app._input._input_history` (`session_manager.py:72`) — reach-through across two private attributes of different classes.

### TUIApp is still too large

~570 lines handling: UI composition, event dispatch, streaming, keyboard mappings, memory dumping, session management, tool selection, configuration, SVG export, and banner display. Too many responsibilities.

### PersistentMemory flushes on every mutation

`_flush()` writes to disk on every `add_message()`, `clear_messages()`, `set_message_usage()`, etc. No debouncing. For long conversations this is unnecessary I/O.

### Textual is a hard dependency

Listed as a required dependency in `pyproject.toml`. Users who only want the agent framework (no TUI) still install Textual. It should be an optional extra: `minimal-harness[tui]`.

### Zero TUI test coverage

The 5 test files only cover OpenAI/Anthropic providers, basic agent run, and external tool loading. The TUI has no tests at all.

### Minor issues

- `config.py:124` imports `warnings` conditionally inside a function body rather than at module top level.
- `.env` file in the repository root contains real-looking development credentials (though it is gitignored).

---

## 3. Vision Alignment

Per `docs/vision.md`:

### Phase 1 (What Is an Agent): Largely complete

Agent concept exists (`SimpleAgent`), event emitter pattern works, TUI is a proof-of-concept consumer. The `to_client_event()` mechanism validates that the consumer-facing side works.

### Phase 2 (Agent Service): Not started

No code exists for:
- Agent as a long-lived service with message inbox
- Receiving/processing messages from other agents
- Deployment-oriented entry points
- Any multi-agent infrastructure

### Phase 3 (Full System): Not started

No code for agent registry, communication tunnels, or collaboration patterns.

---

## 4. Suggestions Moving Forward

| Priority | Action |
|----------|--------|
| **Immediate** | ~~Fix the reverse dependency: remove `to_client_event()` from internal events, use a pure converter in `client/events.py`~~ ✅ |
| **Immediate** | ~~Make `to_schema()` return `dict`, push OpenAI-specific types to the provider layer~~ ✅ |
| **Short-term** | Define proper interfaces (`Protocol` classes) for `StreamPresenter`/`ExportPresenter` dependencies instead of passing `TUIApp` |
| **Short-term** | Make Textual optional via `[tui]` extra |
| **Medium-term** | Add debounced flush to `PersistentMemory` |
| **Medium-term** | Refactor `TUIApp` — extract event dispatch, session, and tool/configuration workflows into dedicated coordinator classes |
| **Phase 2 prep** | Begin designing the `AgentService` abstraction: think about message queues, agent identity, and capability advertisement. The current `SimpleAgent` is single-call (`run()` yields events then returns); Phase 2 needs a persistent event loop |

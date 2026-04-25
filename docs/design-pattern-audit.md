# Design & Programming Pattern Audit Report

## Overview

The codebase shows clear signs of rapid iteration: the core abstractions (`Agent`, `LLMProvider`, `ToolRegistrationProtocol`) exist on paper but are bypassed by concrete imports in `FrameworkClient`, `OpenAIAgent`, and especially `TUIApp`. The event system is duplicated, error handling between subprocess tools and in-process tools is bifurcated, and several critical resource-management and security issues (`eval`, unclosed streams, zombie subprocesses) need immediate attention.

**Recommended priority:**
1. **Critical first** — Fix resource leaks, security issues, and abstraction leaks to make the framework safe and genuinely pluggable.
2. **Major next** — Deduplicate the event system, centralize configuration, and split the `TUIApp` god object to improve maintainability.
3. **Minor last** — Clean up style inconsistencies and micro-optimizations.

---

## Critical Issues (Correctness, Security, Extensibility)

| # | Problem | Location | Refactor Suggestion | Why | Status |
|---|---------|----------|---------------------|-----|--------|
| 3 | **`TUIApp` is a God Object** — It mixes UI, business logic, and infrastructure (theming, slash commands, agent/LLM/memory stack rebuilding, SVG export, memory dump, tool selection, config persistence). | `src/minimal_harness/client/built_in/app.py` (lines 68–647) | Introduce an `AppContext` or `ClientFactory` that owns configuration, registry, and agent lifecycle, and inject it into `TUIApp`. The TUI should only call methods on the context. | Separating concerns makes the TUI unit-testable, reduces the class size by ~60%, and prevents UI changes from breaking business logic. | **Open** — ~550 lines, some decomposition via properties (`config`, `memory`, `active_tools`, `agent` expose `ctx` fields), but class still owns all business logic. |

---

## Major Issues (Maintainability, Architecture, Duplication)

| # | Problem | Location | Refactor Suggestion | Why | Status |
|---|---------|----------|---------------------|-----|--------|
| 4 | **Two parallel, near-identical event hierarchies** — Internal events in `types.py` and client events in `client/events.py` with a long manual `if/elif` conversion chain. | `src/minimal_harness/types.py`, `src/minimal_harness/client/events.py`, `src/minimal_harness/client/client.py:40–65` | Make client events thin wrappers/aliases of internal events, or add a `to_client_event()` method on each internal event dataclass. | Eliminates duplication, prevents field drift, and makes adding new events a single-file change. | **Partially Resolved** — internal events now have `to_client_event()` method at `types.py:80+`, used in `app.py:322`, `test_client.py:151,282`, and `relation_digging/main.py:52`. Two hierarchies still exist. |
| 5 | **Duplicated OpenAI client construction logic** | `test/test_client.py:111–126`, `examples/relation_digging/main.py:17–24` | Add a factory in `minimal_harness.llm.openai` (e.g., `create_openai_client(api_key, base_url)`). | DRY. Centralizes default handling and makes it easier to add retry logic or logging later. | **Open** — identical conditional blocks still in both locations. |
| 6 | **`ChatInput` bypasses Textual's declarative action framework** | `src/minimal_harness/client/built_in/widgets.py` (lines 42–69) | Move `action_submit` bindings to the `App` level and use Textual's `BINDINGS` system. | Using the framework's routing makes key behavior configurable and testable without simulating low-level key events. | **Open** — raw `on_key` handler still intercepts Enter directly. |

---

## Additional Findings from built_in/ Review

The files in `src/minimal_harness/client/built_in/` were reviewed in full (`app.py`, `buffer.py`, `config.py`, `context.py`, `modals.py`, `tui.py`, `widgets.py`, `__init__.py`). Issues are categorized below. References to items from the original report (e.g. "[#1]") indicate direct elaboration of a pre-existing finding.

### Critical Issues (Correctness, Security, Extensibility)

| # | Problem | Location | Refactor Suggestion | Why | Status |
|---|---------|----------|---------------------|-----|--------|
| 8 | **`AppContext` hardcodes the entire OpenAI stack** — `rebuild()` directly instantiates `AsyncOpenAI`, `OpenAILLMProvider`, and `OpenAIAgent` with no abstraction layer. This locks the entire client to OpenAI and prevents any other LLM from being used regardless of the `LLMProvider` protocol. | `src/minimal_harness/client/built_in/context.py` (lines 8–9, 54–75) | Replace the concrete imports with the `LLMProvider` and `Agent` protocol types. Accept an optional `llm_provider_factory: Callable[[dict, ToolRegistry], LLMProvider]` and `agent_factory: Callable[[LLMProvider, list[Tool], ConversationMemory], Agent]` constructor parameter on `AppContext`. | Extensibility is completely blocked; adding Anthropic, local models, or test providers requires modifying `AppContext` instead of being a drop-in swap. | **Resolved** — `AppContext.__init__` accepts `llm_provider_factory` and `agent_factory` parameters. `_create_llm_provider()` supports both OpenAI and Anthropic based on `cfg.get("provider")`. |

### Major Issues (Maintainability, Architecture, Duplication)

| # | Problem | Location | Refactor Suggestion | Why | Status |
|---|---------|----------|---------------------|-----|--------|
| 12 | **Eager module-level imports inside `_banner`** — `get_bash_tools` and `get_patch_file_tools` are imported at lines 329–332 inside the `_banner` method, which is called on every app mount and after `/new`. These imports should be at the top of the file or cached after first load. | `src/minimal_harness/client/built_in/app.py` (lines 329–332) | Move the two import lines to module level, or cache the result of `get_bash_tools()` / `get_patch_file_tools()` in a module-level set the first time `_banner` runs. | Repeated import overhead on every `/new` command; more importantly, the import side-effects (module-level decorator registration in those modules) will fire multiple times. | **Resolved** — imports now wrapped in `_get_built_in_tool_names()` with `_BUILT_IN_TOOL_NAMES` global cache (lines 70–84). |
| 13 | **`AppContext.rebuild()` unconditionally creates a new `ConversationMemory` even when the system prompt is unchanged** — Lines 60–69 in `rebuild()` always replace `self.memory` if the system prompt in the config differs from the built-in default, but on many rebuild cycles (e.g., after tool selection changes that call `rebuild()`) the system prompt is unchanged, yet a new `ConversationMemory` is still allocated. | `src/minimal_harness/client/built_in/context.py` (lines 60–69) | Only replace `self.memory` if the prompt value has actually changed from the previously loaded prompt, not just different from the default. Store `self._last_prompt_path` and compare before allocating. | Unnecessary object churn; more critically, it breaks references to the old memory object that may be held by async tasks still running when `rebuild()` is called mid-stream. | **Partially Resolved** — now checks `if self.memory is None` first and only replaces when system prompt content differs (lines 77–86). Edge case on rebuild with empty memory and unchanged prompt remains. |

(End of file - total 48 lines)
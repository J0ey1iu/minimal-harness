# Design & Programming Pattern Audit Report

## Overview

The codebase shows clear signs of rapid iteration: the core abstractions (`Agent`, `LLMProvider`, `ToolRegistrationProtocol`) exist on paper but are bypassed by concrete imports in `FrameworkClient`, `OpenAIAgent`, and especially `TUIApp`. The event system is duplicated, error handling between subprocess tools and in-process tools is bifurcated, and several critical resource-management and security issues (`eval`, unclosed streams, zombie subprocesses) need immediate attention.

**Recommended priority:**
1. **Critical first** — Fix resource leaks, security issues, and abstraction leaks to make the framework safe and genuinely pluggable.
2. **Major next** — Deduplicate the event system, centralize configuration, and split the `TUIApp` god object to improve maintainability.
3. **Minor last** — Clean up style inconsistencies and micro-optimizations.

---

## Critical Issues (Correctness, Security, Extensibility)

| # | Problem | Location | Refactor Suggestion | Why |
|---|---------|----------|---------------------|-----|
| 1 | **TUI directly peeks into OpenAI-specific chunk shapes** — `event.chunk.choices[0].delta`, `delta.reasoning_content`, `delta.tool_calls`, etc. are hard-coded in the presentation layer, making the TUI unusable with any other LLM provider. | `src/minimal_harness/client/built_in/app.py` (lines 489–515) | Normalize provider-specific deltas into a provider-agnostic `LLMChunk` dataclass inside the LLM layer. `LLMChunkEvent` should carry only generic fields (`content`, `reasoning`, `tool_call_deltas`). The TUI should consume these generic fields. | Closes the abstraction leak and makes the TUI truly pluggable, which is the stated purpose of the `LLMProvider` protocol. |
| 2 | **`OpenAIAgent` is locked to `OpenAILLMProvider` despite an `LLMProvider` protocol** | `src/minimal_harness/agent/openai.py` (line 34) | Change the type hint to `llm_provider: LLMProvider`. | Allows swapping LLM backends without touching agent or client code. |
| 3 | **`TUIApp` is a God Object** — It mixes UI, business logic, and infrastructure (theming, slash commands, agent/LLM/memory stack rebuilding, SVG export, memory dump, tool selection, config persistence). | `src/minimal_harness/client/built_in/app.py` (lines 68–647) | Introduce an `AppContext` or `ClientFactory` that owns configuration, registry, and agent lifecycle, and inject it into `TUIApp`. The TUI should only call methods on the context. | Separating concerns makes the TUI unit-testable, reduces the class size by ~60%, and prevents UI changes from breaking business logic. |

---

## Major Issues (Maintainability, Architecture, Duplication)

| # | Problem | Location | Refactor Suggestion | Why |
|---|---------|----------|---------------------|-----|
| 6 | **Two parallel, near-identical event hierarchies** — Internal events in `types.py` and client events in `client/events.py` with a long manual `if/elif` conversion chain. | `src/minimal_harness/types.py`, `src/minimal_harness/client/events.py`, `src/minimal_harness/client/client.py:40–65` | Make client events thin wrappers/aliases of internal events, or add a `to_client_event()` method on each internal event dataclass. | Eliminates duplication, prevents field drift, and makes adding new events a single-file change. |
| 7 | **Duplicated OpenAI client construction logic** | `test/test_client.py:111–126`, `examples/relation_digging/main.py:17–24` | Add a factory in `minimal_harness.llm.openai` (e.g., `create_openai_client(api_key, base_url)`). | DRY. Centralizes default handling and makes it easier to add retry logic or logging later. |
| 8 | **`ToolRegistrationProtocol` is an incomplete contract** | `src/minimal_harness/tool/base.py` (lines 26–32) | Update the protocol to match the real `ToolRegistry.register_external_tool` signature, including `script_path` and `**kwargs`. | Protocols are documentation as much as they are type constraints. An incomplete protocol undermines the purpose of structural subtyping. |
| 9 | **Tests write artifacts to the repository root** | `test/test_client.py` (lines 171, 186, 201, 216, etc.) | Use pytest's `tmp_path` fixture for all output files. | Keeps the working directory clean, prevents flaky tests from leftover files, and avoids race conditions in parallel test runs. |
| 10 | **`ChatInput` bypasses Textual's declarative action framework** | `src/minimal_harness/client/built_in/widgets.py` (lines 42–69) | Move `action_submit` bindings to the `App` level and use Textual's `BINDINGS` system. | Using the framework's routing makes key behavior configurable and testable without simulating low-level key events. |

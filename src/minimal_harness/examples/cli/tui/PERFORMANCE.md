# ChatTUI Streaming Performance Optimization

## Problem

The ChatTUI app was extremely slow when streaming LLM responses, especially for long texts. Users experienced noticeable lag and stuttering during streaming.

## Root Causes

### 1. Markdown Parsing on Every Update

`ChatMessage` inherited from Textual's `Markdown` widget. Every `.update()` call triggered a full markdown parse of the **entire accumulated text**. As text grew, this became O(n²) — parsing 1K, 2K, 3K... chars on each update.

### 2. `scroll_end()` After Every Chunk

`history.scroll_end()` was called after every single chunk in 3 places. `scroll_end()` is an expensive DOM operation that forces layout recalculation.

### 3. Chunk-Count Batching Was Insufficient

Batching by chunk count (e.g., every 5 or 10 chunks) gave inconsistent performance — fast chunks meant frequent updates, and markdown parsing overhead still scaled with text length.

## Solutions Implemented

### 1. Use `Static` During Streaming, `Markdown` Only at End

Changed `ChatMessage` to inherit from `Static` instead of `Markdown`:

```python
# Before (slow - parses markdown every update)
class ChatMessage(Markdown):
    ...

# After (fast - plain text rendering)
class ChatMessage(Static):
    ...
```

During streaming, `Static.update()` renders plain text with zero markdown parsing overhead. When streaming completes (`is_done=True`), the `Static` widget is removed and replaced with a `Markdown` widget in one shot. Markdown parsing happens exactly **once** at the end.

### 2. Time-Based Throttling

Replaced chunk-count batching with time-based throttling:

```python
UPDATE_INTERVAL = 0.2  # 200ms
SCROLL_INTERVAL = 0.2  # 200ms
```

Display updates and scroll calls are now rate-limited to 5 times per second, regardless of chunk frequency. This provides consistent performance and UX.

### 3. Deferred Scroll After Markdown Swap

When `is_done=True`, the flow is:
1. Final `update()` on the `Static` with accumulated text
2. Remove the `Static` widget
3. Mount a `Markdown` widget with the full final text
4. Call `scroll_end()` via `call_later()` twice — once before and once after the markdown mount

The double `call_later` ensures `scroll_end()` runs after the markdown widget is fully laid out, fixing issues with long text not scrolling to bottom.

## Files Changed

- `src/minimal_harness/examples/cli/widgets.py` — `ChatMessage` base class changed from `Markdown` to `Static`
- `src/minimal_harness/examples/cli/tui/handlers.py` — Time-based throttling, markdown swap on done, deferred scroll

## Key Performance Gains

| Scenario | Before | After |
|----------|--------|-------|
| Streaming 10K char response | ~55K chars worth of markdown parsing | ~10K (one final parse) |
| Update frequency | Varies with chunk rate | Fixed 5/sec |
| Scroll calls | Every chunk (3 places) | Every 200ms max |
| Markdown parsing during streaming | Every update | Zero |

The streaming experience is now consistent regardless of response length.

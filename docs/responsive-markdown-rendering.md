# Responsive Markdown Rendering

This document explains how `minimal_harness` renders markdown content in the TUI so that it gracefully adapts to window resizes without breaking the visual layout of tables, code blocks, and horizontal rules.

## 1. The Problem

Rich's `Markdown` class renders markdown to ANSI escape sequences embedded in a `Text` object. This process bakes in a fixed character width. When the markdown contains elements with hard-drawn lines — tables, code blocks with panels, horizontal rules — those lines are drawn at the rendered width.

If the window is later resized to a narrower width, those pre-rendered lines are longer than the available space. The text wraps at the widget boundary, but the embedded ANSI escape sequences for the visual lines remain at their original positions, producing garbled output.

Example of broken rendering after window shrink:

```
┌───────────────┬──────┬───────────┐ ← table border drawn at 80 cols
│ Name  │Value │Descri│← wrapped, misaligned
│ption  │      │ption │
```

## 2. The Solution: LazyMarkdown

`LazyMarkdown` (`markdown_styles.py`) is a custom Rich renderable that **defers markdown rendering until display time**, when the actual available width is known.

```python
class LazyMarkdown:
    def __init__(self, text: str) -> None:
        self.text = text
        self._cache_width = 0
        self._cache_result: Text | None = None

    def __rich_console__(self, console: Console, options: ConsoleOptions):
        width = max(options.max_width, 20)
        if width == self._cache_width and self._cache_result is not None:
            yield self._cache_result
            return

        buf = StringIO()
        with Console(file=buf, force_terminal=True, width=width) as c:
            c.print(AppMarkdown(self.text))
        result = Text.from_ansi(buf.getvalue())
        self._cache_width = width
        self._cache_result = result
        yield result
```

Key design decisions:

- **`__rich_console__` receives `options.max_width`** — this is the widget's current content width, which changes on resize. Textual widgets call `__rich_console__` each time they are rendered, passing the current available width.
- **Caching per width** — if the same width is requested again (e.g., no resize), the cached `Text` is returned immediately without re-rendering.
- **`__rich_measure__`** — returns `Measurement(0, options.max_width)` so the widget reports its full available width to Textual's layout system, preventing unnecessary scrollbars or truncation.

```python
    def __rich_measure__(self, console: Console, options: ConsoleOptions):
        return Measurement(0, options.max_width)
```

## 3. How It Integrates with the TUI

In `app.py`, the `_render_markdown` method returns a `LazyMarkdown` instance instead of a pre-rendered `Text`:

```python
def _render_markdown(self, text: str, width: int = 80) -> LazyMarkdown:
    return LazyMarkdown(text)
```

All markdown content — whether from streaming, committed responses, or session replay — flows through `AssistantMsg` with a `LazyMarkdown` renderable. This means every markdown element is always rendered at the widget's current width.

### Rendering Pipeline

1. **Agent produces markdown string** (e.g., `"# Hello\n\n| A | B |\n|---|---|\n| 1 | 2 |"`)
2. **`say(is_markdown=True)`** calls `_render_markdown(text)` which wraps it in `LazyMarkdown`
3. **`AssistantMsg(LazyMarkdown(...))`** is mounted to the chat scroll container
4. **Textual renders the widget**, calling `LazyMarkdown.__rich_console__` with the current `options.max_width`
5. **`LazyMarkdown` renders `AppMarkdown` at exactly `options.max_width`** — producing a `Text` with lines that fit the widget
6. **On resize**, Textual re-renders the widget, `LazyMarkdown` detects the width changed, re-renders at the new width, and updates the cache

### Streaming

During streaming, the same pipeline applies. Each chunk update calls `_render_markdown` with the accumulated markdown, producing a fresh `LazyMarkdown`. When the widget re-renders (whether due to streaming updates or resize), it renders at the current width.

## 4. Why Not Textual's Native Markdown Widget?

The TUI originally used `MarkdownMsg` (Textual's built-in `Markdown` widget) for session replay. This widget handles resize natively but does not use `AppMarkdown`'s custom styling (tables, code blocks). Switching session replay to also use `AssistantMsg` with `LazyMarkdown` unified the rendering path:

- **Custom styling**: Tables have `box.ROUNDED` borders and bold headers; code blocks are wrapped in dim-rounded panels.
- **Consistent behavior**: Streaming and committed messages use the same rendering path.
- **No regression**: The original `MarkdownMsg` class is retained in `chat_widgets.py` for backwards compatibility with any external code that may reference it.

## 5. Custom Markdown Elements

`AppMarkdown` extends Rich's `Markdown` with three custom element overrides:

### Left-Aligned Headings

Rich's default `Heading` centers `h1` titles. `LeftHeading` forces all heading levels left:

```python
class LeftHeading(Heading):
    LEVEL_ALIGN = {f"h{i}": "left" for i in range(1, 7)}

elements["heading_open"] = LeftHeading
```

### Styled Tables

`StyledTableElement` replaces the default table element with rounded corners and visible row dividers:

```python
class StyledTableElement(MarkdownElement):
    def __rich_console__(self, console: Console, options: ConsoleOptions):
        table = Table(
            box=box.ROUNDED,
            pad_edge=False,
            style="markdown.table.border",
            show_edge=True,
            show_lines=True,       # row dividers
            collapse_padding=True,
            padding=(0, 1),
        )
        # ... build table from header and body elements
        yield table
```

### Styled Code Blocks

`StyledCodeBlock` wraps syntax-highlighted code in a rounded panel with a dim border:

```python
class StyledCodeBlock(BaseCodeBlock):
    def __rich_console__(self, console: Console, options: ConsoleOptions):
        code = str(self.text).rstrip()
        syntax = Syntax(code, self.lexer_name, theme=self.theme, word_wrap=True, padding=(0, 1))
        yield Panel(syntax, border_style="dim", box=box.ROUNDED, padding=(0, 0))
```

## 6. Export (SVG)

The `action_share` function exports the chat as SVG. It renders each message using the same `AppMarkdown` at the export width, producing a self-contained SVG where all lines are drawn at the correct width. Because `LazyMarkdown` is not used during export (the export path uses `_render_markdown` directly with a fixed width console), the export always produces clean output at the target width.

## 7. Summary

The responsive rendering system works by deferring markdown rendering to display time:

| Approach | Pros | Cons |
|---------|------|------|
| Pre-render to `Text` at fixed width | Fast, simple | Breaks on resize — hard-drawn lines don't reflow |
| Textual's native `Markdown` widget | Handles resize | Can't use custom `AppMarkdown` styling |
| `LazyMarkdown` (this design) | Renders at display width on every paint; cache prevents redundant re-renders; custom styling preserved | Slightly more CPU on first render at a new width |

`LazyMarkdown` is the bridge between the custom-styled `AppMarkdown` and Textual's layout system. It ensures that every render uses the correct width, producing clean output at any window size.

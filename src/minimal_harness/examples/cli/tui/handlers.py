from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable, cast

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionChunk
    from textual.containers import VerticalScroll

from minimal_harness.examples.cli.tui.thinking import extract_thinking
from minimal_harness.examples.cli.widgets import (
    ChatMessage,
    ThinkingWidget,
    ToolCallWidget,
    ToolResultWidget,
)

UPDATE_INTERVAL = 0.3
SCROLL_INTERVAL = 0.2


class StreamingState:
    def __init__(self) -> None:
        self.thinking_text = ""
        self.thinking_widget: "ThinkingWidget | None" = None
        self.thinking_finalized = False
        self.streaming_text = ""
        self.response_widget: "ChatMessage | None" = None
        self.tool_calls_detected: list[dict[str, Any]] = []
        self.tool_call_widgets: dict[int, "ToolCallWidget"] = {}
        self.last_update_time = 0.0
        self.last_scroll_time = 0.0
        self.pending_scroll = False
        self.pending_update = False


async def _ensure_thinking_widget(
    history: "VerticalScroll",
    state: StreamingState,
) -> "ThinkingWidget":
    if state.thinking_widget is None:
        tw = ThinkingWidget("Thinking…", classes="thinking")
        await history.mount(tw)
        state.thinking_widget = tw
        state.thinking_text = ""
        state.thinking_finalized = False
    return state.thinking_widget


async def _finish_thinking(
    history: "VerticalScroll",
    state: StreamingState,
) -> None:
    tw = state.thinking_widget
    if tw is not None and not state.thinking_finalized:
        if state.thinking_text:
            tw.update(f"Thinking\n{state.thinking_text}")
        else:
            tw.remove()
        state.thinking_finalized = True
        state.thinking_widget = None
        state.thinking_text = ""


async def _ensure_response_widget(
    history: "VerticalScroll",
    state: StreamingState,
) -> "ChatMessage":
    if state.response_widget is None:
        await _finish_thinking(history, state)
        rw = ChatMessage("", classes="assistant-message")
        await history.mount(rw)
        state.response_widget = rw
    return state.response_widget


def create_thinking_handler(
    history: "VerticalScroll",
    state: StreamingState,
) -> callable:
    async def _maybe_scroll(now: float) -> None:
        if now - state.last_scroll_time >= SCROLL_INTERVAL:
            history.scroll_end()
            state.last_scroll_time = now
            state.pending_scroll = False

    async def _maybe_update(now: float) -> None:
        if now - state.last_update_time >= UPDATE_INTERVAL:
            if state.response_widget is not None:
                cast(ChatMessage, state.response_widget).update(state.streaming_text)
            state.last_update_time = now
            state.pending_update = False

    async def on_chunk(chunk: "ChatCompletionChunk | None", is_done: bool) -> None:
        if is_done:
            if state.response_widget is not None and state.streaming_text:
                cast(ChatMessage, state.response_widget).update(state.streaming_text)
            state.pending_update = False
            await _finish_thinking(history, state)

            def _final_scroll() -> None:
                history.scroll_end()
                history.call_later(history.scroll_end)

            if state.response_widget is not None and state.streaming_text:
                widget_to_replace = state.response_widget
                final_text = state.streaming_text
                widget_to_replace.remove()
                state.response_widget = None

                from textual.widgets import Markdown

                final_widget = Markdown(final_text, classes="assistant-message")
                await history.mount(final_widget)
                state.response_widget = cast("ChatMessage", final_widget)

            history.call_later(_final_scroll)
            return

        if not chunk:
            return

        now = time.monotonic()

        delta = chunk.choices[0].delta if chunk.choices else None
        if not delta:
            return

        reasoning = extract_thinking(delta)
        if reasoning:
            tw = await _ensure_thinking_widget(history, state)
            state.thinking_text += reasoning
            tw.update(f"Thinking\n{state.thinking_text}")

        if delta.tool_calls:
            await _finish_thinking(history, state)
            for tc in delta.tool_calls:
                detected = state.tool_calls_detected
                tc_widgets = state.tool_call_widgets

                existing = next(
                    (t for t in detected if t["index"] == tc.index),
                    None,
                )
                if not existing:
                    existing = {
                        "index": tc.index,
                        "id": tc.id or "",
                        "name": tc.function.name if tc.function else "",
                        "arguments": tc.function.arguments
                        if tc.function and tc.function.arguments
                        else "",
                    }
                    detected.append(existing)
                    widget = ToolCallWidget(
                        f"\u2699  {existing['name']}()",
                        classes="tool-call",
                        markup=False,
                    )
                    await history.mount(widget)
                    tc_widgets[tc.index] = widget
                else:
                    if tc.id:
                        existing["id"] += tc.id
                    if tc.function:
                        if tc.function.name:
                            existing["name"] += tc.function.name
                        if tc.function.arguments:
                            existing["arguments"] += tc.function.arguments

                if tc.index in tc_widgets:
                    tc_widgets[tc.index].update(
                        f"\u2699  {existing['name']}({existing['arguments']})"
                    )

        if delta.content:
            state.streaming_text += delta.content
            if state.response_widget is None:
                await _ensure_response_widget(history, state)
            await _maybe_update(now)

        await _maybe_scroll(now)

    return on_chunk


def create_tool_end_handler(
    history: "VerticalScroll",
    state: StreamingState,
    on_memory_status_update: "Callable[[], None] | None" = None,
) -> callable:
    async def on_tool_end(tool_call: Any, result: Any) -> None:
        state.response_widget = None
        state.streaming_text = ""
        state.tool_calls_detected = []
        state.tool_call_widgets = {}

        result_widget = ToolResultWidget(
            f"\u2713  {result}", classes="tool-result", markup=False
        )
        await history.mount(result_widget)
        history.scroll_end()

        if on_memory_status_update is not None:
            on_memory_status_update()

    return on_tool_end

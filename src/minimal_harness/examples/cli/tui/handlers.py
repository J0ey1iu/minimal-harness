from __future__ import annotations

from typing import TYPE_CHECKING, Any

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


class StreamingState:
    def __init__(self) -> None:
        self.thinking_text = ""
        self.thinking_widget: "ThinkingWidget | None" = None
        self.thinking_finalized = False
        self.streaming_text = ""
        self.response_widget: "ChatMessage | None" = None
        self.tool_calls_detected: list[dict[str, Any]] = []
        self.tool_call_widgets: dict[int, "ToolCallWidget"] = {}


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
        rw = ChatMessage("Assistant\n", classes="assistant-message")
        await history.mount(rw)
        state.response_widget = rw
    return state.response_widget


def create_thinking_handler(
    history: "VerticalScroll",
    state: StreamingState,
) -> callable:

    async def on_chunk(chunk: "ChatCompletionChunk | None", is_done: bool) -> None:
        if is_done:
            await _finish_thinking(history, state)
            return

        if not chunk:
            return

        delta = chunk.choices[0].delta if chunk.choices else None
        if not delta:
            return

        reasoning = extract_thinking(delta)
        if reasoning:
            tw = await _ensure_thinking_widget(history, state)
            state.thinking_text += reasoning
            tw.update(f"Thinking\n{state.thinking_text}")
            history.scroll_end()

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
            history.scroll_end()

        if delta.content:
            state.streaming_text += delta.content
            w = await _ensure_response_widget(history, state)
            w.update(f"Assistant\n{state.streaming_text}")
            history.scroll_end()

    return on_chunk


def create_tool_end_handler(
    history: "VerticalScroll",
    state: StreamingState,
) -> callable:
    async def on_tool_end(tool_call: Any, result: Any) -> None:
        state.response_widget = None
        state.streaming_text = ""
        state.tool_calls_detected = []
        state.tool_call_widgets = {}

        result_widget = ToolResultWidget(f"\u2713  {result}", classes="tool-result")
        await history.mount(result_widget)
        history.scroll_end()

    return on_tool_end

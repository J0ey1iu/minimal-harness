"""Streaming display controller — manages live-updating widgets during agent output."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from minimal_harness.client.built_in.buffer import StreamBuffer
from minimal_harness.client.built_in.chat_widgets import (
    AssistantMsg,
    ReasoningMsg,
    ToolCallMsg,
)
from minimal_harness.client.built_in.renderer import format_tool_call_static

if TYPE_CHECKING:
    from textual.containers import VerticalScroll


class StreamingController:
    def __init__(
        self,
        chat: VerticalScroll,
        render_markdown: Callable,
        next_msg_id: Callable[[], str],
    ) -> None:
        self._chat = chat
        self._render_markdown = render_markdown
        self._next_id = next_msg_id
        self._reasoning: ReasoningMsg | None = None
        self._content: AssistantMsg | None = None
        self._tool_widgets: dict[int, ToolCallMsg] = {}
        self._last_content: str = ""
        self._last_reasoning: str = ""

    def clear(self) -> None:
        self._reasoning = None
        self._content = None
        self._tool_widgets.clear()
        self._last_content = ""
        self._last_reasoning = ""

    def tick(self, buf: StreamBuffer, streaming: bool, width: int) -> None:
        if not streaming:
            return
        chat = self._chat
        max_scroll = chat.max_scroll_y
        at_bottom = max_scroll == 0 or chat.scroll_y >= max_scroll
        if not at_bottom:
            return

        cur_reasoning = buf.reasoning
        cur_content = buf.content

        if cur_reasoning:
            if cur_reasoning != self._last_reasoning:
                self._last_reasoning = cur_reasoning
                if self._reasoning is None:
                    self._reasoning = ReasoningMsg(cur_reasoning, id=self._next_id())
                    chat.mount(self._reasoning)
                else:
                    self._reasoning.update(cur_reasoning)
        elif self._reasoning is not None:
            self._reasoning.remove()
            self._reasoning = None
            self._last_reasoning = ""

        if cur_content:
            if cur_content != self._last_content:
                self._last_content = cur_content
                rendered = self._render_markdown(cur_content, width)
                if self._content is None:
                    self._content = AssistantMsg(rendered, id=self._next_id())
                    chat.mount(self._content)
                else:
                    self._content.update(rendered)
        elif self._content is not None:
            self._content.remove()
            self._content = None
            self._last_content = ""

        if buf.tool_calls:
            prev_ids = set(self._tool_widgets.keys())
            cur_ids = set(buf.tool_calls.keys())
            for idx in prev_ids - cur_ids:
                self._tool_widgets[idx].remove()
                del self._tool_widgets[idx]
            for idx, call in sorted(buf.tool_calls.items()):
                tw = format_tool_call_static(call)
                tw.no_wrap = False
                tw.overflow = "fold"
                if idx in self._tool_widgets:
                    self._tool_widgets[idx].update(tw)
                else:
                    w = ToolCallMsg(tw, id=self._next_id())
                    chat.mount(w)
                    self._tool_widgets[idx] = w
        elif self._tool_widgets:
            for w in self._tool_widgets.values():
                w.remove()
            self._tool_widgets.clear()

        chat.call_after_refresh(chat.scroll_end, animate=False)

    def flush(
        self, buf: StreamBuffer, width: int
    ) -> tuple[str, str, dict[int, dict[str, str]]]:
        """Finalize streaming widgets and return flushed content for permanent display."""
        had_content = bool(buf.reasoning or buf.content)
        if self._reasoning is not None:
            self._reasoning.remove()
            self._reasoning = None
        if self._content is not None:
            self._content.remove()
            self._content = None
        for w in self._tool_widgets.values():
            w.remove()
        self._tool_widgets.clear()

        reasoning = buf.reasoning
        content = buf.content
        tool_calls = dict(buf.tool_calls)
        buf.tool_calls.clear()

        if had_content:
            buf.mark_flushed()
            self._chat.call_after_refresh(self._chat.scroll_end, animate=False)
        buf.reasoning = ""
        buf.content = ""
        self._last_content = ""
        self._last_reasoning = ""
        return reasoning, content, tool_calls

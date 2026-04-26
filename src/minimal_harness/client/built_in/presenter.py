"""Streaming presentation logic extracted from TUIApp."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text

from minimal_harness.client.built_in.buffer import StreamBuffer
from minimal_harness.client.built_in.chat_widgets import (
    AssistantMsg,
    ReasoningMsg,
    ToolCallMsg,
)
from minimal_harness.client.built_in.markdown_styles import (
    LazyMarkdown,
    resolve_code_theme,
)
from minimal_harness.client.built_in.renderer import format_tool_call_static

if TYPE_CHECKING:
    from textual.containers import VerticalScroll

    from minimal_harness.client.built_in.app import TUIApp


class StreamPresenter:
    def __init__(self, app: TUIApp) -> None:
        self._app = app
        self._streaming_reasoning: ReasoningMsg | None = None
        self._streaming_content: AssistantMsg | None = None
        self._streaming_tool_widgets: dict[int, ToolCallMsg] = {}

    @property
    def _chat(self) -> VerticalScroll:
        return self._app._chat

    def render_markdown(self, text: str, width: int = 80) -> LazyMarkdown:
        code_theme = resolve_code_theme(self._app.theme)
        return LazyMarkdown(text, code_theme=code_theme)

    def tick(self, buf: StreamBuffer) -> None:
        if not self._app.streaming:
            return
        chat = self._chat
        max_scroll = chat.max_scroll_y
        at_bottom = max_scroll == 0 or chat.scroll_y >= max_scroll
        if not at_bottom:
            return
        width = self._app._chat_width
        if buf.reasoning:
            if self._streaming_reasoning is None:
                self._streaming_reasoning = ReasoningMsg(
                    buf.reasoning, id=self._app._next_msg_id()
                )
                chat.mount(self._streaming_reasoning)
            else:
                self._streaming_reasoning.update(buf.reasoning)
        elif self._streaming_reasoning is not None:
            self._streaming_reasoning.remove()
            self._streaming_reasoning = None
        if buf.content:
            rendered = self.render_markdown(buf.content, width)
            if self._streaming_content is None:
                self._streaming_content = AssistantMsg(rendered, id=self._app._next_msg_id())
                chat.mount(self._streaming_content)
            else:
                self._streaming_content.update(rendered)
        elif self._streaming_content is not None:
            self._streaming_content.remove()
            self._streaming_content = None
        if buf.tool_calls:
            prev_ids = set(self._streaming_tool_widgets.keys())
            cur_ids = set(buf.tool_calls.keys())
            for idx in prev_ids - cur_ids:
                self._streaming_tool_widgets[idx].remove()
                del self._streaming_tool_widgets[idx]
            for idx, call in sorted(buf.tool_calls.items()):
                tw = format_tool_call_static(call)
                tw.no_wrap = False
                tw.overflow = "fold"
                if idx in self._streaming_tool_widgets:
                    self._streaming_tool_widgets[idx].update(tw)
                else:
                    w = ToolCallMsg(tw, id=self._app._next_msg_id())
                    chat.mount(w)
                    self._streaming_tool_widgets[idx] = w
        elif self._streaming_tool_widgets:
            for w in self._streaming_tool_widgets.values():
                w.remove()
            self._streaming_tool_widgets.clear()
        chat.call_after_refresh(chat.scroll_end, animate=False)

    def flush(self, buf: StreamBuffer) -> None:
        had_content = bool(buf.reasoning or buf.content)
        if self._streaming_reasoning is not None:
            self._streaming_reasoning.remove()
            self._streaming_reasoning = None
        if self._streaming_content is not None:
            self._streaming_content.remove()
            self._streaming_content = None
        for w in self._streaming_tool_widgets.values():
            w.remove()
        self._streaming_tool_widgets.clear()
        if buf.reasoning:
            mid = self._app._next_msg_id()
            w = ReasoningMsg(buf.reasoning, id=mid)
            self._chat.mount(w)
            self._app._export_history.append((buf.reasoning, "dim", False))
        if buf.content:
            rendered = self.render_markdown(buf.content, self._app._chat_width)
            mid = self._app._next_msg_id()
            w = AssistantMsg(rendered, id=mid)
            self._chat.mount(w)
            self._app._export_history.append((buf.content, None, True))
        if buf.tool_calls:
            for _, call in sorted(buf.tool_calls.items()):
                tw = format_tool_call_static(call)
                tw.no_wrap = False
                tw.overflow = "fold"
                mid = self._app._next_msg_id()
                w = ToolCallMsg(tw, id=mid)
                self._chat.mount(w)
                self._app._export_history.append(
                    (tw.plain, str(tw.style) if tw.style else None, False)
                )
            buf.tool_calls.clear()
        if had_content:
            buf._flushed = True
            self._chat.call_after_refresh(self._chat.scroll_end, animate=False)
        buf.reasoning = ""
        buf.content = ""

    def say_tool_call(self, text: Text) -> None:
        mid = self._app._next_msg_id()
        w = ToolCallMsg(text, id=mid)
        self._chat.mount(w)
        w.scroll_visible()
        self._chat.call_after_refresh(self._chat.scroll_end, animate=False)
        self._app._export_history.append(
            (text.plain, str(text.style) if text.style else None, False)
        )

    def say_tool_result(self, text: Text) -> None:
        mid = self._app._next_msg_id()
        w = ToolCallMsg(text, id=mid)
        self._chat.mount(w)
        w.scroll_visible()
        self._chat.call_after_refresh(self._chat.scroll_end, animate=False)
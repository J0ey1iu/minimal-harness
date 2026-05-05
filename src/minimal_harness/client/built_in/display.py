"""Chat display — handles all content rendered in the chat area."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from rich.text import Text

from minimal_harness.client.built_in.buffer import StreamBuffer
from minimal_harness.client.built_in.chat_widgets import (
    AssistantMsg,
    ChatMsg,
    ReasoningMsg,
    ToolCallMsg,
    ToolResultMsg,
    UserMsg,
)
from minimal_harness.client.built_in.export_tracker import ExportEntry, ExportTracker
from minimal_harness.client.built_in.markdown_styles import (
    LazyMarkdown,
    resolve_code_theme,
)
from minimal_harness.client.built_in.renderer import (
    format_tool_call_static,
    format_tool_result_static,
    truncate_static,
)
from minimal_harness.client.built_in.streaming_controller import StreamingController
from minimal_harness.types import (
    AgentEnd,
    AgentEvent,
    ExecutionStart,
    LLMChunk,
    LLMEnd,
    ToolEnd,
    ToolProgress,
    ToolStart,
)

if TYPE_CHECKING:
    from textual.containers import VerticalScroll


def _format_duration(seconds: float) -> str:
    hours = math.floor(seconds / 3600)
    minutes = math.floor((seconds % 3600) / 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes}m {secs:.0f}s"
    if minutes > 0:
        return f"{minutes}m {secs:.0f}s"
    return f"{secs:.2f}s"


class ChatDisplay:
    """Manages chat area content: messages, streaming, event dispatch, export history."""

    def __init__(
        self,
        chat_container: VerticalScroll,
        theme: str = "",
    ) -> None:
        self._chat = chat_container
        self._theme = theme
        self._msg_counter: int = 0
        self._export = ExportTracker()
        self._streaming = StreamingController(
            chat=self._chat,
            render_markdown=self.render_markdown,
            next_msg_id=self.next_msg_id,
        )

    @property
    def theme(self) -> str:
        return self._theme

    @theme.setter
    def theme(self, value: str) -> None:
        self._theme = value

    @property
    def export_history(self) -> list[ExportEntry]:
        return self._export.history

    @property
    def chat_container(self) -> VerticalScroll:
        return self._chat

    def clear_chat(self) -> None:
        self._export.clear()
        self._chat.query("ChatMsg").remove()
        self._streaming.clear()

    def next_msg_id(self) -> str:
        self._msg_counter += 1
        return f"msg-{self._msg_counter}"

    @property
    def _chat_width(self) -> int:
        w = self._chat.size.width
        return max(w - 4, 20) if w > 0 else 80

    def render_markdown(self, text: str, width: int | None = None) -> LazyMarkdown:
        code_theme = resolve_code_theme(self._theme)
        return LazyMarkdown(text, code_theme=code_theme)

    # -- non-streaming display ------------------------------------------------

    def say(
        self,
        text: str | Text,
        style: str = "",
        is_markdown: bool = False,
        user: bool = False,
    ) -> None:
        mid = self.next_msg_id()
        if isinstance(text, Text):
            w = UserMsg(text, id=mid) if user else ChatMsg(text, id=mid)
            self._export.add(
                ExportEntry(
                    text=text.plain, style=str(text.style) if text.style else None
                )
            )
        elif is_markdown:
            w = AssistantMsg(self.render_markdown(text), id=mid)
            self._export.add(ExportEntry(text=text, is_markdown=True))
        elif style:
            w = (UserMsg if user else ChatMsg)(
                Text(text, style=style, no_wrap=False, overflow="fold"), id=mid
            )
            self._export.add(ExportEntry(text=text, style=style))
        else:
            w = UserMsg(text, id=mid) if user else ChatMsg(text, id=mid)
            self._export.add(ExportEntry(text=text))
        self._chat.mount(w)
        w.scroll_visible()
        self._chat.call_after_refresh(self._chat.scroll_end, animate=False)

    def say_tool_call(self, text: Text) -> None:
        mid = self.next_msg_id()
        w = ToolCallMsg(text, id=mid)
        self._chat.mount(w)
        w.scroll_visible()
        self._chat.call_after_refresh(self._chat.scroll_end, animate=False)
        self._export.add(
            ExportEntry(text=text.plain, style=str(text.style) if text.style else None)
        )

    def say_tool_result(self, text: Text) -> None:
        mid = self.next_msg_id()
        w = ToolResultMsg(text, id=mid)
        self._chat.mount(w)
        w.scroll_visible()
        self._chat.call_after_refresh(self._chat.scroll_end, animate=False)
        self._export.add(
            ExportEntry(text=text.plain, style=str(text.style) if text.style else None)
        )

    def say_reasoning(self, text: str) -> None:
        mid = self.next_msg_id()
        w = ReasoningMsg(text, id=mid)
        self._chat.mount(w)
        w.scroll_visible()
        self._chat.call_after_refresh(self._chat.scroll_end, animate=False)
        self._export.add(ExportEntry(text=text, style="dim"))

    # -- streaming display ----------------------------------------------------

    def tick(self, buf: StreamBuffer, streaming: bool) -> None:
        self._streaming.tick(buf, streaming, self._chat_width)

    def flush(self, buf: StreamBuffer) -> None:
        reasoning, content, tool_calls = self._streaming.flush(buf, self._chat_width)

        width = self._chat_width
        if reasoning:
            mid = self.next_msg_id()
            w = ReasoningMsg(reasoning, id=mid)
            self._chat.mount(w)
            self._export.add(ExportEntry(text=reasoning, style="dim"))
        if content:
            rendered = self.render_markdown(content, width)
            mid = self.next_msg_id()
            w = AssistantMsg(rendered, id=mid)
            self._chat.mount(w)
            self._export.add(ExportEntry(text=content, is_markdown=True))
        if tool_calls:
            for _, call in sorted(tool_calls.items()):
                tw = format_tool_call_static(call)
                tw.no_wrap = False
                tw.overflow = "fold"
                mid = self.next_msg_id()
                w = ToolCallMsg(tw, id=mid)
                self._chat.mount(w)
                self._export.add(
                    ExportEntry(
                        text=tw.plain, style=str(tw.style) if tw.style else None
                    )
                )

    # -- event handling -------------------------------------------------------

    def handle_event(
        self,
        event: AgentEvent,
        buf: StreamBuffer,
    ) -> None:
        if isinstance(event, LLMChunk):
            buf.add_chunk(event.chunk)
        if isinstance(event, LLMEnd):
            if event.reasoning_content:
                buf.reasoning = event.reasoning_content
            if event.content:
                buf.content = event.content
            self.flush(buf)
            if event.usage:
                u = event.usage
                self.say(
                    f"  [{u['prompt_tokens']}+{u['completion_tokens']}={u['total_tokens']} tok]",
                    "dim",
                )
        elif isinstance(event, ExecutionStart):
            names = ", ".join(tc["function"]["name"] for tc in event.tool_calls)
            self.say(f"  \u26a1 Executing: {names}", "bold bright_yellow")
        elif isinstance(event, ToolStart):
            pass
        elif isinstance(event, ToolProgress):
            chunk = event.chunk
            if isinstance(chunk, dict):
                msg = chunk.get("message")
                if msg is None:
                    import json as _json

                    msg = _json.dumps(chunk, ensure_ascii=False, default=str)
            else:
                msg = str(chunk)
            self.say(f"    \u00b7 {truncate_static(msg)}", "dim")
        elif isinstance(event, ToolEnd):
            self.say_tool_result(format_tool_result_static(event.result))
        elif isinstance(event, AgentEnd):
            if event.interrupted:
                self.say("  \u2717 interrupted", "bold bright_red")
            if event.time_taken is not None:
                self.say(f"  \u23f1 {_format_duration(event.time_taken)}", "dim")

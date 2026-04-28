"""Chat display — handles all content rendered in the chat area."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
from minimal_harness.client.built_in.markdown_styles import (
    LazyMarkdown,
    resolve_code_theme,
)
from minimal_harness.client.built_in.renderer import (
    format_tool_call_static,
    format_tool_result_static,
    truncate_static,
)
from minimal_harness.client.events import (
    AgentEndEvent,
    Event,
    ExecutionStartEvent,
    LLMChunkEvent,
    LLMEndEvent,
    ToolEndEvent,
    ToolProgressEvent,
    ToolStartEvent,
)
from minimal_harness.memory import reasoning_message

if TYPE_CHECKING:
    from textual.containers import VerticalScroll


class ChatDisplay:
    """Manages all chat area content: messages, streaming, event dispatch, export history."""

    def __init__(
        self,
        chat_container: VerticalScroll,
        theme: str = "",
    ) -> None:
        self._chat = chat_container
        self._theme = theme
        self._msg_counter: int = 0
        self._export_history: list[tuple[str, str | None, bool]] = []
        self._streaming_reasoning: ReasoningMsg | None = None
        self._streaming_content: AssistantMsg | None = None
        self._streaming_tool_widgets: dict[int, ToolCallMsg] = {}

    @property
    def theme(self) -> str:
        return self._theme

    @theme.setter
    def theme(self, value: str) -> None:
        self._theme = value

    @property
    def export_history(self) -> list[tuple[str, str | None, bool]]:
        return self._export_history

    def clear_chat(self) -> None:
        self._export_history.clear()
        self._chat.query("ChatMsg").remove()

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
            self._export_history.append(
                (text.plain, str(text.style) if text.style else None, False)
            )
        elif is_markdown:
            w = AssistantMsg(self.render_markdown(text), id=mid)
            self._export_history.append((text, None, True))
        elif style:
            w = (UserMsg if user else ChatMsg)(
                Text(text, style=style, no_wrap=False, overflow="fold"), id=mid
            )
            self._export_history.append((text, style, False))
        else:
            w = UserMsg(text, id=mid) if user else ChatMsg(text, id=mid)
            self._export_history.append((text, None, False))
        self._chat.mount(w)
        w.scroll_visible()
        self._chat.call_after_refresh(self._chat.scroll_end, animate=False)

    def say_tool_call(self, text: Text) -> None:
        mid = self.next_msg_id()
        w = ToolCallMsg(text, id=mid)
        self._chat.mount(w)
        w.scroll_visible()
        self._chat.call_after_refresh(self._chat.scroll_end, animate=False)

    def say_tool_result(self, text: Text) -> None:
        mid = self.next_msg_id()
        w = ToolResultMsg(text, id=mid)
        self._chat.mount(w)
        w.scroll_visible()
        self._chat.call_after_refresh(self._chat.scroll_end, animate=False)
        self._export_history.append(
            (text.plain, str(text.style) if text.style else None, False)
        )

    def say_reasoning(self, text: str) -> None:
        mid = self.next_msg_id()
        w = ReasoningMsg(text, id=mid)
        self._chat.mount(w)
        w.scroll_visible()
        self._chat.call_after_refresh(self._chat.scroll_end, animate=False)

    # -- streaming display ----------------------------------------------------

    def tick(self, buf: StreamBuffer, streaming: bool) -> None:
        if not streaming:
            return
        chat = self._chat
        max_scroll = chat.max_scroll_y
        at_bottom = max_scroll == 0 or chat.scroll_y >= max_scroll
        if not at_bottom:
            return
        width = self._chat_width
        if buf.reasoning:
            if self._streaming_reasoning is None:
                self._streaming_reasoning = ReasoningMsg(
                    buf.reasoning, id=self.next_msg_id()
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
                self._streaming_content = AssistantMsg(rendered, id=self.next_msg_id())
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
                    w = ToolCallMsg(tw, id=self.next_msg_id())
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

        width = self._chat_width
        if buf.reasoning:
            mid = self.next_msg_id()
            w = ReasoningMsg(buf.reasoning, id=mid)
            self._chat.mount(w)
            self._export_history.append((buf.reasoning, "dim", False))
        if buf.content:
            rendered = self.render_markdown(buf.content, width)
            mid = self.next_msg_id()
            w = AssistantMsg(rendered, id=mid)
            self._chat.mount(w)
            self._export_history.append((buf.content, None, True))
        if buf.tool_calls:
            for _, call in sorted(buf.tool_calls.items()):
                tw = format_tool_call_static(call)
                tw.no_wrap = False
                tw.overflow = "fold"
                mid = self.next_msg_id()
                w = ToolCallMsg(tw, id=mid)
                self._chat.mount(w)
                self._export_history.append(
                    (tw.plain, str(tw.style) if tw.style else None, False)
                )
            buf.tool_calls.clear()
        if had_content:
            buf._flushed = True
            self._chat.call_after_refresh(self._chat.scroll_end, animate=False)
        buf.reasoning = ""
        buf.content = ""

    # -- event handling -------------------------------------------------------

    def handle_event(
        self,
        event: Event,
        buf: StreamBuffer,
        memory: Any = None,
    ) -> None:
        if isinstance(event, LLMChunkEvent):
            buf.add_chunk(event.chunk)
        elif isinstance(event, LLMEndEvent):
            self.flush(buf)
            if event.reasoning_content and memory is not None:
                memory.add_message(reasoning_message(event.reasoning_content))
            if event.usage:
                u = event.usage
                self.say(
                    f"  [{u['prompt_tokens']}+{u['completion_tokens']}={u['total_tokens']} tok]",
                    "dim",
                )
        elif isinstance(event, ExecutionStartEvent):
            names = ", ".join(tc["function"]["name"] for tc in event.tool_calls)
            self.say(f"  \u26a1 Executing: {names}", "bold bright_yellow")
        elif isinstance(event, ToolStartEvent):
            pass
        elif isinstance(event, ToolProgressEvent):
            chunk = event.chunk
            if isinstance(chunk, dict):
                msg = chunk.get("message")
                if msg is None:
                    import json as _json

                    msg = _json.dumps(chunk, ensure_ascii=False, default=str)
            else:
                msg = str(chunk)
            self.say(f"    \u00b7 {truncate_static(msg)}", "dim")
        elif isinstance(event, ToolEndEvent):
            self.say_tool_result(format_tool_result_static(event.result))
        elif isinstance(event, AgentEndEvent):
            pass

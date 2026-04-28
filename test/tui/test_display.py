from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock

from textual.containers import VerticalScroll

from minimal_harness.client.built_in.buffer import StreamBuffer
from minimal_harness.client.built_in.chat_widgets import (
    AssistantMsg,
    ReasoningMsg,
    ToolCallMsg,
)
from minimal_harness.client.built_in.display import ChatDisplay
from minimal_harness.client.events import (
    ExecutionStartEvent,
    LLMChunkEvent,
    LLMEndEvent,
    ToolEndEvent,
    ToolProgressEvent,
)
from minimal_harness.types import LLMChunkDelta, TokenUsage, ToolCallDelta


def _make_mock_chat() -> MagicMock:
    chat = MagicMock(spec=VerticalScroll)
    type(chat).size = PropertyMock(return_value=MagicMock(width=80))
    chat.max_scroll_y = 0
    chat.scroll_y = 0
    chat.mount.return_value = MagicMock()
    return chat


class TestChatDisplayInit:
    def test_init_sets_defaults(self):
        chat = _make_mock_chat()
        cd = ChatDisplay(chat, theme="nord")
        assert cd._theme == "nord"
        assert cd._msg_counter == 0
        assert cd._export_history == []
        assert cd._streaming_reasoning is None
        assert cd._streaming_content is None
        assert cd._streaming_tool_widgets == {}

    def test_theme_property(self):
        cd = ChatDisplay(_make_mock_chat(), theme="dark")
        assert cd.theme == "dark"
        cd.theme = "light"
        assert cd.theme == "light"

    def test_export_history_property(self):
        cd = ChatDisplay(_make_mock_chat())
        assert cd.export_history == []


class TestChatDisplayNextMsgId:
    def test_increments_counter(self):
        cd = ChatDisplay(_make_mock_chat())
        assert cd.next_msg_id() == "msg-1"
        assert cd.next_msg_id() == "msg-2"
        assert cd._msg_counter == 2


class TestChatDisplaySay:
    def test_say_plain_text(self):
        chat = _make_mock_chat()
        cd = ChatDisplay(chat)
        cd.say("hello")
        assert len(cd._export_history) == 1
        assert cd._export_history[0] == ("hello", None, False)
        chat.mount.assert_called_once()

    def test_say_with_style(self):
        cd = ChatDisplay(_make_mock_chat())
        cd.say("styled text", style="bold red")
        assert cd._export_history[0] == ("styled text", "bold red", False)

    def test_say_as_user(self):
        cd = ChatDisplay(_make_mock_chat())
        cd.say("user input", user=True)
        assert cd._export_history[0] == ("user input", None, False)

    def test_say_markdown(self):
        cd = ChatDisplay(_make_mock_chat())
        cd.say("**bold**", is_markdown=True)
        assert cd._export_history[0] == ("**bold**", None, True)

    def test_say_text_object(self):
        from rich.text import Text

        cd = ChatDisplay(_make_mock_chat())
        t = Text("rich text", style="bold")
        cd.say(t)
        assert cd._export_history[0] == ("rich text", "bold", False)

    def test_clear_chat(self):
        chat = _make_mock_chat()
        cd = ChatDisplay(chat)
        cd.say("hello")
        cd.clear_chat()
        assert cd._export_history == []
        chat.query.assert_called_once_with("ChatMsg")


class TestChatDisplayHandleEvent:
    def test_llm_chunk_event_adds_to_buffer(self):
        cd = ChatDisplay(_make_mock_chat())
        buf = StreamBuffer()
        delta = LLMChunkDelta(content="Hello")
        event = LLMChunkEvent(chunk=delta, is_done=False)
        cd.handle_event(event, buf)
        assert buf.content == "Hello"

    def test_llm_end_event_flushes_buffer(self):
        cd = ChatDisplay(_make_mock_chat())
        buf = StreamBuffer()
        buf.add_chunk(LLMChunkDelta(content="Final answer"))
        event = LLMEndEvent(
            content="Final answer", reasoning_content=None, tool_calls=[], usage=None
        )
        cd.handle_event(event, buf)
        assert buf.content == ""
        assert buf._flushed is True

    def test_llm_end_with_reasoning_adds_to_memory(self):
        cd = ChatDisplay(_make_mock_chat())
        buf = StreamBuffer()
        memory = MagicMock()
        event = LLMEndEvent(
            content="answer",
            reasoning_content="step by step",
            tool_calls=[],
            usage=None,
        )
        cd.handle_event(event, buf, memory=memory)
        memory.add_message.assert_called_once()  # type: ignore[union-attr]

    def test_llm_end_usage_display(self):
        cd = ChatDisplay(_make_mock_chat())
        buf = StreamBuffer()
        usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        event = LLMEndEvent(
            content="answer",
            reasoning_content=None,
            tool_calls=[],
            usage=usage,
        )
        cd.handle_event(event, buf)
        assert any("10+5=15" in item[0] for item in cd._export_history)

    def test_execution_start_event(self):
        cd = ChatDisplay(_make_mock_chat())
        buf = StreamBuffer()
        tool_calls = [{"function": {"name": "get_weather"}}]
        event = ExecutionStartEvent(tool_calls=tool_calls)  # type: ignore[arg-type]
        cd.handle_event(event, buf)
        assert any("Executing:" in item[0] for item in cd._export_history)

    def test_tool_progress_event_with_dict_message(self):
        cd = ChatDisplay(_make_mock_chat())
        buf = StreamBuffer()
        event = ToolProgressEvent(
            tool_call=MagicMock(), chunk={"message": "running..."}
        )
        cd.handle_event(event, buf)
        assert any("running..." in item[0] for item in cd._export_history)

    def test_tool_progress_event_with_raw_dict(self):
        cd = ChatDisplay(_make_mock_chat())
        buf = StreamBuffer()
        event = ToolProgressEvent(tool_call=MagicMock(), chunk={"status": "working"})
        cd.handle_event(event, buf)
        assert any("status" in item[0] for item in cd._export_history)

    def test_tool_end_event(self):
        cd = ChatDisplay(_make_mock_chat())
        buf = StreamBuffer()
        event = ToolEndEvent(tool_call=MagicMock(), result="success")
        cd.handle_event(event, buf)
        assert any("success" in item[0] for item in cd._export_history)


class TestChatDisplayTick:
    def test_tick_does_nothing_when_not_streaming(self):
        chat = _make_mock_chat()
        cd = ChatDisplay(chat)
        cd.tick(StreamBuffer(), streaming=False)
        chat.mount.assert_not_called()

    def test_tick_does_nothing_when_scrolled_up(self):
        chat = _make_mock_chat()
        chat.max_scroll_y = 100
        chat.scroll_y = 50
        cd = ChatDisplay(chat)
        buf = StreamBuffer()
        buf.add_chunk(LLMChunkDelta(content="text"))
        cd.tick(buf, streaming=True)
        chat.mount.assert_not_called()

    def test_tick_renders_reasoning(self):
        cd = ChatDisplay(_make_mock_chat())
        buf = StreamBuffer()
        buf.add_chunk(LLMChunkDelta(reasoning="think..."))
        cd.tick(buf, streaming=True)
        assert cd._streaming_reasoning is not None
        assert isinstance(cd._streaming_reasoning, ReasoningMsg)

    def test_tick_renders_content(self):
        cd = ChatDisplay(_make_mock_chat())
        buf = StreamBuffer()
        buf.add_chunk(LLMChunkDelta(content="answer"))
        cd.tick(buf, streaming=True)
        assert cd._streaming_content is not None
        assert isinstance(cd._streaming_content, AssistantMsg)

    def test_tick_renders_tool_calls(self):
        cd = ChatDisplay(_make_mock_chat())
        buf = StreamBuffer()
        buf.add_chunk(
            LLMChunkDelta(tool_calls=[ToolCallDelta(index=0, id="c1", name="tool")])
        )
        cd.tick(buf, streaming=True)
        assert 0 in cd._streaming_tool_widgets
        assert isinstance(cd._streaming_tool_widgets[0], ToolCallMsg)


class TestChatDisplayFlush:
    def test_flush_removes_streaming_widgets(self):
        cd = ChatDisplay(_make_mock_chat())
        buf = StreamBuffer()
        buf.add_chunk(LLMChunkDelta(content="done", reasoning="think"))
        sr = MagicMock()
        sc = MagicMock()
        st = MagicMock()
        cd._streaming_reasoning = sr
        cd._streaming_content = sc
        cd._streaming_tool_widgets = {0: st}

        cd.flush(buf)
        sr.remove.assert_called_once()
        sc.remove.assert_called_once()
        st.remove.assert_called_once()

    def test_flush_adds_committed_messages(self):
        cd = ChatDisplay(_make_mock_chat())
        buf = StreamBuffer()
        buf.add_chunk(LLMChunkDelta(content="final answer", reasoning="step by step"))
        buf.add_chunk(
            LLMChunkDelta(tool_calls=[ToolCallDelta(index=0, id="c1", name="tool_a")])
        )
        cd.flush(buf)
        assert buf.content == ""
        assert buf.reasoning == ""
        assert buf._flushed is True

    def test_flush_no_op_with_empty_buffer(self):
        cd = ChatDisplay(_make_mock_chat())
        buf = StreamBuffer()
        cd.flush(buf)
        assert buf._flushed is False


class TestChatDisplayRenderMarkdown:
    def test_render_markdown_returns_lazy_markdown(self):
        cd = ChatDisplay(_make_mock_chat(), theme="nord")
        md = cd.render_markdown("# Hello")
        assert md.text == "# Hello"

    def test_chat_width_property(self):
        cd = ChatDisplay(_make_mock_chat())
        assert cd._chat_width == 76

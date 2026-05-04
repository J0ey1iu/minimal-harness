from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock

from textual.containers import VerticalScroll

from minimal_harness.client.built_in.buffer import StreamBuffer
from minimal_harness.client.built_in.display import ChatDisplay, ExportEntry
from minimal_harness.types import (
    AgentEnd,
    ExecutionStart,
    LLMChunk,
    LLMChunkDelta,
    LLMEnd,
    TokenUsage,
    ToolCall,
    ToolCallDelta,
    ToolEnd,
    ToolProgress,
)


def _make_mock_chat() -> MagicMock:
    chat = MagicMock(spec=VerticalScroll)
    type(chat).size = PropertyMock(return_value=MagicMock(width=80))
    chat.max_scroll_y = 0
    chat.scroll_y = 0
    chat.mount.return_value = MagicMock()
    return chat


class TestChatDisplayInit:
    def test_default_state(self):
        cd = ChatDisplay(_make_mock_chat(), theme="nord")
        assert cd._theme == "nord"
        assert cd._msg_counter == 0
        assert cd.export_history == []

    def test_theme_setter_updates(self):
        cd = ChatDisplay(_make_mock_chat(), theme="dark")
        cd.theme = "light"
        assert cd.theme == "light"

    def test_clear_chat_resets_state(self):
        chat = _make_mock_chat()
        cd = ChatDisplay(chat)
        cd.say("hello")
        cd.clear_chat()
        assert cd.export_history == []
        chat.query.assert_called_once_with("ChatMsg")


class TestChatDisplaySay:
    def test_plain_text(self):
        cd = ChatDisplay(_make_mock_chat())
        cd.say("hello")
        assert len(cd.export_history) == 1
        assert cd.export_history[0] == ExportEntry(text="hello")

    def test_with_style(self):
        cd = ChatDisplay(_make_mock_chat())
        cd.say("styled text", style="bold red")
        assert cd.export_history[0] == ExportEntry(text="styled text", style="bold red")

    def test_as_user(self):
        cd = ChatDisplay(_make_mock_chat())
        cd.say("user input", user=True)
        assert cd.export_history[0] == ExportEntry(text="user input")

    def test_markdown(self):
        cd = ChatDisplay(_make_mock_chat())
        cd.say("**bold**", is_markdown=True)
        assert cd.export_history[0] == ExportEntry(text="**bold**", is_markdown=True)

    def test_with_text_object(self):
        from rich.text import Text

        cd = ChatDisplay(_make_mock_chat())
        t = Text("rich text", style="bold")
        cd.say(t)
        assert cd.export_history[0] == ExportEntry(text="rich text", style="bold")


class TestChatDisplayNextMsgId:
    def test_increments(self):
        cd = ChatDisplay(_make_mock_chat())
        assert cd.next_msg_id() == "msg-1"
        assert cd.next_msg_id() == "msg-2"
        assert cd._msg_counter == 2


class TestChatDisplayStreamingLifecycle:
    """Full lifecycle of streaming: chunks -> tick -> flush"""

    def test_tick_does_nothing_when_not_streaming(self):
        chat = _make_mock_chat()
        cd = ChatDisplay(chat)
        cd.tick(StreamBuffer(), streaming=False)
        chat.mount.assert_not_called()

    def test_tick_mounts_reasoning_widget(self):
        chat = _make_mock_chat()
        cd = ChatDisplay(chat)
        buf = StreamBuffer()
        buf.add_chunk(LLMChunkDelta(reasoning="think..."))
        cd.tick(buf, streaming=True)
        chat.mount.assert_called()

    def test_tick_updates_reasoning_in_buffer(self):
        cd = ChatDisplay(_make_mock_chat())
        buf = StreamBuffer()
        buf.add_chunk(LLMChunkDelta(reasoning="step 1"))
        cd.tick(buf, streaming=True)
        buf.add_chunk(LLMChunkDelta(reasoning=" step 2"))
        cd.tick(buf, streaming=True)
        assert buf.reasoning == "step 1 step 2"

    def test_tick_mounts_content_widget(self):
        chat = _make_mock_chat()
        cd = ChatDisplay(chat)
        buf = StreamBuffer()
        buf.add_chunk(LLMChunkDelta(content="answer"))
        cd.tick(buf, streaming=True)
        chat.mount.assert_called()

    def test_tick_mounts_tool_call_widget(self):
        chat = _make_mock_chat()
        cd = ChatDisplay(chat)
        buf = StreamBuffer()
        buf.add_chunk(
            LLMChunkDelta(tool_calls=[ToolCallDelta(index=0, id="c1", name="tool")])
        )
        cd.tick(buf, streaming=True)
        chat.mount.assert_called()

    def test_tick_does_nothing_when_scrolled_up(self):
        chat = _make_mock_chat()
        chat.max_scroll_y = 100
        chat.scroll_y = 50
        cd = ChatDisplay(chat)
        buf = StreamBuffer()
        buf.add_chunk(LLMChunkDelta(content="text"))
        cd.tick(buf, streaming=True)
        chat.mount.assert_not_called()

    def test_llm_chunk_adds_to_buffer(self):
        cd = ChatDisplay(_make_mock_chat())
        buf = StreamBuffer()
        delta = LLMChunkDelta(content="Hello")
        event = LLMChunk(chunk=delta)
        cd.handle_event(event, buf)
        assert buf.content == "Hello"

    def test_llm_end_flushes_buffer(self):
        cd = ChatDisplay(_make_mock_chat())
        buf = StreamBuffer()
        buf.add_chunk(LLMChunkDelta(content="Final answer"))
        usage = TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        event = LLMEnd(
            content="Final answer", reasoning_content=None, tool_calls=[], usage=usage
        )
        cd.handle_event(event, buf)
        assert buf.content == ""

    def test_llm_end_with_reasoning_flushes(self):
        cd = ChatDisplay(_make_mock_chat())
        buf = StreamBuffer()
        buf.add_chunk(LLMChunkDelta(content="answer", reasoning="step by step"))
        usage = TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        event = LLMEnd(
            content="answer",
            reasoning_content="step by step",
            tool_calls=[],
            usage=usage,
        )
        cd.handle_event(event, buf)
        assert buf.content == ""
        assert buf.reasoning == ""

    def test_llm_end_displays_usage(self):
        cd = ChatDisplay(_make_mock_chat())
        buf = StreamBuffer()
        usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        event = LLMEnd(
            content="answer", reasoning_content=None, tool_calls=[], usage=usage
        )
        cd.handle_event(event, buf)
        assert any("10+5=15" in item.text for item in cd.export_history)

    def test_flush_adds_to_export_history(self):
        cd = ChatDisplay(_make_mock_chat())
        buf = StreamBuffer()
        buf.add_chunk(LLMChunkDelta(content="final answer", reasoning="step by step"))
        cd.flush(buf)
        assert buf.content == ""
        assert buf.reasoning == ""
        texts = [e.text for e in cd.export_history]
        assert "step by step" in texts
        assert "final answer" in texts

    def test_flush_no_op_with_empty_buffer(self):
        cd = ChatDisplay(_make_mock_chat())
        buf = StreamBuffer()
        cd.flush(buf)
        assert buf._flushed is False

    def test_full_lifecycle_buffer_state(self):
        """chunks -> tick -> flush: verify buffer state at each phase"""
        cd = ChatDisplay(_make_mock_chat())
        buf = StreamBuffer()

        buf.add_chunk(LLMChunkDelta(content="Hello"))
        buf.add_chunk(LLMChunkDelta(content=" world"))

        cd.tick(buf, streaming=True)
        assert buf.content == "Hello world"

        cd.flush(buf)
        assert buf.content == ""
        assert any("Hello world" in e.text for e in cd.export_history)


class TestChatDisplayHandleEvent:
    def test_execution_start(self):
        cd = ChatDisplay(_make_mock_chat())
        buf = StreamBuffer()
        tool_calls: list[ToolCall] = [
            {
                "id": "1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": ""},
            }
        ]
        event = ExecutionStart(tool_calls=tool_calls)
        cd.handle_event(event, buf)
        assert any("Executing:" in item.text for item in cd.export_history)

    def test_tool_progress_with_message(self):
        cd = ChatDisplay(_make_mock_chat())
        buf = StreamBuffer()
        event = ToolProgress(tool_call=MagicMock(), chunk={"message": "running..."})
        cd.handle_event(event, buf)
        assert any("running..." in item.text for item in cd.export_history)

    def test_tool_progress_with_raw_dict(self):
        cd = ChatDisplay(_make_mock_chat())
        buf = StreamBuffer()
        event = ToolProgress(tool_call=MagicMock(), chunk={"status": "working"})
        cd.handle_event(event, buf)
        assert any("status" in item.text for item in cd.export_history)

    def test_tool_end_displays_result(self):
        cd = ChatDisplay(_make_mock_chat())
        buf = StreamBuffer()
        event = ToolEnd(tool_call=MagicMock(), result="success")
        cd.handle_event(event, buf)
        assert any("success" in item.text for item in cd.export_history)

    def test_agent_end_is_ignored(self):
        cd = ChatDisplay(_make_mock_chat())
        buf = StreamBuffer()
        before = len(cd.export_history)
        cd.handle_event(AgentEnd(response="done"), buf)
        assert len(cd.export_history) == before


class TestChatDisplayExportHistory:
    def test_say_appends(self):
        cd = ChatDisplay(_make_mock_chat())
        cd.say("line1")
        cd.say("line2", style="bold")
        cd.say("**md**", is_markdown=True)
        assert len(cd.export_history) == 3

    def test_say_tool_call_appends(self):
        from rich.text import Text

        cd = ChatDisplay(_make_mock_chat())
        cd.say_tool_call(Text("tool_call"))
        assert len(cd.export_history) == 1
        assert cd.export_history[0].text == "tool_call"

    def test_say_tool_result_appends(self):
        from rich.text import Text

        cd = ChatDisplay(_make_mock_chat())
        cd.say_tool_result(Text("result"))
        assert len(cd.export_history) == 1

    def test_say_reasoning_appends(self):
        cd = ChatDisplay(_make_mock_chat())
        cd.say_reasoning("thinking...")
        assert len(cd.export_history) == 1
        assert cd.export_history[0].text == "thinking..."

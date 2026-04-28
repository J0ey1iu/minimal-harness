from __future__ import annotations

from textual.widgets import Static

from minimal_harness.client.built_in.chat_widgets import (
    AssistantMsg,
    ChatMsg,
    ReasoningMsg,
    ToolCallMsg,
    ToolResultMsg,
    UserMsg,
)


class TestChatMsg:
    def test_base_class(self):
        assert issubclass(ChatMsg, Static)

    def test_init_with_string(self):
        msg = ChatMsg("hello world")
        assert isinstance(msg, ChatMsg)

    def test_init_with_text(self):
        from rich.text import Text

        t = Text("styled text", style="bold")
        msg = ChatMsg(t)
        assert isinstance(msg, ChatMsg)

    def test_init_with_id(self):
        msg = ChatMsg("test", id="custom-id")
        assert msg.id == "custom-id"

    def test_empty_construction(self):
        msg = ChatMsg()
        assert isinstance(msg, ChatMsg)


class TestChatMsgSubclasses:
    def test_user_msg(self):
        msg = UserMsg("user input")
        assert isinstance(msg, ChatMsg)

    def test_reasoning_msg(self):
        msg = ReasoningMsg("thinking...")
        assert isinstance(msg, ChatMsg)

    def test_tool_call_msg(self):
        msg = ToolCallMsg("tool call")
        assert isinstance(msg, ChatMsg)

    def test_tool_result_msg(self):
        msg = ToolResultMsg("tool result")
        assert isinstance(msg, ChatMsg)

    def test_assistant_msg(self):
        msg = AssistantMsg("assistant answer")
        assert isinstance(msg, ChatMsg)

    def test_subclass_construction(self):
        for cls in [UserMsg, ReasoningMsg, ToolCallMsg, ToolResultMsg, AssistantMsg]:
            msg = cls("test")
            assert isinstance(msg, ChatMsg)

    def test_subclass_with_text_object(self):
        from rich.text import Text

        t = Text("rich text")
        msg = UserMsg(t)
        assert isinstance(msg, UserMsg)

    def test_accepts_text_with_style(self):
        from rich.text import Text

        t = Text("styled", style="bold italic")
        msg = ToolCallMsg(t)
        assert isinstance(msg, ToolCallMsg)

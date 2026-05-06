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

    def test_construct_with_string(self):
        msg = ChatMsg("hello world")
        assert isinstance(msg, ChatMsg)

    def test_construct_with_text(self):
        from rich.text import Text

        msg = ChatMsg(Text("styled text", style="bold"))
        assert isinstance(msg, ChatMsg)

    def test_construct_with_id(self):
        msg = ChatMsg("test", id="custom-id")
        assert msg.id == "custom-id"

    def test_empty_construct(self):
        msg = ChatMsg()
        assert isinstance(msg, ChatMsg)


class TestChatMsgSubclasses:
    def test_all_subclasses(self):
        for cls in [UserMsg, ReasoningMsg, ToolCallMsg, ToolResultMsg, AssistantMsg]:
            msg = cls("test")
            assert isinstance(msg, ChatMsg)

    def test_with_text_object(self):
        from rich.text import Text

        assert isinstance(UserMsg(Text("rich text")), UserMsg)

    def test_with_styled_text(self):
        from rich.text import Text

        assert isinstance(ToolCallMsg(Text("styled", style="bold italic")), ToolCallMsg)

    def test_class_hierarchy(self):
        assert isinstance(UserMsg("x"), UserMsg)
        assert isinstance(ReasoningMsg("x"), ReasoningMsg)
        assert isinstance(ToolCallMsg("x"), ToolCallMsg)
        assert isinstance(ToolResultMsg("x"), ToolResultMsg)
        assert isinstance(AssistantMsg("x"), AssistantMsg)

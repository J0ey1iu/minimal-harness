"""Session management for the TUI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from rich.text import Text

from minimal_harness.client.built_in.context import AppContext
from minimal_harness.client.built_in.display import ChatDisplay
from minimal_harness.client.built_in.renderer import (
    format_tool_call_static,
    format_tool_result_static,
)

if TYPE_CHECKING:
    from minimal_harness.agent import AgentRuntime, ConversationSession
    from minimal_harness.memory import Memory


class SessionManager:
    def __init__(
        self,
        runtime: "AgentRuntime",
        ctx: AppContext,
        display: ChatDisplay,
        clear_input: Callable[[], None],
        show_banner: Callable[[], None],
    ) -> None:
        self._runtime = runtime
        self._ctx = ctx
        self._display = display
        self._clear_input = clear_input
        self._show_banner = show_banner

    def replay_session(
        self,
        session: "ConversationSession",
        clear_committed: Callable[[], None],
        clear_buf: Callable[[], None],
    ) -> tuple[bool, list[str]]:
        try:
            memory = session.memory
            title = session.name or "Untitled"
            self._display.say(f"\u2713 Session resumed: {title}", "bold #a6e3a1")
            clear_committed()
            clear_buf()
            self._clear_input()
            self._show_banner()
            self._replay_memory(memory)
            self._display.chat_container.call_after_refresh(
                self._display.chat_container.scroll_end,
                animate=False,
            )
            user_inputs = self._extract_user_inputs(memory)
            return True, user_inputs
        except Exception as e:
            self._display.say(f"\u2717 {e}", "bold #f38ba8")
            return False, []

    @staticmethod
    def _extract_user_inputs(memory: Memory) -> list[str]:
        inputs: list[str] = []
        for msg in memory.get_all_messages():
            if msg.get("role") == "user":
                parts = msg.get("content")
                if isinstance(parts, list):
                    texts = [
                        p.get("text", "")
                        for p in parts
                        if isinstance(p, dict) and p.get("type") == "text"
                    ]
                    text = " ".join(texts)
                    if text:
                        inputs.append(text)
        return inputs

    def _replay_memory(self, memory: Memory) -> None:
        messages = memory.get_all_messages()

        for msg in messages:
            role = msg.get("role")
            if role == "system":
                continue
            if role == "user":
                parts = msg.get("content")
                if not isinstance(parts, list):
                    continue
                texts = []
                for part in parts:
                    if isinstance(part, dict) and part.get("type") == "text":
                        texts.append(part.get("text", ""))
                text = " ".join(texts)
                if text:
                    self._display.say(text, user=True)
            elif role == "assistant":
                content = msg.get("content")
                if isinstance(content, str) and content:
                    self._display.say(content, "", True)
                tcs = msg.get("tool_calls")
                if isinstance(tcs, list):
                    for tc in tcs:
                        if not isinstance(tc, dict):
                            continue
                        text = format_tool_call_static(tc.get("function", {}))
                        self._display.say_tool_call(text)
                self._display.say("")
            elif role == "reasoning":
                content = msg.get("content")
                if isinstance(content, str) and content:
                    self._display.say_reasoning(content)
            elif role == "tool":
                content = msg.get("content")
                if not isinstance(content, str):
                    continue
                if content.startswith(("[Tool Error]", "[Tool Execution Stopped]")):
                    text = Text(f"  \u2717 {content}", style="bold bright_red")
                else:
                    text = format_tool_result_static(content)
                self._display.say_tool_result(text)
                self._display.say("")

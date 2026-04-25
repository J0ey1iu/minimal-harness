"""Session management for the TUI."""

from __future__ import annotations

from typing import Callable, Protocol

from rich.text import Text

from minimal_harness.client.built_in.context import AppContext
from minimal_harness.client.built_in.memory import PersistentMemory

from .renderer import format_tool_call_static, format_tool_result_static


class SayCallback(Protocol):
    def __call__(
        self, text: str | Text, style: str = "", is_markdown: bool = False
    ) -> None: ...


class SessionManager:
    def __init__(
        self,
        ctx: AppContext,
        say: SayCallback,
        scroll_end: Callable[[bool], None] | None = None,
        clear_rlog: Callable[[], None] | None = None,
        clear_input: Callable[[], None] | None = None,
        set_input_history: Callable[[list[str]], None] | None = None,
        banner: Callable[[], None] | None = None,
    ) -> None:
        self._ctx = ctx
        self._say = say
        self._scroll_end = scroll_end
        self._clear_rlog = clear_rlog
        self._clear_input = clear_input
        self._set_input_history = set_input_history
        self._banner = banner

    def load_session(
        self,
        session_id: str,
        clear_committed: Callable[[], None],
        clear_buf: Callable[[], None],
    ) -> bool:
        try:
            memory = PersistentMemory.from_session(session_id)
            title = memory._title or "Untitled"
            self._say(f"✓ Session resumed: {title}", "bold #a6e3a1")
            clear_committed()
            clear_buf()
            if self._clear_rlog:
                self._clear_rlog()
            if self._clear_input:
                self._clear_input()
            if self._banner:
                self._banner()
            self._ctx.memory = memory
            self._replay_memory(memory)
            if self._scroll_end:
                self._scroll_end(False)
            if self._set_input_history:
                inputs = self._extract_user_inputs(memory)
                self._set_input_history(inputs)
            return True
        except Exception as e:
            self._say(f"✗ {e}", "bold #f38ba8")
            return False

    def _extract_user_inputs(self, memory: PersistentMemory) -> list[str]:
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

    def _replay_memory(self, memory: PersistentMemory) -> None:
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
                    self._say(f"\n❯ {text}", "bold #89b4fa")
                    self._say("")
            elif role == "assistant":
                content = msg.get("content")
                if isinstance(content, str) and content:
                    self._say(content, "", True)
                tcs = msg.get("tool_calls")
                if isinstance(tcs, list):
                    for tc in tcs:
                        if not isinstance(tc, dict):
                            continue
                        self._say(format_tool_call_static(tc.get("function", {})))
                self._say("")
                self._say("")
            elif role == "tool":
                content = msg.get("content")
                if not isinstance(content, str):
                    continue
                if content.startswith(("[Tool Error]", "[Tool Execution Stopped]")):
                    self._say(f"    ✗ {content}", "bold #f38ba8")
                else:
                    self._say(format_tool_result_static(content))
                self._say("")
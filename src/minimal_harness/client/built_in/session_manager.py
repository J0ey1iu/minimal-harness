"""Session management for the TUI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Protocol

from rich.text import Text

from minimal_harness.client.built_in.context import AppContext
from minimal_harness.client.built_in.memory import PersistentMemory

if TYPE_CHECKING:
    pass

from .renderer import format_tool_call_static, format_tool_result_static


class SayCallback(Protocol):
    def __call__(
        self,
        text: str | Text,
        style: str = "",
        is_markdown: bool = False,
        user: bool = False,
    ) -> None: ...


class TUIAppInterface(Protocol):
    def say(
        self,
        text: str | Text,
        style: str = "",
        is_markdown: bool = False,
        user: bool = False,
    ) -> None: ...
    def _say_tool_call(self, text: Text) -> None: ...
    def _say_tool_result(self, text: Text) -> None: ...
    @property
    def _chat(self) -> object: ...
    def _next_msg_id(self) -> str: ...
    @property
    def _input(self) -> object: ...
    def _banner(self) -> None: ...


class SessionManager:
    def __init__(
        self,
        ctx: AppContext,
        app: TUIAppInterface,
    ) -> None:
        self._ctx = ctx
        self._app = app

    def load_session(
        self,
        session_id: str,
        clear_committed: "Callable[[], None]",
        clear_buf: "Callable[[], None]",
    ) -> bool:
        try:
            memory = PersistentMemory.from_session(session_id)
            title = memory.title or "Untitled"
            self._app.say(f"✓ Session resumed: {title}", "bold #a6e3a1")
            clear_committed()
            clear_buf()
            self._app._input.text = ""  # type: ignore[attr-defined]
            self._app._banner()
            self._ctx.memory = memory
            self._replay_memory(memory)
            self._app._chat.scroll_end(animate=False)  # type: ignore[attr-defined]
            self._app._input._input_history = self._extract_user_inputs(memory)  # type: ignore[attr-defined]
            self._app._input.reset_history_index()  # type: ignore[attr-defined]
            return True
        except Exception as e:
            self._app.say(f"✗ {e}", "bold #f38ba8")
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
                    self._app.say(text, user=True)
                    self._app.say("")
            elif role == "assistant":
                content = msg.get("content")
                if isinstance(content, str) and content:
                    self._app.say(content, "", True)
                tcs = msg.get("tool_calls")
                if isinstance(tcs, list):
                    for tc in tcs:
                        if not isinstance(tc, dict):
                            continue
                        text = format_tool_call_static(tc.get("function", {}))
                        self._app._say_tool_call(text)
                self._app.say("")
            elif role == "tool":
                content = msg.get("content")
                if not isinstance(content, str):
                    continue
                if content.startswith(("[Tool Error]", "[Tool Execution Stopped]")):
                    text = Text(f"  ✗ {content}", style="bold bright_red")
                else:
                    text = format_tool_result_static(content)
                self._app._say_tool_result(text)
                self._app.say("")

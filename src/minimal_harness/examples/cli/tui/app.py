from __future__ import annotations

import platform
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionChunk

from openai import AsyncOpenAI
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Header, Input, Label, Static, TextArea

from minimal_harness.agent import OpenAIAgent
from minimal_harness.llm import ToolCall
from minimal_harness.llm.openai import OpenAILLMProvider
from minimal_harness.memory import (
    ConversationMemory,
    ExtendedInputContentPart,
)

from ..widgets import (
    ChatMessage,
    ThinkingWidget,
    ToolCallWidget,
    ToolResultWidget,
)
from .styles import CSS
from .thinking import extract_thinking
from .tool import built_in_tools


class ChatTUI(App):
    CSS = CSS
    TITLE = "minimal-harness"

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True, priority=True),
        Binding("ctrl+m", "toggle_model_input", "Model", show=True),
        Binding("ctrl+i", "edit_system_prompt", "Prompt", show=True),
    ]

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "qwen3.5-27b",
        system_prompt: str | None = None,
    ):
        super().__init__()
        self._base_url = base_url
        self._api_key = api_key
        self._model = model
        self._system_prompt = system_prompt or "You are a helpful assistant."
        self._system_prompt += (
            f"\n\nYour are working on a {platform.system()} machine"
            f" and your current working directory is `{Path.cwd()}`"
        )

        self._tools = built_in_tools

        self._init_agent()
        self._is_thinking = False
        self._model_input_visible = False

    def _init_agent(self) -> None:
        client = AsyncOpenAI(base_url=self._base_url, api_key=self._api_key or None)
        self._llm_provider = OpenAILLMProvider(client=client, model=self._model)
        self._memory = ConversationMemory(system_prompt=self._system_prompt)
        self._agent = OpenAIAgent(
            llm_provider=self._llm_provider,
            tools=self._tools,
            memory=self._memory,
        )

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="app-grid"):
            with Container(id="chat-container"):
                with VerticalScroll(id="history"):
                    yield Static(
                        "╭──────────────────────────────────────╮\n"
                        "│      minimal-harness by J0ey1iu      │\n"
                        "╰──────────────────────────────────────╯",
                        classes="welcome-title",
                    )
                    yield Static(
                        "Ctrl+M  change model  ·  Ctrl+I  edit prompt  ·  Ctrl+C  quit",
                        classes="welcome-subtitle",
                    )
            with Vertical(id="bottom-bar"):
                with Horizontal(id="model-bar"):
                    yield Label("Model:")
                    yield Input(
                        value=self._model,
                        placeholder="e.g. qwen3.5-27b",
                        id="model-input",
                    )
                with Container(id="input-wrapper"):
                    yield Input(placeholder="Send a message…", id="input")
                with Horizontal(id="status-bar"):
                    yield Label("", id="status-left")
                    yield Label("", id="status-right")
        yield Footer()

    def on_mount(self) -> None:
        self._update_status("Ready")
        self.query_one("#input", Input).focus()

    def action_quit(self) -> None:  # type: ignore[override]
        self.exit()

    def action_toggle_model_input(self) -> None:
        model_bar = self.query_one("#model-bar", Horizontal)
        self._model_input_visible = not self._model_input_visible
        model_bar.styles.display = "block" if self._model_input_visible else "none"
        if self._model_input_visible:
            model_input = self.query_one("#model-input", Input)
            model_input.value = self._model
            model_input.focus()
        else:
            self.query_one("#input", Input).focus()

    @on(Input.Submitted, "#model-input")
    async def on_model_submitted(self, event: Input.Submitted) -> None:
        new_model = event.value.strip()
        if not new_model:
            return

        event.stop()

        old_model = self._model
        self._model = new_model

        self._init_agent()

        self._model_input_visible = False
        self.query_one("#model-bar", Horizontal).styles.display = "none"
        self.query_one("#input", Input).focus()

        history = self.query_one("#history", VerticalScroll)
        await history.mount(
            Static(
                f"── model changed: {old_model} → {new_model} ──",
                classes="system-notice",
            )
        )
        history.scroll_end()
        self._update_status("Ready")

    @on(Input.Submitted, "#input")
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        user_input = event.value.strip()
        if not user_input:
            return

        if self._is_thinking:
            return

        event.input.value = ""
        self._update_status("Thinking…")
        self._is_thinking = True

        history = self.query_one("#history", VerticalScroll)
        await history.mount(ChatMessage(f"You\n{user_input}", classes="user-message"))
        history.scroll_end()

        await self._process_message(user_input)

        self._is_thinking = False
        self._update_status("Ready")

    async def _process_message(self, user_input: str) -> None:
        history = self.query_one("#history", VerticalScroll)

        state: dict[str, Any] = {
            "thinking_text": "",
            "thinking_widget": None,
            "thinking_finalized": False,
            "streaming_text": "",
            "response_widget": None,
            "tool_calls_detected": [],
            "tool_call_widgets": {},
        }

        async def _ensure_thinking_widget() -> ThinkingWidget:
            if state["thinking_widget"] is None:
                tw = ThinkingWidget("Thinking…", classes="thinking")
                await history.mount(tw)
                state["thinking_widget"] = tw
                state["thinking_text"] = ""
                state["thinking_finalized"] = False
            return state["thinking_widget"]

        async def _finish_thinking() -> None:
            tw = state["thinking_widget"]
            if tw is not None and not state["thinking_finalized"]:
                if state["thinking_text"]:
                    tw.update(f"Thinking\n{state['thinking_text']}")
                else:
                    tw.remove()
                state["thinking_finalized"] = True
                state["thinking_widget"] = None
                state["thinking_text"] = ""

        async def _ensure_response_widget() -> ChatMessage:
            if state["response_widget"] is None:
                await _finish_thinking()
                rw = ChatMessage("Assistant\n", classes="assistant-message")
                await history.mount(rw)
                state["response_widget"] = rw
            return state["response_widget"]

        async def on_chunk(chunk: ChatCompletionChunk | None, is_done: bool) -> None:
            if is_done:
                await _finish_thinking()
                return

            if not chunk:
                return

            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                return

            reasoning = extract_thinking(delta)
            if reasoning:
                tw = await _ensure_thinking_widget()
                state["thinking_text"] += reasoning
                tw.update(f"Thinking\n{state['thinking_text']}")
                history.scroll_end()

            if delta.tool_calls:
                await _finish_thinking()
                for tc in delta.tool_calls:
                    detected: list[dict[str, Any]] = state["tool_calls_detected"]
                    tc_widgets: dict[int, ToolCallWidget] = state["tool_call_widgets"]

                    existing = next(
                        (t for t in detected if t["index"] == tc.index),
                        None,
                    )
                if not existing:
                    existing = {
                        "index": tc.index,
                        "id": tc.id or "",
                        "name": tc.function.name if tc.function else "",
                        "arguments": tc.function.arguments
                        if tc.function and tc.function.arguments
                        else "",
                    }
                    detected.append(existing)
                    widget = ToolCallWidget(
                        f"⚙  {existing['name']}()",
                        classes="tool-call",
                    )
                    await history.mount(widget)
                    tc_widgets[tc.index] = widget
                else:
                    # Only accumulate on subsequent chunks
                    if tc.id:
                        existing["id"] += tc.id
                    if tc.function:
                        if tc.function.name:
                            existing["name"] += tc.function.name
                        if tc.function.arguments:
                            existing["arguments"] += tc.function.arguments

                if tc.index in tc_widgets:
                    tc_widgets[tc.index].update(
                        f"⚙  {existing['name']}({existing['arguments']})"
                    )
                history.scroll_end()

            if delta.content:
                state["streaming_text"] += delta.content
                w = await _ensure_response_widget()
                w.update(f"Assistant\n{state['streaming_text']}")
                history.scroll_end()

        async def on_tool_end(tool_call: ToolCall, result: Any) -> None:
            state["response_widget"] = None
            state["streaming_text"] = ""
            state["tool_calls_detected"] = []
            state["tool_call_widgets"] = {}

            result_widget = ToolResultWidget(f"✓  {result}", classes="tool-result")
            await history.mount(result_widget)
            history.scroll_end()

        try:
            self._llm_provider._on_chunk = on_chunk
            result = await self._agent.run(
                user_input=cast(
                    list[ExtendedInputContentPart],
                    [{"type": "text", "text": user_input}],
                ),
                on_tool_end=on_tool_end,
            )

            final_text = state["streaming_text"] or result or ""
            if final_text:
                w = await _ensure_response_widget()
                w.update(f"Assistant\n{final_text}")
            elif state["response_widget"] is None:
                await _finish_thinking()
                w = await _ensure_response_widget()
                w.update("Assistant\n(No response)")

        except Exception as e:
            await _finish_thinking()
            error_widget = ChatMessage(
                f"Error\n{e!s}",
                classes="assistant-message",
            )
            await history.mount(error_widget)

        history.scroll_end()

    def _update_status(self, status: str) -> None:
        usage = self._memory.get_total_usage()
        tokens = usage.get("total_tokens", 0)
        self.query_one("#status-left", Label).update(f"● {status}")
        self.query_one("#status-right", Label).update(
            f"{self._model}  ·  {tokens:,} tokens"
        )

    def action_edit_system_prompt(self) -> None:
        self.push_screen(SystemPromptScreen(self._system_prompt, self._on_prompt_save))

    def _on_prompt_save(self, new_prompt: str) -> None:
        self._system_prompt = new_prompt
        messages = self._memory.get_all_messages()
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = new_prompt  # type: ignore[typeddict-item]
        self._update_status("Ready")


class SystemPromptScreen(Screen):
    def __init__(self, current_prompt: str, on_save: Callable[[str], None]):
        super().__init__()
        self._current_prompt = current_prompt
        self._on_save = on_save

    def compose(self) -> ComposeResult:
        with Container(id="prompt-modal"):
            yield Static("Edit System Prompt", classes="modal-title")
            yield TextArea(
                self._current_prompt,
                id="prompt-editor",
                classes="prompt-editor",
            )
            with Horizontal(id="modal-buttons"):
                yield Input(value="Ctrl+Enter to save", id="save-hint", disabled=True)
                yield Static("Esc to cancel", classes="modal-hint")

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            self.app.pop_screen()
        elif event.key == "ctrl+enter":
            new_prompt = self.query_one("#prompt-editor", TextArea).text
            self._on_save(new_prompt)
            self.app.pop_screen()

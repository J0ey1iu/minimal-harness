from __future__ import annotations

import platform
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    pass

from openai import AsyncOpenAI
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import Footer, Header, Input, Label, Static

from minimal_harness.agent import OpenAIAgent
from minimal_harness.examples.cli.tui.handlers import (
    StreamingState,
    create_thinking_handler,
    create_tool_end_handler,
)
from minimal_harness.examples.cli.tui.screens import SystemPromptScreen
from minimal_harness.examples.cli.tui.styles import CSS
from minimal_harness.examples.cli.tui.tool import built_in_tools
from minimal_harness.examples.cli.widgets import (
    ChatMessage,
)
from minimal_harness.llm.openai import OpenAILLMProvider
from minimal_harness.memory import (
    ConversationMemory,
    ExtendedInputContentPart,
)


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

    def action_quit(self) -> None:
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
        await history.mount(ChatMessage(f"{user_input}", classes="user-message"))
        history.scroll_end()

        await self._process_message(user_input)

        self._is_thinking = False
        self._update_status("Ready")

    async def _process_message(self, user_input: str) -> None:
        history = self.query_one("#history", VerticalScroll)
        state = StreamingState()

        on_chunk = create_thinking_handler(history, state)
        on_tool_end = create_tool_end_handler(history, state)

        try:
            self._llm_provider._on_chunk = on_chunk
            result = await self._agent.run(
                user_input=cast(
                    list[ExtendedInputContentPart],
                    [{"type": "text", "text": user_input}],
                ),
                on_tool_end=on_tool_end,
            )

            if state.streaming_text:
                w = await _ensure_response_widget(history, state)
                w.update(f"{state.streaming_text}")
            elif result:
                w = await _ensure_response_widget(history, state)
                w.update(f"{result}")
            elif state.response_widget is None:
                await _finish_thinking(history, state)
                w = await _ensure_response_widget(history, state)
                w.update("(No response)")

        except Exception as e:
            await _finish_thinking(history, state)
            error_widget = ChatMessage(
                f"Error\n\n{e!s}",
                classes="assistant-message",
            )
            await history.mount(error_widget)

        history.scroll_end()

    def _update_status(self, status: str) -> None:
        usage = self._memory.get_total_usage()
        tokens = usage.get("total_tokens", 0)
        status_left = self.query_one("#status-left", Label)
        status_left.update(f"● {status}")
        if status == "Ready":
            status_left.set_class(False, "status-primary")
        else:
            status_left.set_class(True, "status-primary")
        self.query_one("#status-right", Label).update(
            f"{self._model}  ·  {tokens:,} tokens"
        )

    def action_edit_system_prompt(self) -> None:
        self.push_screen(SystemPromptScreen(self._system_prompt, self._on_prompt_save))

    def _on_prompt_save(self, new_prompt: str) -> None:
        self._system_prompt = new_prompt
        messages = self._memory.get_all_messages()
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = new_prompt
        self._update_status("Ready")


async def _ensure_response_widget(
    history: VerticalScroll,
    state: StreamingState,
) -> ChatMessage:
    if state.response_widget is None:
        await _finish_thinking(history, state)
        rw = ChatMessage("", classes="assistant-message")
        await history.mount(rw)
        state.response_widget = rw
    return state.response_widget


async def _finish_thinking(
    history: VerticalScroll,
    state: StreamingState,
) -> None:
    tw = state.thinking_widget
    if tw is not None and not state.thinking_finalized:
        if state.thinking_text:
            tw.update(f"Thinking\n{state.thinking_text}")
        else:
            tw.remove()
        state.thinking_finalized = True
        state.thinking_widget = None
        state.thinking_text = ""

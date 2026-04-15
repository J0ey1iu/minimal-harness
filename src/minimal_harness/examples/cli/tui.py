from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionChunk

from openai import AsyncOpenAI

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import Input, Label

from minimal_harness import Tool
from minimal_harness.agent import OpenAIAgent
from minimal_harness.llm.openai import OpenAILLMProvider
from minimal_harness.memory import (
    ConversationMemory,
    ExtendedInputContentPart,
)

from .tools import calculator, get_weather
from .widgets import (
    ChatMessage,
    ThinkingWidget,
    ToolCallWidget,
    ToolResultWidget,
)

CSS = """
Screen {
    background: #0d1117;
    color: #c9d1d9;
}

#chat-container {
    height: 1fr;
    background: #010409;
    border: solid #30363d;
}

#history {
    height: 1fr;
    padding: 1 2;
    overflow-y: scroll;
}

#input-container {
    height: auto;
    background: #161b22;
    border-top: solid #30363d;
    padding: 1 2;
}

#input {
    width: 100%;
    background: #0d1117;
    color: #c9d1d9;
    border: solid #30363d;
}

#input:focus {
    border: solid #58a6ff;
}

#model-bar {
    height: auto;
    padding: 0 2;
    background: #161b22;
    border-bottom: solid #30363d;
    display: none;
}

#model-label {
    width: auto;
    padding: 0 1 0 0;
    color: #8b949e;
}

#model-input {
    width: 1fr;
}

#status {
    width: 100%;
    height: auto;
    padding: 0 2;
    color: #8b949e;
}

.user-message {
    background: #1f2937;
    color: #e5e7eb;
    padding: 1 2;
    margin: 1 0 0 4;
    border: solid #3b82f6;
}

.assistant-message {
    background: #161b22;
    color: #c9d1d9;
    padding: 1 2;
    margin: 1 4 0 0;
    border: solid #30363d;
}

.tool-call {
    background: #1c2128;
    color: #d29922;
    padding: 1 2;
    margin: 0 2;
    border-left: solid #d29922;
}

.tool-result {
    background: #1c2128;
    color: #3fb950;
    padding: 1 2;
    margin: 0 2;
    border-left: solid #3fb950;
}

.thinking {
    background: #1c2128;
    color: #8b949e;
    padding: 1 2;
    margin: 0 2;
    border-left: solid #6e7681;
}

.welcome {
    color: #8b949e;
    text-align: center;
    padding: 2;
}

.header {
    background: #161b22;
    color: #c9d1d9;
    padding: 0 2;
    height: auto;
}

#header-label {
    width: 100%;
    padding: 0;
}
"""


def _extract_thinking(delta: Any) -> str | None:
    reasoning = getattr(delta, "reasoning_content", None)
    if reasoning:
        return reasoning

    reasoning = getattr(delta, "reasoning", None)
    if reasoning:
        return reasoning

    provider_fields = getattr(delta, "provider_specific_fields", None)
    if provider_fields and isinstance(provider_fields, dict):
        reasoning = provider_fields.get("reasoning_content")
        if reasoning:
            return reasoning

    model_extra = getattr(delta, "model_extra", None)
    if model_extra and isinstance(model_extra, dict):
        reasoning = model_extra.get("reasoning_content")
        if reasoning:
            return reasoning
        reasoning = model_extra.get("reasoning")
        if reasoning:
            return reasoning

    return None


class ChatTUI(App):
    CSS = CSS

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True, priority=True),
        Binding("ctrl+m", "toggle_model_input", "Change Model", show=True),
    ]

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "openai/gpt-4o-mini",
        system_prompt: str | None = None,
    ):
        super().__init__()
        self._base_url = base_url
        self._api_key = api_key
        self._model = model
        self._system_prompt = system_prompt or "You are a helpful assistant."

        self._tools = [
            Tool(
                name="get_weather",
                description="Get weather for a specified city",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                    },
                    "required": ["city"],
                },
                fn=get_weather,
            ),
            Tool(
                name="calculator",
                description="Calculate mathematical expression",
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Valid Python mathematical expression",
                        },
                    },
                    "required": ["expression"],
                },
                fn=calculator,
            ),
        ]

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
        with Container(id="chat-container"):
            with VerticalScroll(id="history"):
                yield ChatMessage(
                    "minimal-harness CLI\n"
                    "Press Ctrl+M to change model, Ctrl+C to quit.",
                    classes="welcome",
                )
            with Horizontal(id="model-bar"):
                yield Label("Model:", id="model-label")
                yield Input(
                    value=self._model,
                    placeholder="e.g. openai/gpt-4o-mini",
                    id="model-input",
                )
            with Container(id="input-container"):
                yield Input(placeholder="Type your message...", id="input")
                yield Label("", id="status")

    def on_mount(self) -> None:
        self._update_status("Ready")

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
            ChatMessage(
                f"✦ Model changed: {old_model} → {new_model}",
                classes="welcome",
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
        self._update_status("Thinking...")
        self._is_thinking = True

        history = self.query_one("#history", VerticalScroll)
        await history.mount(ChatMessage(f"⟫ You\n{user_input}", classes="user-message"))
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
                tw = ThinkingWidget("💭 Thinking...", classes="thinking")
                await history.mount(tw)
                state["thinking_widget"] = tw
                state["thinking_text"] = ""
                state["thinking_finalized"] = False
            return state["thinking_widget"]

        async def _finish_thinking() -> None:
            tw = state["thinking_widget"]
            if tw is not None and not state["thinking_finalized"]:
                if state["thinking_text"]:
                    tw.update(f"💭 Thinking\n{state['thinking_text']}")
                else:
                    tw.remove()
                state["thinking_finalized"] = True
                state["thinking_widget"] = None
                state["thinking_text"] = ""

        async def _ensure_response_widget() -> ChatMessage:
            if state["response_widget"] is None:
                await _finish_thinking()
                rw = ChatMessage("⟪ Assistant\n", classes="assistant-message")
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

            reasoning = _extract_thinking(delta)
            if reasoning:
                tw = await _ensure_thinking_widget()
                state["thinking_text"] += reasoning
                tw.update(f"💭 Thinking\n{state['thinking_text']}")
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
                            "arguments": "",
                        }
                        detected.append(existing)
                        widget = ToolCallWidget(
                            f"⚙ Calling: {existing['name']}( )",
                            classes="tool-call",
                        )
                        await history.mount(widget)
                        tc_widgets[tc.index] = widget

                    if tc.id:
                        existing["id"] += tc.id
                    if tc.function:
                        if tc.function.name:
                            existing["name"] += tc.function.name
                        if tc.function.arguments:
                            existing["arguments"] += tc.function.arguments

                    if tc.index in tc_widgets:
                        tc_widgets[tc.index].update(
                            f"⚙ Calling: {existing['name']}({existing['arguments']})"
                        )
                history.scroll_end()

            if delta.content:
                state["streaming_text"] += delta.content
                w = await _ensure_response_widget()
                w.update(f"⟪ Assistant\n{state['streaming_text']}")
                history.scroll_end()

        async def on_tool_end(tool_call: dict[str, Any], result: Any) -> None:
            state["response_widget"] = None
            state["streaming_text"] = ""
            state["tool_calls_detected"] = []
            state["tool_call_widgets"] = {}

            result_widget = ToolResultWidget(
                f"✓ Result: {result}", classes="tool-result"
            )
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
                w.update(f"⟪ Assistant\n{final_text}")
            elif state["response_widget"] is None:
                await _finish_thinking()
                w = await _ensure_response_widget()
                w.update("⟪ Assistant\n(No response)")

        except Exception as e:
            await _finish_thinking()
            error_widget = ChatMessage(
                f"⟪ Assistant\n❌ Error: {e!s}",
                classes="assistant-message",
            )
            await history.mount(error_widget)

        history.scroll_end()

    def _update_status(self, status: str) -> None:
        usage = self._memory.get_total_usage()
        tokens = usage.get("total_tokens", 0)
        status_widget = self.query_one("#status", Label)
        status_widget.update(f"{status} | Model: {self._model} | Tokens: {tokens}")

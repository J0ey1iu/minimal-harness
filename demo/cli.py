import os
import asyncio
import json
from datetime import datetime

from textual.app import App, ComposeResult
from textual.widgets import Header, Input, Static
from textual.containers import Vertical, ScrollableContainer
from textual.binding import Binding
from textual import events
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax

from minimal_harness import OpenAIAgent, OpenAILLMProvider, ConversationMemory
from minimal_harness.tool.glob import get_tools as glob_get_tools
from minimal_harness.tool.grep import get_tools as grep_get_tools
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk


async def get_weather(city: str) -> dict:
    """Simulate weather query"""
    await asyncio.sleep(0.2)
    return {"city": city, "temperature": "22°C", "condition": "Sunny"}


async def calculator(expression: str) -> dict:
    """Simple calculator"""
    result = eval(expression, {"__builtins__": {}})
    return {"expression": expression, "result": result}


class MessageWidget(Static):
    """Custom widget for displaying individual messages"""

    def __init__(
        self, message_type: str, content: str, timestamp: str = None, **kwargs
    ):
        self.message_type = message_type
        self.message_content = content  # Store content separately
        self.timestamp = timestamp or datetime.now().strftime("%H:%M")
        super().__init__(**kwargs)  # Initialize Static first
        self.update_content()  # Then update with our content

    def update_content(self):
        if self.message_type == "user":
            panel = Panel(
                self.message_content,
                title=f"👤 You · {self.timestamp}",
                title_align="left",
                border_style="blue",
                padding=(0, 1),
            )
        elif self.message_type == "assistant":
            panel = Panel(
                Markdown(self.message_content)
                if self.message_content.strip()
                else "...",
                title=f"🤖 Assistant · {self.timestamp}",
                title_align="left",
                border_style="green",
                padding=(0, 1),
            )
        elif self.message_type == "tool":
            try:
                # Try to format as JSON if possible
                parsed = json.loads(self.message_content)
                syntax = Syntax(
                    json.dumps(parsed, indent=2),
                    "json",
                    theme="monokai",
                    line_numbers=False,
                )
                panel = Panel(
                    syntax,
                    title=f"🔧 Tool Call · {self.timestamp}",
                    title_align="left",
                    border_style="yellow",
                    padding=(0, 1),
                )
            except:
                panel = Panel(
                    self.message_content,
                    title=f"🔧 Tool Call · {self.timestamp}",
                    title_align="left",
                    border_style="yellow",
                    padding=(0, 1),
                )
        elif self.message_type == "error":
            panel = Panel(
                self.message_content,
                title=f"❌ Error · {self.timestamp}",
                title_align="left",
                border_style="red",
                padding=(0, 1),
            )
        elif self.message_type == "thinking":
            panel = Panel(
                Markdown(self.message_content),
                title=f"💭 Thinking · {self.timestamp}",
                title_align="left",
                border_style="cyan",
                padding=(0, 1),
            )
        else:
            panel = Panel(self.message_content, border_style="white", padding=(0, 1))

        self.update(panel)


class StreamingWidget(Static):
    """Widget for live streaming responses"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.text_content = ""
        self.thinking_content = ""
        self.tool_calls = {}
        self.is_visible = False

    def start_streaming(self):
        self.is_visible = True
        self.text_content = ""
        self.thinking_content = ""
        self.tool_calls = {}
        self.update_display()

    def stop_streaming(self):
        self.is_visible = False
        self.update("")

    def add_thinking(self, text: str):
        self.thinking_content += text
        self.update_display()

    def add_content(self, text: str):
        self.text_content += text
        self.update_display()

    def update_tool_calls(self, tool_calls: dict):
        self.tool_calls = tool_calls
        self.update_display()

    def update_display(self):
        if not self.is_visible:
            return

        display_content = ""

        # Show tool calls
        if self.tool_calls:
            tool_parts = []
            for idx in sorted(self.tool_calls.keys()):
                name, args, _ = self.tool_calls[idx]
                if name:
                    try:
                        formatted_args = (
                            json.dumps(json.loads(args), indent=2) if args else ""
                        )
                    except:
                        formatted_args = args
                    tool_parts.append(f"🔧 **{name}**\n```json\n{formatted_args}\n```")

            if tool_parts:
                display_content = "\n\n".join(tool_parts)
                if self.thinking_content or self.text_content:
                    display_content += "\n\n---\n\n"

        # Show thinking/thought process
        if self.thinking_content:
            display_content += f"**Thinking:**\n{self.thinking_content}\n\n---\n\n"

        # Show streaming content
        if self.text_content:
            display_content += self.text_content

        # Add cursor
        display_content += " ▋"

        timestamp = datetime.now().strftime("%H:%M")
        panel = Panel(
            Markdown(display_content) if display_content.strip() else "Thinking...",
            title=f"🤖 Assistant · {timestamp}",
            title_align="left",
            border_style="green dim",
            padding=(0, 1),
        )

        self.update(panel)


class CLIApp(App):
    CSS = """
    Screen {
        layout: vertical;
        background: #0d1117;
        color: #f0f6fc;
    }

    Header {
        dock: top;
        height: 3;
        background: #161b22;
        border-bottom: solid #30363d;
    }

    #chat_area {
        height: 1fr;
        overflow: hidden;
    }

    #messages_container {
        height: 1fr;
        overflow-y: auto;
        padding: 1 2;
    }

    #streaming_area {
        height: auto;
        max-height: 8;
        padding: 0 2;
    }

    #input_area {
        height: auto;
        min-height: 4;
        background: #161b22;
        border-top: solid #30363d;
        padding: 1 2;
    }

    #status {
        text-align: center;
        color: #7d8590;
        height: 1;
        content-align: center middle;
    }

    #usage_display {
        text-align: center;
        color: #7d8590;
        height: auto;
        content-align: center middle;
        padding: 0 1;
    }

    #user_input {
        height: 3;
        border: solid #30363d;
        background: #0d1117;
        color: #f0f6fc;
        margin: 0;
        padding: 0 1;
    }

    #user_input:focus {
        border: solid #1f6feb;
        background: #0d1117;
    }

    MessageWidget {
        margin: 0 0 1 0;
        height: auto;
    }

    StreamingWidget {
        height: auto;
        margin: 0 0 1 0;
    }

    .welcome_message {
        margin: 0 0 1 0;
        height: auto;
    }

    Footer {
        dock: bottom;
    }
    """

    BINDINGS = [
        Binding("ctrl-c", "quit", "Quit"),
        Binding("ctrl-l", "clear", "Clear"),
        Binding("escape", "focus_input", "Focus Input"),
    ]

    def __init__(self):
        super().__init__()
        self._setup_agent()
        self.messages = []
        self.is_processing = False

    def _setup_agent(self):
        from minimal_harness.tool.base import Tool

        glob_tools = glob_get_tools()
        grep_tools = grep_get_tools()

        tools = (
            [
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
            + list(glob_tools.values())
            + list(grep_tools.values())
        )

        client = AsyncOpenAI(
            api_key=os.getenv("AIHUBMIX_API_KEY"),
            base_url="https://aihubmix.com/v1",
        )
        llm_provider = OpenAILLMProvider(client=client, model="qwen3.5-27b")
        self._memory = ConversationMemory(
            system_prompt="You are a helpful assistant that can check weather and do calculations. Respond in a friendly and informative manner."
        )
        self._agent = OpenAIAgent(
            llm_provider=llm_provider,
            tools=tools,
            memory=self._memory,
        )

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        # Main chat area that takes most of the space
        with Vertical(id="chat_area"):
            with ScrollableContainer(id="messages_container"):
                yield Static(
                    Panel(
                        "👋 Welcome! I'm your AI assistant.\n\nI can help you with:\n• Weather information\n• Mathematical calculations\n• File searching (glob)\n• Content searching (grep)\n• General questions\n\nType your message below and press Enter to start!",
                        title="🤖 Assistant",
                        border_style="green",
                        padding=(1, 2),
                    ),
                    classes="welcome_message",
                )

            # Streaming area (only visible when streaming)
            with Vertical(id="streaming_area"):
                yield StreamingWidget(id="streaming")

        # Input area at the bottom
        with Vertical(id="input_area"):
            yield Static(
                "💬 Type your message and press Enter • Ctrl+L to clear • Ctrl+C to quit",
                id="status",
            )
            yield Static(
                "Tokens: 0 (prompt: 0, completion: 0)",
                id="usage_display",
            )
            yield Input(placeholder="Ask me anything...", id="user_input")

    def on_mount(self) -> None:
        self.query_one("#user_input").focus()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if self.is_processing:
            return

        user_input = event.value.strip()
        if not user_input:
            return

        self.is_processing = True
        input_widget = self.query_one("#user_input")
        input_widget.disabled = True

        # Add user message
        await self.add_message("user", user_input)

        # Start streaming
        streaming_widget = self.query_one("#streaming")
        streaming_widget.start_streaming()

        # Clear input
        input_widget.value = ""

        # Process response
        await self.process_agent_response(user_input, streaming_widget)

        # Cleanup
        streaming_widget.stop_streaming()
        input_widget.disabled = False
        input_widget.focus()
        self.is_processing = False

    async def add_message(self, message_type: str, content: str):
        """Add a message to the conversation"""
        container = self.query_one("#messages_container")
        message = MessageWidget(message_type, content)
        await container.mount(message)
        container.scroll_end(animate=True)

    async def process_agent_response(
        self, user_input: str, streaming_widget: StreamingWidget
    ):
        assistant_response = ""
        thinking_content = ""
        tool_calls_acc = {}
        collected_tool_calls = []

        async def on_tool_result(tc, result):
            name = tc["function"]["name"]
            args = tc["function"]["arguments"]
            try:
                formatted_args = json.dumps(json.loads(args), indent=2) if args else ""
            except:
                formatted_args = args
            tool_content = f"**{name}**\n```json\n{formatted_args}\n```\n\n**Result:**\n```json\n{json.dumps(result, indent=2)}\n```"
            await self.add_message("tool", tool_content)

        async def on_chunk(chunk: ChatCompletionChunk | None, is_done: bool):
            nonlocal assistant_response, thinking_content

            if is_done:
                # Add thinking content if any
                if thinking_content.strip():
                    await self.add_message("thinking", thinking_content.strip())
                # Add final assistant response
                if assistant_response.strip():
                    await self.add_message("assistant", assistant_response.strip())
                return

            if chunk is None:
                return

            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                return

            # Handle tool calls
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if tc.function:
                        name = tc.function.name or ""
                        args = tc.function.arguments or ""

                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = ("", "", "")

                        acc_name, acc_args, _ = tool_calls_acc[idx]
                        acc_name += name
                        acc_args += args
                        tool_calls_acc[idx] = (acc_name, acc_args, "")

                streaming_widget.update_tool_calls(tool_calls_acc)

                # Store completed tool calls
                for idx, (name, args, _) in tool_calls_acc.items():
                    if name and args:
                        try:
                            json.loads(args)  # Validate JSON
                            formatted_args = json.dumps(json.loads(args), indent=2)
                            if (name, formatted_args, "") not in collected_tool_calls:
                                collected_tool_calls.append((name, formatted_args, ""))
                        except:
                            pass

                await asyncio.sleep(0.01)
                return

            # Handle reasoning content (thought process)
            # Check multiple possible field names for reasoning/thinking
            reasoning = (
                getattr(delta, "reasoning_content", None)
                or getattr(delta, "reasoning", None)
                or getattr(delta, "thought", None)
            )
            if reasoning:
                thinking_content += reasoning
                streaming_widget.add_thinking(reasoning)
                await asyncio.sleep(0.01)
                return

            # Handle final content (actual answer)
            if delta.content:
                assistant_response += delta.content
                streaming_widget.add_content(delta.content)
                await asyncio.sleep(0.01)

        try:
            await self._agent.run(
                user_input, on_chunk=on_chunk, on_tool_result=on_tool_result
            )
        except Exception as e:
            await self.add_message("error", f"An error occurred: {str(e)}")

        await self._update_usage_display_async()

    def action_clear(self) -> None:
        """Clear the conversation"""
        container = self.query_one("#messages_container")
        for child in list(container.children):
            if not hasattr(child, "classes") or "welcome_message" not in str(
                child.classes
            ):
                child.remove()

    def action_focus_input(self) -> None:
        """Focus the input field"""
        self.query_one("#user_input").focus()

    async def _update_usage_display_async(self) -> None:
        """Update the usage display with current token counts"""
        usage = self._memory.get_total_usage()
        total = usage["total_tokens"]
        prompt = usage["prompt_tokens"]
        completion = usage["completion_tokens"]
        widget = self.query_one("#usage_display")
        widget.update(f"Tokens: {total} (prompt: {prompt}, completion: {completion})")

    def on_key(self, event: events.Key) -> None:
        """Handle key events"""
        if event.key == "escape":
            self.query_one("#user_input").focus()


if __name__ == "__main__":
    app = CLIApp()
    app.run()

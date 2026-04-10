import os
import asyncio
import json

from textual.app import App, ComposeResult
from textual.widgets import Header, Input, RichLog, Static
from textual.containers import VerticalScroll
from textual.binding import Binding

from minimal_harness import Tool, OpenAIAgent, OpenAILLMProvider, ConversationMemory
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


class CLIApp(App):
    CSS = """
    Screen {
        background: #1e1e2e;
    }

    #conversation {
        height: 1fr;
        padding: 1;
    }

    #chat_log {
        height: 1fr;
    }

    #streaming_response {
        height: auto;
        dock: bottom;
        padding: 0 1;
    }

    #input-area {
        height: auto;
        padding: 1;
        background: #2e2e3e;
        border-top: solid #4a4a5a;
    }

    Input {
        margin: 1;
    }
    """

    BINDINGS = [
        Binding("ctrl-c", "quit", "Quit", show=True),
    ]

    def __init__(self):
        super().__init__()
        self._setup_agent()

    def _setup_agent(self):
        tools = [
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

        client = AsyncOpenAI(
            api_key=os.getenv("AIHUBMIX_API_KEY"),
            base_url="https://aihubmix.com/v1",
        )
        llm_provider = OpenAILLMProvider(client=client, model="qwen3.5-27b")
        memory = ConversationMemory(
            system_prompt="You are an assistant that can check weather and do calculations."
        )
        self._agent = OpenAIAgent(
            llm_provider=llm_provider,
            tools=tools,
            memory=memory,
        )

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="conversation"):
            yield RichLog(id="chat_log", markup=True, auto_scroll=True)
            yield Static(id="streaming_response", markup=True)
        with VerticalScroll(id="input-area"):
            yield Input(
                placeholder="Type your message and press Enter...", id="user_input"
            )

    def on_mount(self) -> None:
        self.query_one("#chat_log").write("Welcome! How can I help you today?")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        user_input = event.value.strip()
        if not user_input:
            return

        input_widget = self.query_one("#user_input")
        input_widget.disabled = True

        chat_log = self.query_one("#chat_log")
        streaming = self.query_one("#streaming_response")

        chat_log.write(f"[bold blue]You:[/bold blue] {user_input}")

        assistant_response = ""
        tool_calls_acc: dict[int, tuple[str, str]] = {}
        is_final_done = False

        def format_tool_calls_display():
            parts = []
            for idx in sorted(tool_calls_acc.keys()):
                name, args_str = tool_calls_acc[idx]
                parts.append(
                    f"[yellow]Calling tool: {name}[/yellow]\n[yellow dim]{args_str}[/yellow dim]"
                )
            return "\n\n".join(parts)

        async def on_chunk(chunk: ChatCompletionChunk | None, is_done: bool):
            nonlocal assistant_response, is_final_done

            if is_done:
                is_final_done = True
                streaming.update("")

                for idx in sorted(tool_calls_acc.keys()):
                    name, args_str = tool_calls_acc[idx]
                    chat_log.write(f"[yellow]Calling tool: {name}[/yellow]")
                    for line in args_str.split("\n"):
                        chat_log.write(f"[yellow dim]  {line}[/yellow dim]")
                    chat_log.write("")

                if assistant_response:
                    lines = assistant_response.split("\n")
                    for line in lines:
                        chat_log.write(f"[bold green]Assistant:[/bold green] {line}")
                chat_log.write("")
                return

            delta = chunk.choices[0].delta if chunk.choices else None

            if delta is None:
                return

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if tc.function:
                        name = tc.function.name or ""
                        args = tc.function.arguments or ""

                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = ("", "")

                        acc_name, acc_args = tool_calls_acc[idx]
                        acc_name += name
                        acc_args += args

                        try:
                            args_dict = json.loads(acc_args) if acc_args else {}
                            args_str = json.dumps(
                                args_dict, indent=2, ensure_ascii=False
                            )
                        except json.JSONDecodeError:
                            args_str = acc_args

                        tool_calls_acc[idx] = (acc_name, args_str)

                streaming.update(
                    format_tool_calls_display() + "\n\n[dim]▌[/dim]"
                    if format_tool_calls_display()
                    else ""
                )
                return

            if delta.content:
                assistant_response += delta.content
                display = (
                    format_tool_calls_display() + "\n\n"
                    if format_tool_calls_display()
                    else ""
                )
                display += f"[bold green]{assistant_response}[/bold green][dim]▌[/dim]"
                streaming.update(display)

        try:
            await self._agent.run(user_input, on_chunk=on_chunk)

            if not is_final_done:
                streaming.update("")
        except Exception as e:
            streaming.update("")
            chat_log.write(f"[red]Error:[/red] {str(e)}")

        input_widget.value = ""
        input_widget.disabled = False
        input_widget.focus()


if __name__ == "__main__":
    app = CLIApp()
    app.run()

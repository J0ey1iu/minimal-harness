import os
import asyncio
import json

from textual.app import App, ComposeResult
from textual.widgets import Header, Input, RichLog, Static
from textual.containers import VerticalScroll
from textual.binding import Binding

from minimal_harness import Tool, Agent, OpenAILLMProvider, ConversationMemory
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
        self._agent = Agent(
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
        tool_calls_info: list[tuple[str, str]] = []
        in_tool_call = False
        is_final_done = False

        async def on_chunk(chunk: ChatCompletionChunk | None, is_done: bool):
            nonlocal assistant_response, in_tool_call, is_final_done

            if is_done:
                is_final_done = True
                streaming.update("")
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
                    if tc.function:
                        name = tc.function.name or ""
                        args = tc.function.arguments or ""
                        try:
                            args_dict = json.loads(args) if args else {}
                            args_str = json.dumps(
                                args_dict, indent=2, ensure_ascii=False
                            )
                        except json.JSONDecodeError:
                            args_str = args
                        tool_calls_info.append((name, args_str))
                        in_tool_call = True
                return

            if delta.content:
                in_tool_call = False
                assistant_response += delta.content
                streaming.update(f"{assistant_response}[dim]▌[/dim]")

        try:
            await self._agent.run(user_input, on_chunk=on_chunk)

            if not is_final_done:
                streaming.update("")
        except Exception as e:
            streaming.update("")
            chat_log.write(f"[red]Error:[/red] {str(e)}")

        for name, args_str in tool_calls_info:
            chat_log.write(f"[yellow]Calling tool: {name}[/yellow]")
            for line in args_str.split("\n"):
                chat_log.write(f"[yellow dim]  {line}[/yellow dim]")
            chat_log.write("")

        input_widget.value = ""
        input_widget.disabled = False
        input_widget.focus()


if __name__ == "__main__":
    app = CLIApp()
    app.run()

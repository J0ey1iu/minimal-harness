"""Interactive CLI application using FrameworkClient with proper event visualization."""

import asyncio
import json
import os
from collections import OrderedDict
from collections.abc import Iterable

from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style as PTStyle
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

from minimal_harness import StreamingTool
from minimal_harness.agent import OpenAIAgent
from minimal_harness.client import FrameworkClient
from minimal_harness.client.events import (
    AgentEndEvent,
    AgentStartEvent,
    ChunkEvent,
    ExecutionStartEvent,
    LLMEndEvent,
    LLMStartEvent,
    ToolEndEvent,
    ToolProgressEvent,
    ToolStartEvent,
)
from minimal_harness.llm.openai import OpenAILLMProvider
from minimal_harness.memory import ConversationMemory
from minimal_harness.tool.built_in.bash import bash_tool
from minimal_harness.tool.built_in.create_file import create_file_tool
from minimal_harness.tool.built_in.delete_file import delete_file_tool
from minimal_harness.tool.built_in.patch_file import patch_file_tool
from minimal_harness.tool.built_in.read_file import read_file_tool

console = Console()


def get_client():
    from openai import AsyncOpenAI

    api_key = os.getenv("MH_API_KEY")
    base_url = os.getenv("MH_BASE_URL")
    model = os.getenv("MH_MODEL", "qwen3.5-27b")

    kwargs: dict = {"base_url": base_url}
    if api_key:
        kwargs["api_key"] = api_key
    client = AsyncOpenAI(**kwargs)  # type: ignore[arg-type]
    llm_provider = OpenAILLMProvider(client=client, model=model)
    memory = ConversationMemory(system_prompt="You are a helpful assistant.")
    agent = OpenAIAgent(
        llm_provider=llm_provider,
        tools=[*BUILTIN_TOOLS, calculator_tool],
        memory=memory,
    )
    return FrameworkClient(agent=agent)


BUILTIN_TOOLS = [
    bash_tool,
    create_file_tool,
    delete_file_tool,
    patch_file_tool,
    read_file_tool,
]


async def calculator_handler(expression: str):
    yield {"status": "progress", "message": f"Calculating: {expression}"}
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        yield {"success": True, "expression": expression, "result": result}
    except Exception as e:
        yield {"success": False, "error": str(e)}


calculator_tool = StreamingTool(
    name="calculator",
    description="Evaluate a mathematical expression and return the result.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate",
            },
        },
        "required": ["expression"],
    },
    fn=calculator_handler,
)


def _extract_text(user_input) -> str:
    if isinstance(user_input, str):
        return user_input
    if isinstance(user_input, Iterable):
        parts = []
        for item in user_input:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts) if parts else str(user_input)
    return str(user_input)


class ToolCallAccumulator:
    def __init__(self):
        self._calls: OrderedDict[int, dict] = OrderedDict()

    def update(self, tc_deltas) -> None:
        for tc_delta in tc_deltas:
            idx = tc_delta.index
            if idx not in self._calls:
                self._calls[idx] = {
                    "id": "",
                    "name": "",
                    "arguments": "",
                }
            entry = self._calls[idx]
            if tc_delta.id:
                entry["id"] += tc_delta.id
            if tc_delta.function:
                if tc_delta.function.name:
                    entry["name"] += tc_delta.function.name
                if tc_delta.function.arguments:
                    entry["arguments"] += tc_delta.function.arguments

    @property
    def calls(self) -> list[dict]:
        return list(self._calls.values())

    def reset(self) -> None:
        self._calls.clear()


class PhaseRenderer:
    def __init__(self):
        self.reasoning_chunks: list[str] = []
        self.content_chunks: list[str] = []
        self.tool_call_acc = ToolCallAccumulator()
        self.in_reasoning = False
        self.in_content = False
        self.in_tool_call = False
        self._live: Live | None = None
        self._phase_count = 0

    def reset(self):
        self.reasoning_chunks = []
        self.content_chunks = []
        self.tool_call_acc = ToolCallAccumulator()
        self.in_reasoning = False
        self.in_content = False
        self.in_tool_call = False
        self._phase_count = 0

    def start(self):
        self._live = Live(
            console=console, vertical_overflow="visible", refresh_per_second=15
        )
        self._live.__enter__()

    def stop(self):
        if self._live:
            self._live.__exit__(None, None, None)
            self._live = None

    def _build_display(self) -> Text:
        parts = Text()

        if self.reasoning_chunks:
            reasoning_text = "".join(self.reasoning_chunks)
            if parts:
                parts.append("\n")
            parts.append("💭 Thinking\n", style="bold dim cyan")
            parts.append(reasoning_text, style="dim")
            parts.append("\n")

        if self.content_chunks:
            content_text = "".join(self.content_chunks)
            if parts:
                parts.append("\n")
            if not self.in_tool_call:
                parts.append("💬 Response\n", style="bold green")
            parts.append(content_text)
            if not self.in_tool_call:
                parts.append("\n")

        if self.tool_call_acc.calls:
            if parts:
                parts.append("\n")
            parts.append("🔧 Tool Calls\n", style="bold yellow")
            for call in self.tool_call_acc.calls:
                name = call["name"]
                args_str = call["arguments"]
                try:
                    args_parsed = json.loads(args_str)
                    args_display = json.dumps(args_parsed, ensure_ascii=False)
                except (json.JSONDecodeError, ValueError):
                    args_display = args_str
                parts.append(f"  {name}(", style="yellow")
                parts.append(args_display, style="dim yellow")
                parts.append(")\n", style="yellow")

        return parts

    def _refresh(self):
        if self._live:
            self._live.update(self._build_display())

    def add_reasoning_chunk(self, text: str):
        if not self.in_reasoning:
            self.in_reasoning = True
            self.in_content = False
        self.reasoning_chunks.append(text)
        self._refresh()

    def add_content_chunk(self, text: str):
        if not self.in_content:
            if self.in_reasoning:
                self.in_reasoning = False
                self.reasoning_chunks.append("\n")
            self.in_content = True
        self.content_chunks.append(text)
        self._refresh()

    def add_tool_call_delta(self, tc_deltas):
        if not self.in_tool_call:
            if self.in_content:
                self.in_content = False
            if self.in_reasoning:
                self.in_reasoning = False
            self.in_tool_call = True
        self.tool_call_acc.update(tc_deltas)
        self._refresh()

    def finalize_reasoning(self):
        self.in_reasoning = False

    def finalize_content(self):
        self.in_content = False

    def finalize_tool_calls(self):
        self.in_tool_call = False


def extract_chunk_delta(chunk):
    if not hasattr(chunk, "choices") or not chunk.choices:
        return None, None, None
    delta = chunk.choices[0].delta
    reasoning = getattr(delta, "reasoning_content", None)
    content = delta.content
    tool_calls = delta.tool_calls
    return reasoning, content, tool_calls


async def run_interactive(client: FrameworkClient):
    session = PromptSession()
    style = PTStyle.from_dict({"prompt": "cyan bold"})

    console.print(
        Panel(
            "[bold green]Minimal Harness CLI[/bold green]\n"
            "Type your message and press Enter.\n"
            "Type 'exit' or 'quit' to stop.",
            border_style="green",
        )
    )

    while True:
        try:
            user_input = await session.prompt_async("You: ", style=style)
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.strip().lower() in ("exit", "quit"):
            break
        if not user_input.strip():
            continue

        renderer = PhaseRenderer()
        renderer.start()

        try:
            async for event in client.run(
                user_input=[{"type": "text", "text": user_input}]
            ):
                if isinstance(event, AgentStartEvent):
                    renderer.reset()

                elif isinstance(event, ChunkEvent):
                    if event.chunk is not None and not event.is_done:
                        reasoning, content, tool_calls = extract_chunk_delta(
                            event.chunk
                        )
                        if reasoning:
                            renderer.add_reasoning_chunk(reasoning)
                        if content:
                            renderer.add_content_chunk(content)
                        if tool_calls:
                            renderer.add_tool_call_delta(tool_calls)

                elif isinstance(event, LLMStartEvent):
                    pass

                elif isinstance(event, LLMEndEvent):
                    renderer.finalize_reasoning()
                    renderer.finalize_content()
                    renderer.finalize_tool_calls()

                    if event.content and not renderer.content_chunks:
                        renderer.add_content_chunk(event.content)

                elif isinstance(event, ExecutionStartEvent):
                    renderer.stop()
                    tool_names = ", ".join(
                        tc.get("function", {}).get("name", "unknown")
                        for tc in event.tool_calls
                    )
                    console.print(
                        Panel(
                            escape(f"Executing: {tool_names}"),
                            style="bold magenta",
                            border_style="magenta",
                        )
                    )

                elif isinstance(event, ToolStartEvent):
                    tool_name = event.tool_call.get("function", {}).get(
                        "name", "unknown"
                    )
                    args = event.tool_call.get("function", {}).get("arguments", "{}")
                    try:
                        args_parsed = json.loads(args)
                        args_display = json.dumps(args_parsed, ensure_ascii=False)
                    except (json.JSONDecodeError, ValueError):
                        args_display = args
                    console.print(
                        f"  [bold cyan]▸ {escape(tool_name)}[/bold cyan]",
                        Text(f"({args_display})", style="dim yellow"),
                    )

                elif isinstance(event, ToolProgressEvent):
                    chunk_data = event.chunk
                    if isinstance(chunk_data, dict):
                        msg = chunk_data.get(
                            "message", chunk_data.get("status", str(chunk_data))
                        )
                        console.print("    [dim]⏳[/dim]", escape(str(msg)))

                elif isinstance(event, ToolEndEvent):
                    tool_name = event.tool_call.get("function", {}).get(
                        "name", "unknown"
                    )
                    console.print(
                        f"  [bold green]✓ {escape(tool_name)}[/bold green]",
                        Text(" → "),
                        Text(escape(str(event.result))),
                    )

                elif isinstance(event, AgentEndEvent):
                    renderer.stop()
                    console.print()
                    console.print(
                        Panel(
                            Markdown(event.response),
                            title="Assistant",
                            border_style="blue",
                        )
                    )

        except Exception as e:
            renderer.stop()
            console.print("\n[bold red]Error:[/bold red]", escape(str(e)))

        console.print()


async def main():
    client = get_client()
    await run_interactive(client)


if __name__ == "__main__":
    asyncio.run(main())

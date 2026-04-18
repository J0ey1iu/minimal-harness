"""Interactive CLI application using FrameworkClient with proper event visualization."""

import asyncio
import json
import os
from collections import OrderedDict

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

    @property
    def empty(self) -> bool:
        return len(self._calls) == 0

    def reset(self) -> None:
        self._calls.clear()


def _fmt_args(args_str: str) -> str:
    try:
        parsed = json.loads(args_str)
        return json.dumps(parsed, ensure_ascii=False)
    except (json.JSONDecodeError, ValueError):
        return args_str


class LLMRoundSegment:
    """One LLM call: may contain reasoning, content, and/or tool calls."""

    def __init__(self):
        self.reasoning: str = ""
        self.content: str = ""
        self.tool_call_acc = ToolCallAccumulator()

    def render(self) -> Text:
        parts = Text()
        if self.reasoning:
            parts.append("💭 Thinking\n", style="bold dim cyan")
            parts.append(self.reasoning, style="dim cyan")
            parts.append("\n")
        if self.content:
            if parts:
                parts.append("\n")
            parts.append("💬 Response\n", style="bold green")
            parts.append(self.content, style="white")
            parts.append("\n")
        if not self.tool_call_acc.empty:
            if parts:
                parts.append("\n")
            parts.append("🔧 Tool Calls\n", style="bold yellow")
            for call in self.tool_call_acc.calls:
                name = call["name"]
                args_str = call["arguments"]
                parts.append(f"  ▸ {name}", style="yellow")
                parts.append(f"\n    {_fmt_args(args_str)}\n", style="dim yellow")
        return parts


class ToolExecSegment:
    """One tool execution with start/progress/result."""

    def __init__(self):
        self.name: str = ""
        self.args: str = ""
        self.progress: list[str] = []
        self.result: str | None = None
        self.done: bool = False

    def render(self) -> Text:
        parts = Text()
        icon = "✓" if self.done else "⏳"
        style = "bold green" if self.done else "bold magenta"
        parts.append(f"{icon} {self.name}\n", style=style)
        if self.args:
            parts.append(f"  {_fmt_args(self.args)}\n", style="dim")
        if self.progress and not self.done:
            parts.append("  " + " | ".join(self.progress) + "\n", style="dim magenta")
        if self.result is not None and self.done:
            result_str = str(self.result)
            if len(result_str) > 500:
                result_str = result_str[:500] + "..."
            parts.append(f"  → {result_str}\n", style="white")
        return parts


class LiveRenderer:
    """Phase-based live renderer that keeps segments in chronological order."""

    def __init__(self):
        self.segments: list = []
        self.current_round: LLMRoundSegment | None = None
        self.current_tool: ToolExecSegment | None = None
        self._live: Live | None = None
        self._done = False

    def reset(self):
        self.segments = []
        self.current_round = None
        self.current_tool = None
        self._done = False

    def start(self):
        self._live = Live(console=console, refresh_per_second=20, transient=False)
        self._live.__enter__()
        self._update()

    def stop(self):
        self._done = True
        if self._live:
            self._live.__exit__(None, None, None)
            self._live = None

    def _ensure_round(self) -> LLMRoundSegment:
        if self.current_round is None:
            self.current_round = LLMRoundSegment()
            self.segments.append(self.current_round)
        return self.current_round

    def _close_round(self):
        self.current_round = None

    def _ensure_tool(self) -> ToolExecSegment:
        if self.current_tool is None:
            self.current_tool = ToolExecSegment()
            self.segments.append(self.current_tool)
        return self.current_tool

    def _close_tool(self):
        self.current_tool = None

    def _build(self) -> Text:
        result = Text()
        for i, seg in enumerate(self.segments):
            if i > 0:
                result.append("\n")
            result.append(seg.render())
        return result

    def _update(self):
        if self._live is not None and not self._done:
            self._live.update(self._build())

    def add_reasoning_chunk(self, text: str):
        self._ensure_round().reasoning += text
        self._update()

    def add_content_chunk(self, text: str):
        self._ensure_round().content += text
        self._update()

    def add_tool_call_delta(self, tc_deltas):
        self._ensure_round().tool_call_acc.update(tc_deltas)
        self._update()

    def finalize_llm(self):
        self._close_round()
        self._update()

    def start_tool_execution(self, tool_calls):
        self._update()

    def set_tool_start(self, tool_name: str, args: str):
        tool = self._ensure_tool()
        tool.name = tool_name
        tool.args = args
        self._update()

    def add_tool_progress(self, msg: str):
        tool = self._ensure_tool()
        tool.progress.append(msg)
        self._update()

    def set_tool_result(self, result: str):
        tool = self._ensure_tool()
        tool.result = result
        tool.done = True
        self._close_tool()
        self._update()


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

        renderer = LiveRenderer()
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
                    round_seg = renderer.current_round
                    if event.content and (round_seg is None or not round_seg.content):
                        renderer.add_content_chunk(event.content)
                    renderer.finalize_llm()

                elif isinstance(event, ExecutionStartEvent):
                    renderer.start_tool_execution(event.tool_calls)

                elif isinstance(event, ToolStartEvent):
                    tool_name = event.tool_call.get("function", {}).get(
                        "name", "unknown"
                    )
                    args = event.tool_call.get("function", {}).get("arguments", "{}")
                    renderer.set_tool_start(tool_name, args)

                elif isinstance(event, ToolProgressEvent):
                    chunk_data = event.chunk
                    if isinstance(chunk_data, dict):
                        msg = chunk_data.get(
                            "message", chunk_data.get("status", str(chunk_data))
                        )
                        renderer.add_tool_progress(str(msg))
                    elif chunk_data is not None:
                        renderer.add_tool_progress(str(chunk_data))

                elif isinstance(event, ToolEndEvent):
                    renderer.set_tool_result(str(event.result))

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

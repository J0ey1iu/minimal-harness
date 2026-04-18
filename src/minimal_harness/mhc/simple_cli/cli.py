import asyncio
import json
import platform
import sys
import threading
from pathlib import Path
from typing import Any, cast

from openai import AsyncOpenAI
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from minimal_harness.agent import OpenAIAgent
from minimal_harness.llm import ToolCall
from minimal_harness.llm.openai import OpenAILLMProvider
from minimal_harness.memory import (
    ConversationMemory,
    ExtendedInputContentPart,
)
from minimal_harness.tool.registry import ToolRegistry

console = Console()


class _CbreakMode:
    def __init__(self) -> None:
        self._original_settings: Any = None
        self._fd: int | None = None
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def __enter__(self) -> "_CbreakMode":
        if platform.system() == "Windows" or not sys.stdin.isatty():
            return self
        import termios
        import tty

        fd = sys.stdin.fileno()
        self._fd = fd
        self._original_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        self._active = True
        return self

    def __exit__(self, *_: object) -> None:
        self._restore()

    def _restore(self) -> None:
        if (
            self._active
            and self._fd is not None
            and self._original_settings is not None
        ):
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._original_settings)
            self._active = False

    def _set_cbreak(self) -> None:
        if self._fd is not None:
            import tty

            tty.setcbreak(self._fd)
            self._active = True

    def canonical_input(self, prompt: str) -> str:
        if not self._active or platform.system() == "Windows":
            return input(prompt)

        import termios

        assert self._fd is not None and self._original_settings is not None
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._original_settings)
        self._active = False

        try:
            result = input(prompt).strip()
        finally:
            self._set_cbreak()

        return result


def _monitor_esc_key(stop_event: asyncio.Event, pause_event: threading.Event) -> None:
    if platform.system() == "Windows":
        import msvcrt
        import time

        while not stop_event.is_set():
            if pause_event.is_set():
                time.sleep(0.05)
                continue
            if msvcrt.kbhit():  # type: ignore[attr-defined]
                ch = msvcrt.getch()  # type: ignore[attr-defined]
                if ch == b"\x1b":
                    stop_event.set()
                    return
            time.sleep(0.05)
    else:
        import select

        while not stop_event.is_set():
            if pause_event.is_set():
                import time

                time.sleep(0.05)
                continue
            if select.select([sys.stdin], [], [], 0.05)[0]:
                ch = sys.stdin.read(1)
                if ch == "\x1b":
                    stop_event.set()
                    return


def extract_thinking(delta: Any) -> str | None:
    for attr in ("reasoning_content", "reasoning"):
        value = getattr(delta, attr, None)
        if value:
            return value

    provider_fields = getattr(delta, "provider_specific_fields", None)
    if provider_fields and isinstance(provider_fields, dict):
        value = provider_fields.get("reasoning_content")
        if value:
            return value

    model_extra = getattr(delta, "model_extra", None)
    if model_extra and isinstance(model_extra, dict):
        for key in ("reasoning_content", "reasoning"):
            value = model_extra.get(key)
            if value:
                return value

    return None


class _ToolStatus:
    __slots__ = ("name", "args_preview", "progress", "result")

    def __init__(self, name: str, args_preview: str) -> None:
        self.name = name
        self.args_preview = args_preview
        self.progress: list[str] = []
        self.result: Text | None = None


class SimpleStreamHandler:
    def __init__(self) -> None:
        self.response_text = ""
        self.thinking_text = ""
        self.tool_calls: list[dict[str, Any]] = []
        self.tool_call_args: dict[int, str] = {}
        self.response_started = False
        self.thinking_started = False
        self._live: Live | None = None
        self._tool_statuses: dict[str, _ToolStatus] = {}
        self._total_tools: int = 0

    async def on_chunk(self, chunk: Any | None, is_done: bool) -> None:
        if is_done or not chunk:
            return

        choices = getattr(chunk, "choices", None)
        if not choices:
            return

        delta = choices[0].delta
        if not delta:
            return

        reasoning = extract_thinking(delta)
        if reasoning:
            self._handle_thinking(reasoning)

        if delta.tool_calls:
            self._handle_tool_call_deltas(delta.tool_calls)

        if delta.content:
            self._handle_content(delta.content)

    def _handle_thinking(self, reasoning: str) -> None:
        if not self.thinking_started:
            console.print(Text("[Thinking]", style="dim"))
            self.thinking_started = True

        if len(reasoning) > len(self.thinking_text) and reasoning.startswith(
            self.thinking_text
        ):
            new_part = reasoning[len(self.thinking_text) :]
            self.thinking_text = reasoning
        else:
            new_part = reasoning
            self.thinking_text += reasoning

        console.print(Text(new_part, style="dim"), end="")

    def _end_thinking_block(self) -> None:
        if self.thinking_started:
            console.print()
            self.thinking_started = False

    def _handle_tool_call_deltas(self, tool_calls: Any) -> None:
        for tc in tool_calls:
            idx = tc.index
            if idx not in self.tool_call_args:
                self.tool_call_args[idx] = ""
                name = tc.function.name if tc.function and tc.function.name else "?"
                self._end_thinking_block()
                console.print(Text(f"[Tool call: {name}]", style="yellow"))
                self.tool_calls.append({"index": idx, "name": name})
            if tc.function and tc.function.arguments:
                self.tool_call_args[idx] += tc.function.arguments

    def _handle_content(self, content: str) -> None:
        if not self.response_started:
            self._end_thinking_block()
            console.print(Text("Assistant:", style="cyan"), end=" ")
            self.response_started = True

        self.response_text += content
        console.print(content, end="", highlight=False)

    def _render_tool_statuses(self) -> Group:
        sections: list[Group] = []
        for tc_id, status in self._tool_statuses.items():
            lines: list[Text] = []
            lines.append(Text(f"⚡ {status.name}", style="yellow"))
            lines.append(Text(f"   args: {status.args_preview}", style="dim"))
            for msg in status.progress:
                lines.append(Text(f"   {status.name}: {msg}", style="dim"))
            if status.result is not None:
                lines.append(status.result)
            sections.append(Group(*lines))
        return Group(*sections)

    async def on_execution_start(self, tool_calls: Any) -> None:
        self._end_thinking_block()
        n = len(tool_calls)
        label = "tool" if n == 1 else "tools"
        console.print(f"\n[Running {n} {label}…]", style="magenta")
        self._tool_statuses = {}
        self._total_tools = n
        self._live = Live("", console=console, refresh_per_second=8, transient=False)
        self._live.start()

    async def on_tool_start(self, tool_call: Any, _: Any) -> None:
        name = tool_call["function"]["name"]
        tc_id = tool_call["id"]
        try:
            args_raw = tool_call["function"].get("arguments", "")
            args_obj = json.loads(args_raw) if args_raw else {}
            preview = json.dumps(args_obj, ensure_ascii=False)
            if len(preview) > 200:
                preview = preview[:200] + "…"
        except (json.JSONDecodeError, TypeError):
            preview = "-"

        self._tool_statuses[tc_id] = _ToolStatus(name, preview)
        if self._live:
            self._live.update(self._render_tool_statuses())

    async def on_tool_progress(self, tc: ToolCall, chunk: Any) -> None:
        tc_id = tc["id"]
        if tc_id in self._tool_statuses:
            self._tool_statuses[tc_id].progress.append(str(chunk))
            if self._live:
                self._live.update(self._render_tool_statuses())

    async def on_tool_end(self, tool_call: Any, result: Any) -> None:
        name = tool_call["function"]["name"]
        tc_id = tool_call["id"]
        if tc_id not in self._tool_statuses:
            return
        status = self._tool_statuses[tc_id]
        if isinstance(result, Exception):
            status.result = Text(f"✗ {name} failed: {result}", style="red")
        else:
            result_str = str(result)
            if len(result_str) > 300:
                result_str = result_str[:300] + "…"
            status.result = Text(f"✓ {name} → {result_str}", style="green")

        if self._live:
            self._live.update(self._render_tool_statuses())

        if all(s.result is not None for s in self._tool_statuses.values()):
            self._stop_live()

    def _stop_live(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def finish(self) -> None:
        self._stop_live()
        self._end_thinking_block()
        if self.response_started:
            console.print()


class SimpleCli:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "qwen3.5-27b",
        system_prompt: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.system_prompt = system_prompt

    def run(self) -> None:
        try:
            asyncio.run(self._run_async())
        except KeyboardInterrupt:
            console.print("\nGoodbye!", style="bold")
            sys.exit(0)

    async def _run_async(self) -> None:
        cbreak = _CbreakMode()
        esc_pause = threading.Event()

        async def wait_for_user_input(first_result: str) -> str:
            console.print(f"\n[User Input Required] {first_result}", style="red")
            esc_pause.set()
            result = cbreak.canonical_input("Your answer: ")
            esc_pause.clear()
            return result

        client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key or None)
        llm_provider = OpenAILLMProvider(client=client, model=self.model)

        default_prompt = self.system_prompt or "You are a helpful assistant."
        default_prompt += (
            f"\n\nYou are working on a {platform.system()} machine"
            f" and your current working directory is `{Path.cwd()}`"
        )

        memory = ConversationMemory(system_prompt=default_prompt)
        tools = ToolRegistry.get_instance().get_all()
        agent = OpenAIAgent(
            llm_provider=llm_provider,
            tools=tools,
            memory=memory,
        )

        info = (
            f"[bold]Model[/bold] : {self.model}\n"
            f"[bold]URL[/bold]   : {self.base_url}\n\n"
            "Type [bold]'exit'[/bold] or [bold]'quit'[/bold] to stop\n"
            "Press [bold]ESC[/bold] to stop generation"
        )
        console.print(
            Panel(info, title="simple-cli — minimal-harness chat", expand=False)
        )
        console.print()

        while True:
            try:
                console.print(Text("You:", style="blue"), end=" ")
                user_input = input().strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\nGoodbye!", style="bold")
                break

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit"):
                console.print("Goodbye!", style="bold")
                break

            console.print()
            handler = SimpleStreamHandler()
            stop_event = asyncio.Event()

            with cbreak:
                esc_thread = threading.Thread(
                    target=_monitor_esc_key,
                    args=(stop_event, esc_pause),
                    daemon=True,
                )
                esc_thread.start()

                try:
                    llm_provider._on_chunk = handler.on_chunk  # type: ignore[attr-defined]

                    await agent.run(
                        user_input=cast(
                            list[ExtendedInputContentPart],
                            [{"type": "text", "text": user_input}],
                        ),
                        on_tool_start=handler.on_tool_start,
                        on_tool_end=handler.on_tool_end,
                        on_execution_start=handler.on_execution_start,
                        wait_for_user_input=wait_for_user_input,
                        on_tool_progress=handler.on_tool_progress,
                        stop_event=stop_event,
                    )

                    if stop_event.is_set():
                        console.print(Text("\n[Stopped by user]", style="red"))
                except Exception as e:
                    handler._stop_live()
                    console.print(Text(f"\nError: {e}", style="bold red"))
                    console.print_exception()
                finally:
                    stop_event.set()
                    esc_thread.join(timeout=0.1)

            handler.finish()
            console.print()

            usage = memory.get_total_usage()
            tokens = usage.get("total_tokens", 0)
            console.print(Text(f"[Tokens used: {tokens:,}]", style="dim"))
            console.print()

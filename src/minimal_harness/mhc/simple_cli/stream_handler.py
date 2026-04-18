"""Stream handler for rendering LLM responses and tool execution in the terminal."""

import json
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.text import Text

from minimal_harness.types import ToolCall

console = Console()

# Truncation limits for display strings.
_MAX_ARGS_PREVIEW_LEN = 200
_MAX_RESULT_PREVIEW_LEN = 300


def extract_thinking(delta: Any) -> str | None:
    """Pull reasoning/thinking content out of a streaming delta.

    Different providers expose thinking tokens under different attribute
    names; this helper checks the known locations in priority order.
    """
    # Direct attributes
    for attr in ("reasoning_content", "reasoning"):
        value = getattr(delta, attr, None)
        if value:
            return value

    # Provider-specific fields dict
    provider_fields = getattr(delta, "provider_specific_fields", None)
    if isinstance(provider_fields, dict):
        value = provider_fields.get("reasoning_content")
        if value:
            return value

    # model_extra fallback
    model_extra = getattr(delta, "model_extra", None)
    if isinstance(model_extra, dict):
        for key in ("reasoning_content", "reasoning"):
            value = model_extra.get(key)
            if value:
                return value

    return None


class _ToolStatus:
    """Lightweight value object tracking a single tool invocation's state."""

    __slots__ = ("name", "args_preview", "progress", "result")

    def __init__(self, name: str, args_preview: str) -> None:
        self.name = name
        self.args_preview = args_preview
        self.progress: list[str] = []
        self.result: Text | None = None


class SimpleStreamHandler:
    """Handles streaming LLM output and tool execution rendering via Rich."""

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

    # ------------------------------------------------------------------
    # Streaming chunk callbacks
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Internal helpers — thinking / content / tool deltas
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Tool execution lifecycle callbacks
    # ------------------------------------------------------------------

    def _render_tool_statuses(self) -> Group:
        sections: list[Group] = []
        for status in self._tool_statuses.values():
            lines: list[Text] = [
                Text(f"\u26a1 {status.name}", style="yellow"),
                Text(f"   args: {status.args_preview}", style="dim"),
            ]
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
        console.print(f"\n[Running {n} {label}\u2026]", style="magenta")
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
            if len(preview) > _MAX_ARGS_PREVIEW_LEN:
                preview = preview[:_MAX_ARGS_PREVIEW_LEN] + "\u2026"
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
            status.result = Text(f"\u2717 {name} failed: {result}", style="red")
        else:
            result_str = str(result)
            if len(result_str) > _MAX_RESULT_PREVIEW_LEN:
                result_str = result_str[:_MAX_RESULT_PREVIEW_LEN] + "\u2026"
            status.result = Text(f"\u2713 {name} \u2192 {result_str}", style="green")

        if self._live:
            self._live.update(self._render_tool_statuses())

        if all(s.result is not None for s in self._tool_statuses.values()):
            self._stop_live()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _pause_live(self) -> None:
        if self._live is not None:
            self._live.stop()

    def _resume_live(self) -> None:
        if self._live is not None:
            self._live.start()

    def _stop_live(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def finish(self) -> None:
        self._stop_live()
        self._end_thinking_block()
        if self.response_started:
            console.print()

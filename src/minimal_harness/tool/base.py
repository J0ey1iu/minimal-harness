from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Protocol

from minimal_harness.types import (
    StreamingToolFunction,
    ToolCall,
    ToolEnd,
    ToolEvent,
    ToolProgress,
    ToolResult,
    ToolStart,
)


class ToolExecutionError(Exception):
    def __init__(self, message: str, stderr: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.stderr = stderr


class Tool(Protocol):
    name: str
    description: str
    parameters: dict
    display_name: str
    display_name_locale: dict[str, str] | None
    description_locale: dict[str, str] | None

    def to_schema(self) -> dict: ...
    def to_anthropic_schema(self) -> dict[str, Any]: ...
    def execute(
        self,
        args: dict[str, Any],
        tool_call: ToolCall,
        stop_event: asyncio.Event | None,
    ) -> AsyncIterator[ToolEvent]: ...
    def resolve_display_name(self, locale: str = "") -> str: ...
    def resolve_description(self, locale: str = "") -> str: ...


def create_streaming_tool(
    name: str,
    fn: StreamingToolFunction,
    description: str | None = None,
    parameters: dict | None = None,
    display_name: str | None = None,
    display_name_locale: dict[str, str] | None = None,
    description_locale: dict[str, str] | None = None,
) -> StreamingTool:
    tool_description = description or (fn.__doc__ or "").strip()
    tool_params = parameters or {}
    return StreamingTool(
        name=name,
        display_name=display_name,
        description=tool_description,
        parameters=tool_params,
        fn=fn,
        display_name_locale=display_name_locale,
        description_locale=description_locale,
    )


class StreamingTool(Tool):
    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict,
        fn: StreamingToolFunction,
        display_name: str | None = None,
        display_name_locale: dict[str, str] | None = None,
        description_locale: dict[str, str] | None = None,
        timeout: float | None = None,
    ):
        self.name = name
        self.display_name = display_name or name
        self.description = description
        self.parameters = parameters
        self.fn = fn
        self.display_name_locale = display_name_locale
        self.description_locale = description_locale
        # RFC #57: wall-clock budget for the whole tool run (seconds).
        # ``None`` keeps the pre-existing unbounded behaviour. On
        # timeout/cancel, chunks already emitted stay visible
        # (partial-chunk replay) and the run ends with a ToolEnd error.
        self.timeout = timeout

    def resolve_display_name(self, locale: str = "") -> str:
        if locale and self.display_name_locale and locale in self.display_name_locale:
            return self.display_name_locale[locale]
        return self.display_name

    def resolve_description(self, locale: str = "") -> str:
        if locale and self.description_locale and locale in self.description_locale:
            return self.description_locale[locale]
        return self.description

    def to_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    async def execute(
        self,
        args: dict[str, Any],
        tool_call: ToolCall,
        stop_event: asyncio.Event | None,
    ) -> AsyncIterator[ToolEvent]:
        yield ToolStart(tool_call)

        final_result = None
        error_msg: str | None = None
        try:
            if self.timeout is None:
                async for chunk in self.fn(**args):
                    if isinstance(chunk, ToolResult):
                        final_result = chunk
                    else:
                        yield ToolProgress(tool_call, chunk)
                        final_result = chunk
            else:
                async for chunk in self._run_bounded(args, stop_event):
                    if isinstance(chunk, ToolResult):
                        final_result = chunk
                    else:
                        yield ToolProgress(tool_call, chunk)
                        final_result = chunk
        except asyncio.CancelledError:
            error_msg = "stopped by the user"
        except ToolExecutionError as e:
            error_msg = f"[Error] {e.message}"
        except asyncio.TimeoutError:
            error_msg = f"[Error] tool exceeded its {self.timeout:g}s budget"
        except Exception as e:
            error_msg = f"[Error] {type(e).__name__}: {e}"

        if error_msg is not None:
            yield ToolEnd(tool_call, error_msg)
        else:
            yield ToolEnd(tool_call, final_result)

    async def _run_bounded(
        self,
        args: dict[str, Any],
        stop_event: asyncio.Event | None,
    ) -> AsyncIterator[Any]:
        """RFC #57 bounded + cancellable runner for a local streaming fn.

        The producer task streams chunks through a queue while the
        consumer races each wait against the deadline and the stop
        event. On timeout/cancel the producer is destroyed but every
        chunk already emitted above stays visible (partial-chunk
        replay) — the UI keeps the diagnostics the tool emitted just
        before it was killed.
        """
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

        async def _produce() -> None:
            try:
                async for chunk in self.fn(**args):
                    await queue.put(("chunk", chunk))
                await queue.put(("end", None))
            except asyncio.CancelledError:
                await queue.put(("cancelled", None))
                raise
            except Exception as exc:
                await queue.put(("error", exc))

        producer = asyncio.create_task(_produce())
        timeout = self.timeout
        assert timeout is not None, "_run_bounded requires a timeout"
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        try:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise asyncio.TimeoutError
                kind, payload = await asyncio.wait_for(queue.get(), timeout=remaining)
                if kind == "chunk":
                    if stop_event is not None and stop_event.is_set():
                        raise asyncio.CancelledError
                    yield payload
                elif kind == "error":
                    raise payload
                elif kind == "cancelled":
                    raise asyncio.CancelledError
                else:
                    return
        finally:
            if not producer.done():
                producer.cancel()
                try:
                    await producer
                except (asyncio.CancelledError, Exception):
                    pass

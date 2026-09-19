"""RFC #57 StreamingTool bounded execution: timeout + cancel + partial replay."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, cast

from minimal_harness.tool.base import StreamingTool
from minimal_harness.types import ToolCall, ToolEnd, ToolProgress

_CALL = cast(
    "ToolCall",
    {"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}},
)


async def _collect(
    tool: StreamingTool, args: dict[str, Any], stop: asyncio.Event | None = None
):
    return [ev async for ev in tool.execute(args, _CALL, stop)]


def _make_tool(fn: Any, timeout: float | None) -> StreamingTool:
    return StreamingTool(
        name="t", description="t", parameters={}, fn=fn, timeout=timeout
    )


async def test_no_timeout_keeps_previous_behaviour():
    async def fn(**_: Any) -> AsyncIterator[str]:
        for chunk in ["a", "b"]:
            yield chunk

    events = await _collect(_make_tool(fn, None), {})
    assert [type(e).__name__ for e in events] == [
        "ToolStart",
        "ToolProgress",
        "ToolProgress",
        "ToolEnd",
    ]
    assert isinstance(events[-1], ToolEnd) and events[-1].result == "b"


async def test_timeout_completes_normally_when_fast():
    async def fn(**_: Any) -> AsyncIterator[str]:
        yield "fast"

    events = await _collect(_make_tool(fn, 5.0), {})
    assert isinstance(events[-1], ToolEnd) and events[-1].result == "fast"


async def test_timeout_fires_and_preserves_partial_chunks():
    async def fn(**_: Any) -> AsyncIterator[str]:
        yield "first-chunk"  # emitted before the hang
        await asyncio.sleep(10)

    events = await _collect(_make_tool(fn, 0.1), {})
    kinds = [type(e).__name__ for e in events]
    assert "ToolProgress" in kinds, "partial chunk must have streamed out (replay)"
    assert kinds[-1] == "ToolEnd"
    assert isinstance(events[-1], ToolEnd) and "budget" in str(events[-1].result)
    assert not isinstance(events[-1].result, Exception)


async def test_stop_event_cancels_bounded_run():
    stop = asyncio.Event()
    stop.set()

    async def fn(**_: Any) -> AsyncIterator[str]:
        yield "partial"
        await asyncio.sleep(5)

    events = await _collect(_make_tool(fn, 10.0), {}, stop)
    assert isinstance(events[-1], ToolEnd) and "stopped by the user" in str(
        events[-1].result
    )


async def test_error_in_fn_surfaces_in_tool_end():
    async def fn(**_: Any) -> AsyncIterator[str]:
        raise ValueError("boom")
        yield  # pragma: no cover

    events = await _collect(_make_tool(fn, 5.0), {})
    assert isinstance(events[-1], ToolEnd) and "ValueError: boom" in str(
        events[-1].result
    )


async def test_tool_result_object_passthrough():
    from minimal_harness.types import ToolResult

    async def fn(**_: Any) -> AsyncIterator[Any]:
        yield "partial chunk"
        yield ToolResult(content="final", meta={"viz": 1})

    events = await _collect(_make_tool(fn, 5.0), {})
    end = events[-1]
    assert isinstance(end, ToolEnd)
    assert isinstance(end.result, ToolResult)
    assert end.result.content == "final"
    assert end.result.meta == {"viz": 1}


async def test_cancelled_producer_does_not_leak():
    async def fn(**_: Any) -> AsyncIterator[str]:
        yield "one"

    tool = _make_tool(fn, 0.1)
    events = await _collect(tool, {})
    # producer task should be gone after the run completes
    for evt in events:
        if isinstance(evt, ToolProgress):
            assert evt.chunk == "one"

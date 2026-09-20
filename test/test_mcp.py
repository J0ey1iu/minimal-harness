"""RFC #57 MCP stdio client tests — driven against a real fake subprocess."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from minimal_harness.tool.factory import DefaultToolFactory
from minimal_harness.tool.mcp import MCPManager, MCPToolExecutorFactory, ToolError
from minimal_harness.types import (
    MCPToolBinding,
    ToolCall,
    ToolEnd,
    ToolMetadata,
    ToolResult,
    ToolStart,
)

FAKE_SERVER = """\
import json, sys

def read():
    line = sys.stdin.readline()
    return json.loads(line) if line else None

while True:
    msg = read()
    if msg is None:
        break
    if "id" not in msg:
        continue  # notification
    rid, method = msg["id"], msg["method"]
    result = {"ok": True}
    if method == "initialize":
        result = {"protocolVersion": "2024-11-05", "capabilities": {}, "serverInfo": {"name": "fake", "version": "0.0.1"}}
    elif method == "tools/list":
        result = {"tools": [{"name": "echo", "description": "echo args", "inputSchema": {"type": "object"}}]}
    elif method == "tools/call":
        name = msg["params"]["name"]
        args = msg["params"]["arguments"] or {}
        if name == "boom":
            result = {"content": [{"type": "text", "text": "kaboom"}], "isError": True}
        else:
            result = {"content": [{"type": "text", "text": "hello " + str(args.get("name", ""))}]}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": rid, "result": result}) + "\\n")
    sys.stdout.flush()
"""


@pytest.fixture()
def server_script(tmp_path: Path) -> Path:
    p = tmp_path / "fake_mcp_server.py"
    p.write_text(FAKE_SERVER, encoding="utf-8")
    return p


def _binding(script: Path, slug: str = "fake") -> MCPToolBinding:
    return MCPToolBinding(
        command=sys.executable,
        args=["-u", str(script)],
        server_slug=slug,
        client_name="test",
        client_version="0.0.1",
    )


async def test_connect_handshake_and_call_tool(server_script):
    manager = MCPManager()
    try:
        binding = _binding(server_script)
        out = await manager.call_tool(binding, "echo", {"name": "world"})
        assert out == "hello world"
    finally:
        await manager.shutdown()


async def test_server_connection_reused_across_calls(server_script):
    manager = MCPManager()
    try:
        binding = _binding(server_script)
        await manager.call_tool(binding, "echo", {"name": "a"})
        conn1 = manager._conns["fake"].process.pid
        await manager.call_tool(binding, "echo", {"name": "b"})
        conn2 = manager._conns["fake"].process.pid
        assert conn1 == conn2, "per-server subprocess must be reused"
    finally:
        await manager.shutdown()


async def test_tool_error_surfaces(server_script):
    manager = MCPManager()
    try:
        with pytest.raises(ToolError, match="kaboom"):
            await manager.call_tool(_binding(server_script), "boom", {})
    finally:
        await manager.shutdown()


async def test_shutdown_terminates_subprocess(server_script):
    manager = MCPManager()
    binding = _binding(server_script)
    await manager.call_tool(binding, "echo", {"name": "x"})
    pid = manager._conns["fake"].process.pid
    await manager.shutdown()
    assert manager._conns == {}
    # process exited
    import os

    poll = os.waitpid(pid, os.WNOHANG) if os.name == "posix" else None
    if poll is not None:
        assert poll[0] == pid


async def test_executor_via_default_tool_factory(server_script):
    manager = MCPManager()
    try:
        factory = DefaultToolFactory(
            executor_factories={"mcp": MCPToolExecutorFactory(manager=manager)}
        )
        tool = factory.create(
            ToolMetadata(
                name="echo",
                description="echo",
                binding=_binding(server_script),
            )
        )
        tool_call: ToolCall = {
            "id": "mcp-1",
            "type": "function",
            "function": {"name": "echo", "arguments": '{"name": "pi"}'},
        }
        events = [ev async for ev in tool.execute({"name": "pi"}, tool_call, None)]
        assert isinstance(events[0], ToolStart)
        end = events[-1]
        assert isinstance(end, ToolEnd)
        assert isinstance(end.result, ToolResult)
        assert end.result.content == "hello pi"
    finally:
        await manager.shutdown()


async def test_factory_without_mcp_driver_raises(server_script):
    factory = DefaultToolFactory()
    with pytest.raises(ValueError, match="mcp"):
        factory.create(
            ToolMetadata(
                name="echo", description="echo", binding=_binding(server_script)
            )
        )


async def test_list_tools_enumerates_server(server_script):
    manager = MCPManager()
    try:
        tools = await manager.list_tools(_binding(server_script))
        assert any(t.get("name") == "echo" for t in tools)
    finally:
        await manager.shutdown()


async def test_disconnect_kills_one_server_keeps_others(server_script):
    manager = MCPManager()
    try:
        a = _binding(server_script, slug="a")
        b = _binding(server_script, slug="b")
        await manager.call_tool(a, "echo", {"name": "a"})
        await manager.call_tool(b, "echo", {"name": "b"})
        pid_b = manager._conns["b"].process.pid
        await manager.disconnect("a")
        assert "a" not in manager._conns
        assert "b" in manager._conns
        out = await manager.call_tool(b, "echo", {"name": "again"})
        assert out == "hello again"
        assert manager._conns["b"].process.pid == pid_b
    finally:
        await manager.shutdown()

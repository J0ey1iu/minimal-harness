"""MCP stdio client binding (JSON-RPC 2.0 over subprocess stdio).

RFC #57 (mhc-desktop): MCP is the de-facto standard tool protocol, but the
base only shipped ``RemoteTool`` (SSE-over-HTTP) and script bindings. This
module adds the missing driver:

- :class:`MCPToolBinding` (in :mod:`minimal_harness.types`) declares a
  stdio MCP server; it resolves through the ``"mcp"`` executor driver on
  :class:`~minimal_harness.tool.factory.DefaultToolFactory`.
- :class:`MCPManager` owns the per-server subprocess lifecycle (reused
  across calls, one ``asyncio.Lock`` per server serializing strict-id
  JSON-RPC writes), the ``initialize`` handshake and ``tools/call``.
- :class:`MCPToolExecutor` adapts the manager to the
  :class:`~minimal_harness.tool.remote.RemoteToolExecutor` protocol so
  MCP tools emit the same ``ToolStart / ToolProgress / ToolEnd`` event
  stream as every other tool driver.

Usage::

    manager = MCPManager()
    factory = DefaultToolFactory(
        executor_factories={"mcp": MCPToolExecutorFactory(manager=manager)}
    )
    await manager.shutdown()  # terminate subprocesses on app exit

No new dependencies: JSON-RPC framing uses ``asyncio`` subprocesses and
the stdlib ``json`` module.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator

from minimal_harness.types import (
    MCPToolBinding,
    ToolCall,
    ToolEnd,
    ToolEvent,
    ToolResult,
    ToolStart,
)

if TYPE_CHECKING:
    pass


class ToolError(Exception):
    """Raised when an MCP server cannot be reached or returns an error."""


def _flatten_content(content: Any) -> str:
    """Concatenate MCP ``content`` blocks (``[{type, text}]``) into text."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
            else:
                parts.append(str(text))
        else:
            parts.append(str(block))
    return "".join(parts)


@dataclass
class MCPConnection:
    """One live stdio subprocess plus its JSON-RPC framing state."""

    process: asyncio.subprocess.Process
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    next_id: int = 1


class MCPManager:
    """Owns MCP server subprocesses; one process per ``server_slug``.

    A strict-id client: requests are serialized per server via a lock,
    replies are matched by ``id`` (stray server frames are skipped), and
    the ``initialize`` handshake runs once per process.
    """

    def __init__(
        self, client_name: str = "minimal-harness", client_version: str = "0.8.1"
    ) -> None:
        self._client_name = client_name
        self._client_version = client_version
        self._conns: dict[str, MCPConnection] = {}

    async def connect(self, server: MCPToolBinding) -> MCPConnection:
        conn = self._conns.get(server.server_slug)
        if conn is not None and conn.process.returncode is None:
            return conn
        proc = await asyncio.create_subprocess_exec(
            shutil.which(server.command) or server.command,
            *server.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, **server.env},
        )
        conn = MCPConnection(process=proc)
        self._conns[server.server_slug] = conn
        try:
            await self._rpc(
                conn,
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "clientInfo": {
                        "name": server.client_name or self._client_name,
                        "version": server.client_version or self._client_version,
                    },
                    "capabilities": {},
                },
            )
            await self._notify(conn, "notifications/initialized", {})
        except Exception as e:
            await self._discard(server.server_slug)
            raise ToolError(f"MCP '{server.server_slug}' initialize failed: {e}") from e
        return conn

    async def _discard(self, slug: str) -> None:
        conn = self._conns.pop(slug, None)
        if conn is None:
            return
        conn.process.terminate()
        try:
            await asyncio.wait_for(conn.process.wait(), timeout=3)
        except asyncio.TimeoutError:
            conn.process.kill()
            await conn.process.wait()

    async def _rpc(
        self, conn: MCPConnection, method: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        async with conn.lock:
            req_id = conn.next_id
            conn.next_id += 1
            envelope = {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": method,
                "params": params,
            }
            conn.process.stdin.write(  # type: ignore[union-attr]
                (json.dumps(envelope, ensure_ascii=False) + "\n").encode()
            )
            await conn.process.stdin.drain()  # type: ignore[union-attr]
            while True:
                raw = await conn.process.stdout.readline()  # type: ignore[union-attr]
                if not raw:
                    raise ToolError("MCP subprocess closed before reply")
                try:
                    msg = json.loads(raw.decode("utf-8", errors="replace").strip())
                except json.JSONDecodeError:
                    continue
                if msg.get("id") != req_id:
                    continue
                if "error" in msg:
                    err = msg["error"] or {}
                    raise ToolError(
                        f"MCP error {err.get('code')}: {err.get('message')}"
                    )
                result = msg.get("result")
                return result if isinstance(result, dict) else {}

    async def _notify(
        self, conn: MCPConnection, method: str, params: dict[str, Any]
    ) -> None:
        envelope = {"jsonrpc": "2.0", "method": method, "params": params}
        conn.process.stdin.write(  # type: ignore[union-attr]
            (json.dumps(envelope, ensure_ascii=False) + "\n").encode()
        )
        await conn.process.stdin.drain()  # type: ignore[union-attr]

    async def call_tool(
        self,
        server: MCPToolBinding,
        name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Invoke ``tools/call`` and return the flattened text content."""
        conn = await self.connect(server)
        res = await self._rpc(
            conn, "tools/call", {"name": name, "arguments": arguments}
        )
        if res.get("isError"):
            raise ToolError(
                f"MCP '{server.server_slug}' tool '{name}' returned error: "
                f"{_flatten_content(res.get('content') or [])}"
            )
        return _flatten_content(res.get("content") or [])

    async def shutdown(self) -> None:
        """Terminate all servers (SIGTERM → 3s → SIGKILL)."""

        async def _stop(slug: str) -> None:
            await self._discard(slug)

        await asyncio.gather(*(_stop(slug) for slug in list(self._conns)))


class MCPToolExecutor:
    """Adapts :class:`MCPManager` to the ``RemoteToolExecutor`` protocol."""

    def __init__(self, manager: MCPManager, server: MCPToolBinding) -> None:
        self._manager = manager
        self._server = server

    async def execute(
        self,
        args: dict[str, Any],
        tool_call: ToolCall,
        stop_event: Any,  # unused: MCP calls are routed through the manager
    ) -> AsyncIterator[ToolEvent]:
        yield ToolStart(tool_call)
        try:
            content = await self._manager.call_tool(
                self._server, tool_call["function"]["name"], args
            )
        except ToolError as e:
            yield ToolEnd(tool_call, f"[Error] {e}")
            return
        # Check cancellation between the manager call and the payload,
        # mirroring the mid-stream cancel semantics of other drivers.
        if stop_event is not None and stop_event.is_set():
            yield ToolEnd(tool_call, "stopped by the user")
            return
        yield ToolEnd(tool_call, ToolResult(content=content))


class MCPToolExecutorFactory:
    """Executor factory for the ``\"mcp\"`` driver.

    Shares one :class:`MCPManager` across every tool bound to it, so the
    per-server subprocess is reused across calls.
    """

    def __init__(
        self,
        manager: MCPManager | None = None,
        client_name: str = "minimal-harness",
        client_version: str = "0.8.1",
    ) -> None:
        self._manager = manager or MCPManager(
            client_name=client_name, client_version=client_version
        )
        self.manager = self._manager

    def create(self, binding: MCPToolBinding) -> MCPToolExecutor:
        if not isinstance(binding, MCPToolBinding):
            raise TypeError(
                "MCPToolExecutorFactory requires an MCPToolBinding, got "
                f"{type(binding).__name__}"
            )
        return MCPToolExecutor(self._manager, binding)


__all__ = [
    "MCPConnection",
    "MCPManager",
    "MCPToolExecutor",
    "MCPToolExecutorFactory",
    "ToolError",
    "_flatten_content",
]

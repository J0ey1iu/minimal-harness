import asyncio
import locale
from typing import AsyncIterator

from minimal_harness.tool.base import StreamingTool, Tool
from minimal_harness.types import ToolResult


def _decode(data: bytes | None) -> str:
    if not data:
        return ""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        fallback = locale.getpreferredencoding(False) or "utf-8"
        return data.decode(fallback, errors="replace")


_MAX_PROGRESS_CHARS = 3000
_PROGRESS_FLUSH_INTERVAL = 0.3
_PROGRESS_FLUSH_LINES = 10


async def _reader(
    stream: asyncio.StreamReader | None,
    dest: list[str],
    queue: asyncio.Queue[str],
    max_progress_chars: int,
) -> None:
    """Read lines from a subprocess stream.

    All lines are accumulated in *dest* (for the final LLM result).
    Only the first *max_progress_chars* characters are put into *queue*
    (for TUI streaming progress), preventing TUI freezing on large output.
    """
    if stream is None:
        return
    total_progress = 0
    progress_truncated = False
    while True:
        try:
            line = await stream.readline()
        except Exception:
            break
        if not line:
            break
        text = _decode(line).rstrip("\n").rstrip("\r")

        dest.append(text)

        if not progress_truncated:
            if total_progress + len(text) > max_progress_chars:
                progress_truncated = True
                remaining = max_progress_chars - total_progress
                if remaining > 0:
                    await queue.put(text[:remaining])
                await queue.put("... (output truncated)")
                continue
            total_progress += len(text)
            if text:
                await queue.put(text)


async def bash_handler(
    command: str, timeout: float | None = None, workdir: str | None = None
) -> AsyncIterator[dict | ToolResult]:
    yield {
        "status": "progress",
        "message": f"Executing: {command[:50]}{'...' if len(command) > 50 else ''}",
    }

    process = await asyncio.create_subprocess_shell(
        command,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=workdir,
    )

    queue: asyncio.Queue[str] = asyncio.Queue()
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    timed_out = False

    stdout_task = asyncio.create_task(
        _reader(process.stdout, stdout_lines, queue, _MAX_PROGRESS_CHARS)
    )
    stderr_task = asyncio.create_task(
        _reader(process.stderr, stderr_lines, queue, _MAX_PROGRESS_CHARS)
    )

    start_time = asyncio.get_running_loop().time()

    progress_buf: list[str] = []
    last_flush = 0.0

    try:
        while True:
            if timeout is not None:
                elapsed = asyncio.get_running_loop().time() - start_time
                if elapsed >= timeout:
                    timed_out = True
                    break

            try:
                line = await asyncio.wait_for(queue.get(), timeout=0.1)
                progress_buf.append(line)
            except asyncio.TimeoutError:
                pass

            now = asyncio.get_running_loop().time()
            if progress_buf and (
                len(progress_buf) >= _PROGRESS_FLUSH_LINES
                or now - last_flush >= _PROGRESS_FLUSH_INTERVAL
            ):
                yield {"status": "progress", "message": "\n".join(progress_buf)}
                progress_buf.clear()
                last_flush = now

            if process.returncode is not None and queue.empty():
                if progress_buf:
                    yield {"status": "progress", "message": "\n".join(progress_buf)}
                break
    except asyncio.CancelledError:
        raise
    finally:
        if process.returncode is None:
            try:
                process.kill()
                await process.wait()
            except Exception:
                pass

        for task in (stdout_task, stderr_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)

    if timed_out:
        yield ToolResult(
            content=f"Command timed out after {timeout}s",
            meta={"exit_code": -1},
        )
        return

    stdout_all = "\n".join(stdout_lines)
    stderr_all = "\n".join(stderr_lines)
    exit_code = process.returncode or 0

    if exit_code == 0:
        content = stdout_all
    else:
        parts = [f"Exit code: {exit_code}"]
        if stderr_all:
            parts.append(stderr_all)
        if stdout_all and stdout_all != stderr_all:
            parts.append(stdout_all)
        content = "\n".join(parts)

    meta: dict = {"exit_code": exit_code}
    if stderr_all:
        meta["stderr"] = stderr_all

    yield ToolResult(content=content, meta=meta)


bash_tool = StreamingTool(
    name="bash",
    display_name="Bash",
    display_name_locale={"zh": "命令行"},
    description="Execute a shell command and return the terminal output (stdout + stderr). Compatible with Windows, Linux, and macOS.",
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute",
            },
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds. If the command exceeds this duration, it will be killed (default: no limit)",
            },
            "workdir": {
                "type": "string",
                "description": "Working directory for the command (optional)",
            },
        },
        "required": ["command"],
    },
    fn=bash_handler,
)


def get_tools() -> dict[str, Tool]:
    return {"bash": bash_tool}

import asyncio
import locale
from typing import AsyncIterator

from minimal_harness.tool.base import StreamingTool


def _decode(data: bytes | None) -> str:
    if not data:
        return ""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        fallback = locale.getpreferredencoding(False) or "utf-8"
        return data.decode(fallback, errors="replace")


async def bash_handler(
    command: str, timeout: float | None = None, workdir: str | None = None
) -> AsyncIterator[dict]:
    yield {"status": "progress", "message": f"Executing: {command[:50]}{'...' if len(command) > 50 else ''}"}

    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=workdir,
    )

    queue: asyncio.Queue[str] = asyncio.Queue()
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    timed_out = False

    async def _reader(stream: asyncio.StreamReader | None, dest: list[str]) -> None:
        if stream is None:
            return
        while True:
            try:
                line = await stream.readline()
            except Exception:
                break
            if not line:
                break
            text = _decode(line).rstrip("\n").rstrip("\r")
            dest.append(text)
            if text:
                await queue.put(text)

    stdout_task = asyncio.create_task(_reader(process.stdout, stdout_lines))
    stderr_task = asyncio.create_task(_reader(process.stderr, stderr_lines))

    start_time = asyncio.get_running_loop().time()

    try:
        while True:
            if timeout is not None:
                elapsed = asyncio.get_running_loop().time() - start_time
                if elapsed >= timeout:
                    timed_out = True
                    raise asyncio.TimeoutError

            try:
                chunk = await asyncio.wait_for(queue.get(), timeout=0.1)
                yield {"status": "progress", "message": chunk}
            except asyncio.TimeoutError:
                if process.returncode is not None and queue.empty():
                    break
    except asyncio.TimeoutError:
        if not timed_out:
            raise
        process.kill()
        try:
            await process.wait()
        except Exception:
            pass
        yield {"stdout": "", "stderr": f"Command timed out after {timeout}s"}
        return
    except asyncio.CancelledError:
        process.kill()
        try:
            await process.wait()
        except Exception:
            pass
        raise
    finally:
        for task in (stdout_task, stderr_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        while not queue.empty():
            try:
                yield {"status": "progress", "message": queue.get_nowait()}
            except asyncio.QueueEmpty:
                break

    stdout_all = "\n".join(stdout_lines)
    stderr_all = "\n".join(stderr_lines)
    yield {"stdout": stdout_all, "stderr": stderr_all}


bash_tool = StreamingTool(
    name="bash",
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


def get_tools() -> dict[str, StreamingTool]:
    return {"bash": bash_tool}

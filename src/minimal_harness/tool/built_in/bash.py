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
) -> AsyncIterator[str]:
    yield f"I'm about to execute bash command: {command[:50]}{'...' if len(command) > 50 else ''}"

    # Use create_subprocess_shell so the command is parsed by the system
    # shell exactly as if it were typed into a terminal.
    #
    # On Windows this avoids a well-known quoting bug with cmd.exe /c:
    # when create_subprocess_exec() passes the command string through
    # list2cmdline(), cmd.exe sees the argument start with a quote and
    # strips the first and last quotes. This breaks any argument that
    # contains spaces and quotes (e.g. python -c "print('hello world')").
    # create_subprocess_shell() delegates to the standard library's
    # shell=True handling which works around the issue.
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=workdir,
    )

    queue: asyncio.Queue[str] = asyncio.Queue()

    async def _reader(stream: asyncio.StreamReader | None) -> None:
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
            if text:
                await queue.put(text)

    stdout_task = asyncio.create_task(_reader(process.stdout))
    stderr_task = asyncio.create_task(_reader(process.stderr))

    start_time = asyncio.get_running_loop().time()
    timed_out = False

    try:
        while True:
            if timeout is not None:
                elapsed = asyncio.get_running_loop().time() - start_time
                if elapsed >= timeout:
                    timed_out = True
                    raise asyncio.TimeoutError

            try:
                chunk = await asyncio.wait_for(queue.get(), timeout=0.1)
                yield chunk
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
        stdout_task.cancel()
        stderr_task.cancel()
        yield f"[Timeout] Command timed out after {timeout}s and was killed."
        return
    except asyncio.CancelledError:
        process.kill()
        try:
            await process.wait()
        except Exception:
            pass
        stdout_task.cancel()
        stderr_task.cancel()
        raise
    finally:
        for task in (stdout_task, stderr_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)

    yield f"[Exit code: {process.returncode}]"


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

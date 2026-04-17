import asyncio
import locale
import platform

from minimal_harness.tool.base import Tool


async def bash_handler(command: str, timeout: float | None = None) -> str:
    current_os = platform.system()
    if current_os == "Windows":
        process = await asyncio.create_subprocess_exec(
            "cmd.exe",
            "/c",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    else:
        process = await asyncio.create_subprocess_exec(
            "/bin/sh",
            "-c",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        return f"[Timeout] Command timed out after {timeout}s and was killed."
    encoding = locale.getpreferredencoding(False) or "utf-8"
    output_parts = []
    if stdout:
        output_parts.append(stdout.decode(encoding, errors="replace"))
    if stderr:
        output_parts.append(stderr.decode(encoding, errors="replace"))
    output = "\n".join(output_parts).strip()
    if not output:
        return f"[OK] Command exited with code {process.returncode} (no output)"
    return f"{output}\n[Exit code: {process.returncode}]"


bash_tool = Tool(
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
        },
        "required": ["command"],
    },
    fn=bash_handler,
)


def get_tools() -> dict[str, Tool]:
    return {"bash": bash_tool}

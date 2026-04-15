import asyncio
import locale
import platform
from pathlib import Path
from typing import Optional


async def create_file(file_path: str, content: str = "") -> dict:
    """Create a new file with the given content. Fails if the file already exists."""
    path = Path(file_path).expanduser().resolve()
    if path.exists():
        return {"success": False, "error": f"File already exists: {path}"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {"success": True, "file_path": str(path), "bytes_written": len(content)}


async def read_file(
    file_path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
) -> dict:
    """Read the contents of a file, optionally restricting to a line range.

    Args:
        file_path: Path to the file.
        start_line: 1-based first line to include (default: beginning of file).
        end_line: 1-based last line to include (default: end of file).
    """
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        return {"success": False, "error": f"File not found: {path}"}

    all_lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    total_lines = len(all_lines)

    start = (start_line - 1) if start_line is not None else 0
    end = end_line if end_line is not None else total_lines

    if start < 0 or end > total_lines or start >= end:
        return {
            "success": False,
            "error": (
                f"Invalid line range [{start_line}–{end_line}] "
                f"for file with {total_lines} lines."
            ),
        }

    selected = all_lines[start:end]
    content = "".join(selected)

    return {
        "success": True,
        "file_path": str(path),
        "content": content,
        "total_lines": total_lines,
        "range": [start + 1, end],
    }


async def patch_file(
    file_path: str,
    content: str,
    mode: str = "append",
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
) -> dict:
    """Patch a file with several supported modes.

    Modes
    -----
    append      – Append *content* to the end of the file.
    prepend     – Prepend *content* to the beginning of the file.
    overwrite   – Replace the entire file with *content*.
    insert      – Insert *content* **before** *start_line* (1-based).
                  Existing lines are not removed.
    replace     – Replace lines [*start_line* … *end_line*] (1-based,
                  inclusive) with *content*.  If *end_line* is omitted it
                  defaults to *start_line* (i.e. replace a single line).
    delete      – Delete lines [*start_line* … *end_line*] (1-based,
                  inclusive).  *content* is ignored.

    Args:
        file_path: Path to the file.
        content: The text to write (ignored for ``delete`` mode).
        mode: One of the modes described above.
        start_line: 1-based line number (required for insert / replace / delete).
        end_line: 1-based ending line (inclusive, used by replace / delete).
    """
    path = Path(file_path).expanduser().resolve()

    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = existing.splitlines(keepends=True)
    total_lines = len(lines)

    # ── whole-file modes ────────────────────────────────────────────
    if mode == "overwrite":
        path.write_text(content, encoding="utf-8")
        return {
            "success": True,
            "file_path": str(path),
            "mode": mode,
            "bytes_written": len(content),
        }

    if mode == "append":
        new_text = existing + content
        path.write_text(new_text, encoding="utf-8")
        return {
            "success": True,
            "file_path": str(path),
            "mode": mode,
            "bytes_written": len(content),
        }

    if mode == "prepend":
        new_text = content + existing
        path.write_text(new_text, encoding="utf-8")
        return {
            "success": True,
            "file_path": str(path),
            "mode": mode,
            "bytes_written": len(content),
        }

    # ── line-level modes ────────────────────────────────────────────
    if start_line is None:
        return {
            "success": False,
            "error": f"Mode '{mode}' requires at least start_line.",
        }

    idx = start_line - 1  # 0-based

    if mode == "insert":
        if idx < 0 or idx > total_lines:
            return {
                "success": False,
                "error": (
                    f"start_line {start_line} out of range "
                    f"for file with {total_lines} lines."
                ),
            }
        # Ensure the preceding line ends with a newline so the insert
        # doesn't merge with an existing line.
        if idx > 0 and lines[idx - 1] and not lines[idx - 1].endswith("\n"):
            lines[idx - 1] += "\n"
        content_lines = content.splitlines(keepends=True)
        lines[idx:idx] = content_lines
        new_text = "".join(lines)
        path.write_text(new_text, encoding="utf-8")
        return {
            "success": True,
            "file_path": str(path),
            "mode": mode,
            "start_line": start_line,
            "lines_inserted": len(content_lines),
        }

    if mode in ("replace", "delete"):
        end = end_line if end_line is not None else start_line
        if idx < 0 or end > total_lines or idx >= end:
            return {
                "success": False,
                "error": (
                    f"Invalid line range [{start_line}–{end}] "
                    f"for file with {total_lines} lines."
                ),
            }
        if mode == "delete":
            removed = lines[idx:end]
            del lines[idx:end]
            new_text = "".join(lines)
            path.write_text(new_text, encoding="utf-8")
            return {
                "success": True,
                "file_path": str(path),
                "mode": mode,
                "lines_deleted": len(removed),
            }
        # replace
        content_lines = content.splitlines(keepends=True)
        lines[idx:end] = content_lines
        new_text = "".join(lines)
        path.write_text(new_text, encoding="utf-8")
        return {
            "success": True,
            "file_path": str(path),
            "mode": mode,
            "range_replaced": [start_line, end],
            "lines_added": len(content_lines),
        }

    return {
        "success": False,
        "error": (
            f"Invalid mode: '{mode}'. "
            "Use 'append', 'prepend', 'overwrite', 'insert', 'replace', or 'delete'."
        ),
    }


async def delete_file(file_path: str) -> dict:
    """Delete a file from disk."""
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        return {"success": False, "error": f"File not found: {path}"}
    path.unlink()
    return {"success": True, "file_path": str(path)}


async def bash(command: str, timeout: float | None = None) -> str:
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

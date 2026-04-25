from pathlib import Path
from typing import AsyncIterator

from minimal_harness.tool.base import StreamingTool


async def local_file_operation_handler(
    file_path: str,
    mode: str,
    content: str | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
    old_string: str | None = None,
    new_string: str | None = None,
) -> AsyncIterator[dict]:
    yield {
        "status": "progress",
        "message": f"About to perform '{mode}' on file: {file_path}",
    }
    path = Path(file_path).expanduser().resolve()

    if mode == "read":
        if not path.exists():
            yield {
                "success": False,
                "error": f"File not found: {path}",
                "message": f"File not found: {path}",
            }
            return

        raw_text = path.read_text(encoding="utf-8")
        all_lines = raw_text.splitlines(keepends=True)
        total_lines = len(all_lines)
        total_bytes = len(raw_text.encode("utf-8"))

        start = (start_line - 1) if start_line is not None else 0
        end = end_line if end_line is not None else total_lines

        if start < 0 or end > total_lines or start >= end:
            yield {
                "success": False,
                "error": (
                    f"Invalid line range [{start_line}–{end_line}] "
                    f"for file with {total_lines} lines."
                ),
                "message": (
                    f"Invalid line range [{start_line}–{end_line}] "
                    f"for file with {total_lines} lines."
                ),
            }
            return

        selected = all_lines[start:end]
        result_content = "".join(selected)
        selected_bytes = len(result_content.encode("utf-8"))
        selected_lines = end - start

        if start_line is not None or end_line is not None:
            message = (
                f"Read {selected_lines} line(s) from {path} "
                f"(lines {start + 1}–{end} of {total_lines} total, "
                f"{selected_bytes} of {total_bytes} bytes)."
            )
        else:
            message = (
                f"Read entire file: {path} — "
                f"{total_lines} line(s), {total_bytes} byte(s)."
            )

        yield {
            "success": True,
            "message": message,
            "file_path": str(path),
            "content": result_content,
            "total_lines": total_lines,
            "range": [start + 1, end],
        }
        return

    if mode == "write":
        write_content = content if content is not None else ""
        existed_before = path.exists()
        old_size = path.stat().st_size if existed_before else 0

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(write_content, encoding="utf-8")
        new_size = len(write_content.encode("utf-8"))

        if existed_before:
            message = (
                f"Overwrote existing file: {path} — "
                f"was {old_size} bytes, now {new_size} bytes."
            )
        else:
            message = f"Created new file: {path} — {new_size} bytes."

        yield {
            "success": True,
            "message": message,
            "file_path": str(path),
            "bytes_written": new_size,
            "existed_before": existed_before,
        }
        return

    if mode == "patch":
        if old_string is None:
            yield {
                "success": False,
                "error": "Mode 'patch' requires old_string.",
                "message": "Patch failed: old_string is required but was not provided.",
            }
            return

        if not path.exists():
            yield {
                "success": False,
                "error": f"File not found: {path}",
                "message": f"Patch failed: file not found — {path}",
            }
            return

        original = path.read_text(encoding="utf-8")
        occurrences = original.count(old_string)

        if occurrences == 0:
            yield {
                "success": False,
                "error": (
                    "old_string not found in file. "
                    "Make sure the string matches exactly (including whitespace)."
                ),
                "message": (
                    "Patch failed: old_string was not found in the file. "
                    "Double-check that it matches exactly, including indentation and newlines."
                ),
            }
            return

        if occurrences > 1:
            yield {
                "success": False,
                "error": (
                    f"old_string appears {occurrences} times in the file. "
                    "It must be unique to patch safely."
                ),
                "message": (
                    f"Patch failed: old_string appears {occurrences} times in the file. "
                    "It must appear exactly once to avoid ambiguous replacements."
                ),
            }
            return

        replacement = new_string if new_string is not None else ""
        patched = original.replace(old_string, replacement, 1)
        path.write_text(patched, encoding="utf-8")

        old_len = len(old_string)
        new_len = len(replacement)
        idx = original.index(old_string)
        line_before = original[:idx].count("\n") + 1

        message = (
            f"Patched {path} at line {line_before}: "
            f"replaced {old_len} character(s) with {new_len} character(s)."
        )

        yield {
            "success": True,
            "message": message,
            "file_path": str(path),
            "line": line_before,
            "old_string_length": old_len,
            "new_string_length": new_len,
        }
        return

    if mode == "delete":
        if not path.exists():
            yield {
                "success": False,
                "error": f"File not found: {path}",
                "message": f"Delete failed: file not found — {path}",
            }
            return

        size = path.stat().st_size
        path.unlink()
        yield {
            "success": True,
            "message": f"Deleted file: {path} (was {size} bytes).",
            "file_path": str(path),
            "deleted_bytes": size,
        }
        return

    yield {
        "success": False,
        "error": (
            f"Invalid mode: '{mode}'. "
            "Use 'read', 'write', 'patch', or 'delete'."
        ),
        "message": (
            f"Invalid mode: '{mode}'. "
            "Valid modes are: read, write, patch, delete."
        ),
    }


local_file_operation_tool = StreamingTool(
    name="local_file_operation",
    description=(
        "Perform local file operations:\n"
        "  read   — read file, optionally restrict to [start_line, end_line] (1-based, inclusive).\n"
        "  write  — write full content; creates or overwrites.\n"
        "  patch  — replace exactly one old_string with new_string. old_string must be unique.\n"
        "  delete — delete the file from disk."
    ),
    parameters={
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the file"},
            "mode": {
                "type": "string",
                "description": "Operation mode",
                "enum": ["read", "write", "patch", "delete"],
            },
            "content": {"type": "string", "description": "Full file content (for 'write')"},
            "start_line": {"type": "integer", "description": "1-based first line (for 'read')"},
            "end_line": {"type": "integer", "description": "1-based last line, inclusive (for 'read')"},
            "old_string": {"type": "string", "description": "Exact string to replace (for 'patch')"},
            "new_string": {"type": "string", "description": "Replacement string (for 'patch'; omit to delete old_string)"},
        },
        "required": ["file_path", "mode"],
    },
    fn=local_file_operation_handler,
)


def get_tools() -> dict[str, StreamingTool]:
    return {"local_file_operation": local_file_operation_tool}
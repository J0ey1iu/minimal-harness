from pathlib import Path
from typing import AsyncIterator

from minimal_harness.tool.base import StreamingTool, Tool
from minimal_harness.types import ToolResult


async def local_file_operation_handler(
    file_path: str,
    mode: str,
    content: str | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
    old_string: str | None = None,
    new_string: str | None = None,
) -> AsyncIterator[dict | ToolResult]:
    yield {
        "status": "progress",
        "message": f"About to perform '{mode}' on file: {file_path}",
    }
    path = Path(file_path).expanduser().resolve()

    if mode == "read":
        if not path.exists():
            yield ToolResult(
                content=f"File not found: {path}",
                meta={"error": f"File not found: {path}"},
            )
            return

        raw_text = path.read_text(encoding="utf-8")
        all_lines = raw_text.splitlines(keepends=True)
        total_lines = len(all_lines)

        start = (start_line - 1) if start_line is not None else 0
        end = end_line if end_line is not None else total_lines

        if start < 0 or end > total_lines or start >= end:
            yield ToolResult(
                content=(
                    f"Invalid line range [{start_line}–{end_line}] "
                    f"for file with {total_lines} lines."
                ),
                meta={
                    "error": (
                        f"Invalid line range [{start_line}–{end_line}] "
                        f"for file with {total_lines} lines."
                    )
                },
            )
            return

        selected = all_lines[start:end]
        result_content = "".join(selected)

        yield ToolResult(
            content=result_content,
            meta={
                "file_path": str(path),
                "total_lines": total_lines,
                "range": [start + 1, end],
            },
        )
        return

    if mode == "write":
        write_content = content if content is not None else ""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(write_content, encoding="utf-8")
        byte_count = len(write_content.encode("utf-8"))

        yield ToolResult(
            content=f"Written {byte_count} bytes to {path}",
            meta={"file_path": str(path)},
        )
        return

    if mode == "patch":
        if old_string is None:
            yield ToolResult(
                content="Patch failed: old_string is required but was not provided.",
                meta={"error": "Mode 'patch' requires old_string."},
            )
            return

        if not path.exists():
            yield ToolResult(
                content=f"Patch failed: file not found — {path}",
                meta={"error": f"File not found: {path}"},
            )
            return

        original = path.read_text(encoding="utf-8")
        occurrences = original.count(old_string)

        if occurrences == 0:
            yield ToolResult(
                content=(
                    "Patch failed: old_string was not found in the file. "
                    "Make sure it matches exactly (including whitespace)."
                ),
                meta={
                    "error": (
                        "old_string not found in file. "
                        "Make sure the string matches exactly (including whitespace)."
                    )
                },
            )
            return

        if occurrences > 1:
            yield ToolResult(
                content=(
                    f"Patch failed: old_string appears {occurrences} times in the file. "
                    "It must appear exactly once to avoid ambiguous replacements."
                ),
                meta={
                    "error": (
                        f"old_string appears {occurrences} times in the file. "
                        "It must be unique to patch safely."
                    )
                },
            )
            return

        replacement = new_string if new_string is not None else ""
        patched = original.replace(old_string, replacement, 1)
        path.write_text(patched, encoding="utf-8")

        idx = original.index(old_string)
        line_before = original[:idx].count("\n") + 1

        yield ToolResult(
            content=f"Patched {path} at line {line_before}",
            meta={"file_path": str(path)},
        )
        return

    if mode == "delete":
        if not path.exists():
            yield ToolResult(
                content=f"Delete failed: file not found — {path}",
                meta={"error": f"File not found: {path}"},
            )
            return

        path.unlink()
        yield ToolResult(
            content=f"Deleted {path}",
            meta={"file_path": str(path)},
        )
        return

    yield ToolResult(
        content=f"Invalid mode: '{mode}'. Valid modes are: read, write, patch, delete.",
        meta={
            "error": f"Invalid mode: '{mode}'. Use 'read', 'write', 'patch', or 'delete'."
        },
    )


local_file_operation_tool = StreamingTool(
    name="local_file_operation",
    display_name="File Operation",
    display_name_locale={"zh": "文件操作"},
    description_locale={"zh": "执行本地文件操作：读取、写入、替换、删除"},
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
            "content": {
                "type": "string",
                "description": "Full file content (for 'write')",
            },
            "start_line": {
                "type": "integer",
                "description": "1-based first line (for 'read')",
            },
            "end_line": {
                "type": "integer",
                "description": "1-based last line, inclusive (for 'read')",
            },
            "old_string": {
                "type": "string",
                "description": "Exact string to replace (for 'patch')",
            },
            "new_string": {
                "type": "string",
                "description": "Replacement string (for 'patch'; omit to delete old_string)",
            },
        },
        "required": ["file_path", "mode"],
    },
    fn=local_file_operation_handler,
)


def get_tools() -> dict[str, Tool]:
    return {"local_file_operation": local_file_operation_tool}

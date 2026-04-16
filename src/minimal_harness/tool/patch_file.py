from pathlib import Path
from typing import Optional

from minimal_harness.tool.base import Tool


async def patch_file_handler(
    file_path: str,
    content: str,
    mode: str = "append",
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
) -> dict:
    path = Path(file_path).expanduser().resolve()

    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = existing.splitlines(keepends=True)
    total_lines = len(lines)

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

    if start_line is None:
        return {
            "success": False,
            "error": f"Mode '{mode}' requires at least start_line.",
        }

    idx = start_line - 1

    if mode == "insert":
        if idx < 0 or idx > total_lines:
            return {
                "success": False,
                "error": (
                    f"start_line {start_line} out of range "
                    f"for file with {total_lines} lines."
                ),
            }
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


patch_file_tool = Tool(
    name="patch_file",
    description=(
        "Patch a file. Supported modes:\n"
        "  - 'append'    : Append content to the end of the file (default).\n"
        "  - 'prepend'   : Prepend content to the beginning of the file.\n"
        "  - 'overwrite' : Replace the entire file with content.\n"
        "  - 'insert'    : Insert content before start_line. Existing lines are not removed.\n"
        "  - 'replace'   : Replace lines [start_line … end_line] (1-based, inclusive) with content. "
        "end_line defaults to start_line.\n"
        "  - 'delete'    : Delete lines [start_line … end_line] (1-based, inclusive). content is ignored."
    ),
    parameters={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to patch",
            },
            "content": {
                "type": "string",
                "description": "Content to write (ignored for 'delete' mode)",
            },
            "mode": {
                "type": "string",
                "description": "Patch mode",
                "enum": [
                    "append",
                    "prepend",
                    "overwrite",
                    "insert",
                    "replace",
                    "delete",
                ],
            },
            "start_line": {
                "type": "integer",
                "description": "1-based line number (required for insert/replace/delete)",
            },
            "end_line": {
                "type": "integer",
                "description": "1-based ending line, inclusive (used by replace/delete; defaults to start_line)",
            },
        },
        "required": ["file_path", "content"],
    },
    fn=patch_file_handler,
)


def get_tools() -> dict[str, Tool]:
    return {"patch_file": patch_file_tool}

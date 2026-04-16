from pathlib import Path
from typing import Optional

from minimal_harness.tool.base import Tool


async def read_file_handler(
    file_path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
) -> dict:
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


read_file_tool = Tool(
    name="read_file",
    description="Read the contents of a file, optionally restricting to a line range (1-based, inclusive).",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to read",
            },
            "start_line": {
                "type": "integer",
                "description": "1-based first line to include (default: beginning of file)",
            },
            "end_line": {
                "type": "integer",
                "description": "1-based last line to include (default: end of file)",
            },
        },
        "required": ["file_path"],
    },
    fn=read_file_handler,
)


def get_tools() -> dict[str, Tool]:
    return {"read_file": read_file_tool}

import fnmatch
import re

from minimal_harness.tool.base import Tool


async def grep_handler(
    pattern: str,
    include: str | None = None,
    path: str | None = None,
) -> list[dict]:
    import os

    results = []
    target_path = path or "."

    if include:
        include_patterns = [p.strip() for p in include.split(",")]
    else:
        include_patterns = None

    for root, dirs, files in os.walk(target_path):
        for filename in files:
            if include_patterns:
                if not any(fnmatch.fnmatch(filename, p) for p in include_patterns):
                    continue

            filepath = os.path.join(root, filename)
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    for line_num, line in enumerate(f, 1):
                        if re.search(pattern, line):
                            results.append(
                                {
                                    "file": filepath,
                                    "line": line_num,
                                    "content": line.rstrip(),
                                }
                            )
            except (UnicodeDecodeError, IOError):
                continue

    return results


grep_tool = Tool(
    name="grep",
    description="Fast content search tool that works with any codebase size. Searches file contents using regular expressions.",
    parameters={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "The regex pattern to search for in file contents",
            },
            "include": {
                "type": "string",
                "description": "File pattern to include in the search (e.g., '*.js', '*.{ts,tsx}')",
            },
            "path": {
                "type": "string",
                "description": "The directory to search in. Defaults to the current working directory.",
            },
        },
        "required": ["pattern"],
    },
    fn=grep_handler,
)


def get_tools() -> dict[str, Tool]:
    return {"grep": grep_tool}

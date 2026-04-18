from pathlib import Path
from typing import AsyncIterator

from minimal_harness.tool.base import StreamingTool


async def create_file_handler(file_path: str, content: str = "") -> AsyncIterator[dict]:
    path = Path(file_path).expanduser().resolve()
    if path.exists():
        yield {"success": False, "error": f"File already exists: {path}"}
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    yield {"success": True, "file_path": str(path), "bytes_written": len(content)}


create_file_tool = StreamingTool(
    name="create_file",
    description="Create a new file with the given content. Fails if the file already exists.",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to create",
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file (defaults to empty string)",
            },
        },
        "required": ["file_path"],
    },
    fn=create_file_handler,
)


def get_tools() -> dict[str, StreamingTool]:
    return {"create_file": create_file_tool}

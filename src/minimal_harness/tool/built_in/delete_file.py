from pathlib import Path
from typing import AsyncIterator

from minimal_harness.tool.base import StreamingTool


async def delete_file_handler(file_path: str) -> AsyncIterator[dict]:
    yield {"status": "progress", "message": f"I'm about to delete file: {file_path}"}
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        yield {"success": False, "error": f"File not found: {path}"}
        return
    path.unlink()
    yield {"success": True, "file_path": str(path)}


delete_file_tool = StreamingTool(
    name="delete_file",
    description="Delete a file from disk.",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to delete",
            },
        },
        "required": ["file_path"],
    },
    fn=delete_file_handler,
)


def get_tools() -> dict[str, StreamingTool]:
    return {"delete_file": delete_file_tool}

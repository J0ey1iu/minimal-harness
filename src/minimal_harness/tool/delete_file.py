from pathlib import Path

from minimal_harness.tool.base import Tool


async def delete_file_handler(file_path: str) -> dict:
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        return {"success": False, "error": f"File not found: {path}"}
    path.unlink()
    return {"success": True, "file_path": str(path)}


delete_file_tool = Tool(
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


def get_tools() -> dict[str, Tool]:
    return {"delete_file": delete_file_tool}

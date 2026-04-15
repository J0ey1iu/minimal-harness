import asyncio
from pathlib import Path


async def get_weather(city: str) -> dict:
    """Simulate weather query"""
    await asyncio.sleep(0.2)
    return {"city": city, "temperature": "22°C", "condition": "Sunny"}


async def calculator(expression: str) -> dict:
    """Simple calculator"""
    result = eval(expression, {"__builtins__": {}})
    return {"expression": expression, "result": result}


async def create_file(file_path: str, content: str) -> dict:
    """Create a new file with the given content"""
    path = Path(file_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {"success": True, "file_path": str(path), "bytes_written": len(content)}


async def read_file(file_path: str) -> dict:
    """Read the contents of a file"""
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        return {"success": False, "error": f"File not found: {path}"}
    content = path.read_text(encoding="utf-8")
    return {
        "success": True,
        "file_path": str(path),
        "content": content,
        "size": len(content),
    }


async def patch_file(file_path: str, content: str, mode: str = "append") -> dict:
    """Patch a file by appending or overwriting content"""
    path = Path(file_path).expanduser().resolve()
    if mode == "overwrite":
        path.write_text(content, encoding="utf-8")
        bytes_written = len(content)
    elif mode == "append":
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        path.write_text(existing + content, encoding="utf-8")
        bytes_written = len(content)
    else:
        return {
            "success": False,
            "error": f"Invalid mode: {mode}. Use 'append' or 'overwrite'.",
        }
    return {
        "success": True,
        "file_path": str(path),
        "mode": mode,
        "bytes_written": bytes_written,
    }

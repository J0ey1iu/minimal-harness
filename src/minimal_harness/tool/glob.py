import fnmatch

from minimal_harness.tool.base import Tool


async def glob_handler(path: str = ".", pattern: str = "*") -> list[str]:
    import os

    matched = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                matched.append(os.path.join(root, name))
    return matched


glob_tool = Tool(
    name="glob",
    description="Fast file pattern matching tool that works with any codebase size. Use this to find files by name patterns.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The directory to search in. Defaults to the current working directory.",
            },
            "pattern": {
                "type": "string",
                "description": "The glob pattern to match files against (e.g., '*.js', 'src/**/*.ts').",
            },
        },
        "required": ["pattern"],
    },
    fn=glob_handler,
)


def get_tools() -> dict[str, Tool]:
    return {"glob": glob_tool}

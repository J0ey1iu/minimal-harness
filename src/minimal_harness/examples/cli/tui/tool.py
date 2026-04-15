from minimal_harness import Tool

from ..tools import calculator, create_file, get_weather, patch_file, read_file

built_in_tools = [
    Tool(
        name="get_weather",
        description="Get weather for a specified city",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        },
        fn=get_weather,
    ),
    Tool(
        name="calculator",
        description="Calculate mathematical expression",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Valid Python mathematical expression",
                },
            },
            "required": ["expression"],
        },
        fn=calculator,
    ),
    Tool(
        name="create_file",
        description="Create a new file with the given content",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to create",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file",
                },
            },
            "required": ["file_path", "content"],
        },
        fn=create_file,
    ),
    Tool(
        name="read_file",
        description="Read the contents of a file",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read",
                },
            },
            "required": ["file_path"],
        },
        fn=read_file,
    ),
    Tool(
        name="patch_file",
        description="Patch a file by appending or overwriting content",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to patch",
                },
                "content": {
                    "type": "string",
                    "description": "Content to append or overwrite",
                },
                "mode": {
                    "type": "string",
                    "description": "Mode: 'append' (default) or 'overwrite'",
                    "enum": ["append", "overwrite"],
                },
            },
            "required": ["file_path", "content"],
        },
        fn=patch_file,
    ),
]

from minimal_harness import Tool
from minimal_harness.examples.cli.tools import (
    bash,
    create_file,
    delete_file,
    patch_file,
    read_file,
)
from minimal_harness.tool.registry import ToolRegistry


def _get_builtin_tools() -> list[Tool]:
    registry = ToolRegistry.get_instance()
    for tool, fn in [
        (
            Tool(
                name="bash",
                description="Execute a shell command and return the terminal output (stdout + stderr). Compatible with Windows, Linux, and macOS.",
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to execute",
                        },
                        "timeout": {
                            "type": "number",
                            "description": "Timeout in seconds. If the command exceeds this duration, it will be killed (default: no limit)",
                        },
                    },
                    "required": ["command"],
                },
                fn=bash,
            ),
            bash,
        ),
        (
            Tool(
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
                fn=create_file,
            ),
            create_file,
        ),
        (
            Tool(
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
                fn=read_file,
            ),
            read_file,
        ),
        (
            Tool(
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
                fn=patch_file,
            ),
            patch_file,
        ),
        (
            Tool(
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
                fn=delete_file,
            ),
            delete_file,
        ),
    ]:
        registry.register(tool)
    return registry.get_all()

from minimal_harness import Tool
from minimal_harness.tool.built_in import (
    ask_user,
    bash,
    create_file,
    delete_file,
    patch_file,
    read_file,
)
from minimal_harness.tool.registry import ToolRegistry


def _get_builtin_tools() -> list[Tool]:
    registry = ToolRegistry.get_instance()
    for tool in [
        bash.bash_tool,
        create_file.create_file_tool,
        read_file.read_file_tool,
        patch_file.patch_file_tool,
        delete_file.delete_file_tool,
        ask_user.ask_user_tool,
    ]:
        registry.register(tool)
    return registry.get_all()

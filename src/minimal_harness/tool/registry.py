from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from minimal_harness.tool.base import StreamingTool, create_streaming_tool

if TYPE_CHECKING:
    from minimal_harness.tool.base import StreamingToolFunction


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, "StreamingTool"] = {}
        self._listeners: list[Callable[[], None]] = []

    def register(self, tool: "StreamingTool") -> None:
        self._tools[tool.name] = tool
        self._notify()

    def register_external_tool(
        self,
        name: str,
        description: str,
        parameters: dict,
        fn: StreamingToolFunction,
        uri: Path | str | None = None,
        **kwargs: Any,
    ) -> None:
        tool = create_streaming_tool(name, fn, description, parameters)
        if uri is not None:
            from minimal_harness.tool.wrapper import ExternalToolWrapper

            tool.fn = ExternalToolWrapper(  # type: ignore[assignment]
                original_fn=fn,
                script_path=uri,
                tool_name=name,
                tool_description=description,
                tool_params=parameters,
            )
        self.register(tool)

    def unregister(self, name: str) -> bool:
        if name in self._tools:
            del self._tools[name]
            self._notify()
            return True
        return False

    def get(self, name: str) -> "StreamingTool | None":
        return self._tools.get(name)

    def get_all(self) -> list["StreamingTool"]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def clear(self) -> None:
        self._tools.clear()
        self._notify()

    def add_listener(self, listener: Callable[[], None]) -> None:
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[], None]) -> None:
        self._listeners.remove(listener)

    def _notify(self) -> None:
        for listener in self._listeners:
            listener()

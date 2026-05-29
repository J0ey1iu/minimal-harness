"""Tool widget provider — extensible rendering for individual tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from minimal_harness.client.built_in.chat_widgets import ChatMsg


class ToolWidgetProvider(ABC):
    """Protocol for customizing how a specific tool is rendered in the TUI.

    Register providers in :class:`ToolWidgetRegistry` and pass the registry
    to ``ChatDisplay`` and ``SessionReplayer``.
    """

    @property
    @abstractmethod
    def tool_name(self) -> str:
        """The tool name this provider handles (e.g. ``"handoff"``)."""
        ...

    @abstractmethod
    def make_widget(self, tool_call: dict, msg_id: str) -> ChatMsg:
        """Create the initial widget for a tool call.

        Called when the tool call is first displayed (during LLMEnd, flush,
        or replay).  The returned widget is mounted into the chat area.
        """
        ...

    @abstractmethod
    def on_progress(self, widget: ChatMsg, chunk: Any) -> bool:
        """Handle a progress event for this tool call.

        Args:
            widget: The widget returned by :meth:`make_widget`.
            chunk: The raw progress chunk from the tool execution.

        Returns:
            ``True`` if the event was fully handled;
            ``False`` to fall back to the default progress rendering.
        """
        ...

    @abstractmethod
    def on_end(self, widget: ChatMsg, result: Any) -> bool:
        """Handle a ToolEnd event for this tool call.

        Args:
            widget: The widget returned by :meth:`make_widget`.
            result: The raw result from the tool execution.

        Returns:
            ``True`` if the event was fully handled;
            ``False`` to fall back to the default result rendering.
        """
        ...


class ToolWidgetRegistry:
    """Maps tool names to their custom :class:`ToolWidgetProvider` instances."""

    def __init__(self) -> None:
        self._providers: dict[str, ToolWidgetProvider] = {}

    def register(self, provider: ToolWidgetProvider) -> None:
        """Register a provider for its :attr:`~ToolWidgetProvider.tool_name`."""
        self._providers[provider.tool_name] = provider

    def get(self, tool_name: str) -> ToolWidgetProvider | None:
        """Return the registered provider for *tool_name*, or ``None``."""
        return self._providers.get(tool_name)

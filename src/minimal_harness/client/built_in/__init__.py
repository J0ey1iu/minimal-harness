"""Terminal UI client for the minimal-harness framework.

This module re-exports the main components for backward compatibility.
"""

from minimal_harness.client.built_in.app import TUIApp, main
from minimal_harness.client.built_in.buffer import StreamBuffer
from minimal_harness.client.built_in.config import (
    DEFAULT_CONFIG,
    collect_tools,
    load_agents_config,
    load_config,
    save_config,
)
from minimal_harness.client.built_in.constants import (
    FLUSH_INTERVAL,
    J0EY1IU_QUOTES,
    MAX_DISPLAY_LENGTH,
    THEMES,
)
from minimal_harness.client.built_in.context import AppContext
from minimal_harness.client.built_in.display import ChatDisplay
from minimal_harness.client.built_in.export_presenter import ExportPresenter
from minimal_harness.client.built_in.modals import (
    ConfigScreen,
    ConfirmScreen,
    PromptScreen,
    SessionSelectScreen,
    ToolSelectScreen,
)
from minimal_harness.client.built_in.widgets import ChatInput

__all__ = [
    "AppContext",
    "TUIApp",
    "main",
    "StreamBuffer",
    "DEFAULT_CONFIG",
    "FLUSH_INTERVAL",
    "J0EY1IU_QUOTES",
    "MAX_DISPLAY_LENGTH",
    "THEMES",
    "collect_tools",
    "load_agents_config",
    "load_config",
    "save_config",
    "ConfigScreen",
    "ConfirmScreen",
    "PromptScreen",
    "SessionSelectScreen",
    "ToolSelectScreen",
    "ChatInput",
    "ChatDisplay",
    "ExportPresenter",
]

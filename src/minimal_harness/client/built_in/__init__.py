"""Terminal UI client for the minimal-harness framework.

This module re-exports the main components for backward compatibility.
"""

from minimal_harness.client.built_in.app import TUIApp, main
from minimal_harness.client.built_in.buffer import StreamBuffer
from minimal_harness.client.built_in.config import (
    DEFAULT_CONFIG,
    J0EY1IU_QUOTES,
    THEMES,
    collect_tools,
    load_config,
    save_config,
)
from minimal_harness.client.built_in.modals import (
    ConfigScreen,
    ConfirmScreen,
    PromptScreen,
    ToolSelectScreen,
)
from minimal_harness.client.built_in.widgets import ChatInput

__all__ = [
    "TUIApp",
    "main",
    "StreamBuffer",
    "DEFAULT_CONFIG",
    "J0EY1IU_QUOTES",
    "THEMES",
    "collect_tools",
    "load_config",
    "save_config",
    "ConfigScreen",
    "ConfirmScreen",
    "PromptScreen",
    "ToolSelectScreen",
    "ChatInput",
]
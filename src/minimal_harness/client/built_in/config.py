"""Configuration helpers for the TUI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from minimal_harness.tool.base import StreamingTool
from minimal_harness.tool.built_in.bash import get_tools as get_bash_tools
from minimal_harness.tool.built_in.patch_file import get_tools as get_patch_file_tools
from minimal_harness.tool.external_loader import load_external_tools
from minimal_harness.tool.registry import ToolRegistry

CONFIG_FILE = Path.home() / ".minimal_harness" / "config.json"

DEFAULT_CONFIG: dict[str, Any] = {
    "base_url": "https://aihubmix.com/v1",
    "api_key": "",
    "model": "qwen3.5-27b",
    "system_prompt": "You are a helpful assistant.",
    "tools_path": "",
    "theme": "tokyo-night",
    "selected_tools": [],
}

THEMES = [
    "textual-dark",
    "nord",
    "gruvbox",
    "monokai",
    "tokyo-night",
    "dracula",
    "catppuccin-mocha",
    "solarized-dark",
    "solarized-light",
]

J0EY1IU_QUOTES = [
    "Here we go again...",
    "Shit happened, shit happens, and shit will happen.",
    "Fuck, I am coding at 11pm again.",
    "Work at work, work at home, rest in tomb.",
    "Use vim, otherwise you are stupid.",
    "Things are fun until they have a deadline.",
    "Knowledge graph is just another piece of shit when used in the wrong way.",
    "// no comment.",
    ":s/IDE/vim/g",
    "I can explain it to you, but I can't understand it for you.",
    "When in doubt, refactor.",
]


def load_config() -> dict[str, Any]:
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            return {
                **DEFAULT_CONFIG,
                **{k: data[k] for k in DEFAULT_CONFIG if k in data},
            }
        except (json.JSONDecodeError, OSError):
            pass
    save_config(dict(DEFAULT_CONFIG))
    return dict(DEFAULT_CONFIG)


def save_config(config: dict[str, Any]) -> None:
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def collect_tools(
    config: dict[str, Any], extra: Sequence[StreamingTool] = ()
) -> dict[str, StreamingTool]:
    if path := config.get("tools_path", "").strip():
        load_external_tools(path)
    tools: dict[str, StreamingTool] = {}
    for getter in (get_bash_tools, get_patch_file_tools):
        tools.update(getter())
    for t in ToolRegistry.get_instance().get_all():
        tools[t.name] = t
    for t in extra:
        tools.setdefault(t.name, t)
    return tools
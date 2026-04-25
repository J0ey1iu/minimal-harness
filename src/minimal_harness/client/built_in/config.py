"""Configuration helpers for the TUI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from minimal_harness.settings import Settings
from minimal_harness.tool.base import StreamingTool
from minimal_harness.tool.built_in.bash import get_tools as get_bash_tools
from minimal_harness.tool.built_in.patch_file import get_tools as get_patch_file_tools
from minimal_harness.tool.external_loader import load_external_tools
from minimal_harness.tool.registry import ToolRegistry

CONFIG_FILE = Path.home() / ".minimal_harness" / "config.json"
SYSTEM_PROMPTS_DIR = Path.home() / ".minimal_harness" / "system-prompts"

DEFAULT_CONFIG: dict[str, Any] = {
    "base_url": Settings.base_url(),
    "api_key": Settings.api_key(),
    "model": Settings.model(),
    "system_prompt": str(SYSTEM_PROMPTS_DIR / "default.md"),
    "tools_path": "",
    "theme": Settings.theme(),
    "selected_tools": [],
    "provider": "openai",
}


def load_config() -> dict[str, Any]:
    ensure_system_prompts_dir()
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


def ensure_system_prompts_dir() -> None:
    SYSTEM_PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    default_file = SYSTEM_PROMPTS_DIR / "default.md"
    if not default_file.exists():
        default_file.write_text("You are a helpful assistant.", encoding="utf-8")


def list_system_prompts() -> list[Path]:
    if not SYSTEM_PROMPTS_DIR.exists():
        return []
    return sorted(SYSTEM_PROMPTS_DIR.glob("*.md"))


def read_system_prompt(path: Path) -> str:
    if path.exists() and path.is_file():
        return path.read_text(encoding="utf-8")
    return ""


def collect_tools(
    config: dict[str, Any],
    registry: ToolRegistry,
) -> dict[str, StreamingTool]:
    if path := config.get("tools_path", "").strip():
        load_external_tools(path, registry)
    tools: dict[str, StreamingTool] = {}
    for getter in (get_bash_tools, get_patch_file_tools):
        tools.update(getter())
    for t in registry.get_all():
        tools[t.name] = t
    return tools

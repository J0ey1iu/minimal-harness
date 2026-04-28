"""Config file I/O and default configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from minimal_harness.settings import Settings

CONFIG_FILE = Path.home() / ".minimal_harness" / "config.json"

DEFAULT_CONFIG: dict[str, Any] = {
    "base_url": Settings.base_url(),
    "api_key": Settings.api_key(),
    "model": Settings.model(),
    "tools_path": "",
    "theme": Settings.theme(),
    "provider": "openai",
    "default_agent": "general_assistant",
}


def load_config() -> dict[str, Any]:
    from minimal_harness.client.built_in.config.agents import (
        ensure_agents_config,
        ensure_system_prompts_dir,
    )
    from minimal_harness.client.built_in.config.models import MODELS_FILE, save_models

    ensure_system_prompts_dir()
    ensure_agents_config()
    file_existed = CONFIG_FILE.exists()
    if file_existed:
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            data.pop("selected_tools", None)
            config = {
                **DEFAULT_CONFIG,
                **{k: data[k] for k in DEFAULT_CONFIG if k in data},
            }
        except (json.JSONDecodeError, OSError):
            config = dict(DEFAULT_CONFIG)
    else:
        config = dict(DEFAULT_CONFIG)

    if not MODELS_FILE.exists():
        model = config.get("model", "")
        if model:
            save_models([model])

    return config


def save_config(config: dict[str, Any]) -> None:
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

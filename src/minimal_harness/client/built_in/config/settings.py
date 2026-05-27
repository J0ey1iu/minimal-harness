"""Config file I/O."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from minimal_harness.client.built_in.config.defaults import DEFAULT_CONFIG


def _get_config_file() -> Path:
    from minimal_harness.client.built_in.config.paths import get_config_dir

    return get_config_dir() / "config.json"


def load_config() -> dict[str, Any]:
    from minimal_harness.client.built_in.config.agents import (
        ensure_agents_config,
        ensure_system_prompts_dir,
    )
    from minimal_harness.client.built_in.config.models import (
        get_models_file,
        save_models,
    )

    ensure_system_prompts_dir()
    ensure_agents_config()
    config_file = _get_config_file()
    file_existed = config_file.exists()
    if file_existed:
        try:
            data = json.loads(config_file.read_text(encoding="utf-8"))
            data.pop("selected_tools", None)
            config = {
                **DEFAULT_CONFIG,
                **{k: data[k] for k in DEFAULT_CONFIG if k in data},
            }
        except (json.JSONDecodeError, OSError):
            config = dict(DEFAULT_CONFIG)
    else:
        config = dict(DEFAULT_CONFIG)

    models_file = get_models_file()
    if not models_file.exists():
        model = config.get("model", "")
        if model:
            save_models([model])

    return config


def save_config(config: dict[str, Any]) -> None:
    config_file = _get_config_file()
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

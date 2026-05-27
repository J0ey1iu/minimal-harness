"""Model persistence — recently used models list."""

from __future__ import annotations

import json
from pathlib import Path


def get_models_file() -> Path:
    from minimal_harness.client.built_in.config.paths import get_config_dir

    return get_config_dir() / "models.json"


def load_models() -> list[str]:
    models_file = get_models_file()
    if models_file.exists():
        try:
            data = json.loads(models_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [str(m) for m in data if m]
        except (json.JSONDecodeError, OSError):
            pass
    return []


def save_models(models: list[str]) -> None:
    models_file = get_models_file()
    models_file.parent.mkdir(parents=True, exist_ok=True)
    models_file.write_text(
        json.dumps(models, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def add_model(model: str) -> None:
    if not model:
        return
    models = load_models()
    if model not in models:
        models.insert(0, model)
        save_models(models)

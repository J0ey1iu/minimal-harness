"""Model persistence — recently used models list."""

from __future__ import annotations

import json
from pathlib import Path

MODELS_FILE = Path.home() / ".minimal_harness" / "models.json"


def load_models() -> list[str]:
    if MODELS_FILE.exists():
        try:
            data = json.loads(MODELS_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [str(m) for m in data if m]
        except (json.JSONDecodeError, OSError):
            pass
    return []


def save_models(models: list[str]) -> None:
    MODELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    MODELS_FILE.write_text(
        json.dumps(models, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def add_model(model: str) -> None:
    if not model:
        return
    models = load_models()
    if model not in models:
        models.insert(0, model)
        save_models(models)

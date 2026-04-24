from __future__ import annotations

import os


class Settings:
    DEFAULT_BASE_URL: str = "https://aihubmix.com/v1"
    DEFAULT_MODEL: str = "qwen3.5-27b"
    DEFAULT_MAX_ITERATIONS: int = 50
    DEFAULT_THEME: str = "tokyo-night"

    @classmethod
    def base_url(cls) -> str:
        return os.environ.get("MH_BASE_URL", cls.DEFAULT_BASE_URL)

    @classmethod
    def api_key(cls) -> str:
        return os.environ.get("MH_API_KEY", "")

    @classmethod
    def model(cls) -> str:
        return os.environ.get("MH_MODEL", cls.DEFAULT_MODEL)

    @classmethod
    def max_iterations(cls) -> int:
        val = os.environ.get("MH_MAX_ITERATIONS")
        return int(val) if val else cls.DEFAULT_MAX_ITERATIONS

    @classmethod
    def theme(cls) -> str:
        return os.environ.get("MH_THEME", cls.DEFAULT_THEME)

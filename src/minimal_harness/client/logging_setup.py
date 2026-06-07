"""Logging configuration — writes logs to the resolved config dir with daily rotation."""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path

from minimal_harness.log_utils import CorrelationFilter


def _get_log_dir() -> Path:
    from minimal_harness.client.built_in.config.paths import get_config_dir

    return get_config_dir() / "log"


_FORMAT = "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_LOG_LEVEL_MAP: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# Third-party loggers that produce excessive DEBUG noise.
# We cap them at WARNING so our own DEBUG logs remain readable.
_NOISY_LOGGERS: list[str] = [
    "aiosqlite",
    "httpx",
    "httpcore",
    "openai",
    "watchfiles",
    "asyncio",
    "concurrent",
]


def _resolve_level(level: str | int | None) -> int:
    if level is None:
        return logging.INFO
    if isinstance(level, int):
        return level
    return _LOG_LEVEL_MAP.get(level.upper(), logging.INFO)


def setup_service_logging(
    level: str | int | None = None,
    log_file: str | Path | None = None,
) -> None:
    """Configure root logger for service / production mode.

    Idempotent — if root logger already has handlers, this function is a no-op.
    This allows callers who configure logging themselves to bypass this setup.

    Parameters
    ----------
    level : str or int, optional
        Log level. Falls back to ``MH_LOG_LEVEL`` env var, then ``INFO``.
    log_file : str or Path, optional
        Path to a log file. Falls back to ``MH_LOG_DIR`` env var.
        When set, a ``RotatingFileHandler`` (10 MB, 5 backups) is added.
    """
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    resolved_level = _resolve_level(
        level if level is not None else os.environ.get("MH_LOG_LEVEL", "INFO")
    )
    root_logger.setLevel(resolved_level)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    root_logger.addFilter(CorrelationFilter())

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(resolved_level)
    stderr_handler.setFormatter(logging.Formatter(_FORMAT, _DATE_FORMAT))
    root_logger.addHandler(stderr_handler)

    log_dir = log_file or os.environ.get("MH_LOG_DIR")
    if log_dir:
        path = Path(log_dir)
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)
            path = path / "service.log"
        file_handler = RotatingFileHandler(
            filename=str(path),
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(resolved_level)
        file_handler.setFormatter(logging.Formatter(_FORMAT, _DATE_FORMAT))
        root_logger.addHandler(file_handler)

    logging.getLogger("minimal_harness").info(
        "service logging initialised — level=%s log_file=%s",
        logging.getLevelName(resolved_level),
        log_dir or "(stderr only)",
    )


def setup_logging(level: str | int | None = None) -> None:
    log_dir = _get_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    resolved_level = _resolve_level(
        level if level is not None else os.environ.get("MH_LOG_LEVEL", "INFO")
    )
    root_logger.setLevel(resolved_level)
    root_logger.addFilter(CorrelationFilter())

    handler = TimedRotatingFileHandler(
        filename=log_dir / "tui.log",
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    handler.setLevel(resolved_level)
    handler.setFormatter(logging.Formatter(_FORMAT, _DATE_FORMAT))
    root_logger.addHandler(handler)

    error_handler = TimedRotatingFileHandler(
        filename=log_dir / "error.log",
        when="midnight",
        interval=1,
        backupCount=60,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(_FORMAT, _DATE_FORMAT))
    root_logger.addHandler(error_handler)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(logging.Formatter(_FORMAT, _DATE_FORMAT))
    root_logger.addHandler(stderr_handler)

    logging.getLogger(__name__).info("Logging initialised — log_dir=%s", log_dir)

"""Logging configuration — writes logs to ~/.minimal_harness/log/ with daily rotation."""

from __future__ import annotations

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

LOG_DIR = Path.home() / ".minimal_harness" / "log"

_FORMAT = "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    root_logger.setLevel(logging.DEBUG)

    handler = TimedRotatingFileHandler(
        filename=LOG_DIR / "tui.log",
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(_FORMAT, _DATE_FORMAT))
    root_logger.addHandler(handler)

    error_handler = TimedRotatingFileHandler(
        filename=LOG_DIR / "error.log",
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

    logging.getLogger(__name__).info("Logging initialised — log_dir=%s", LOG_DIR)

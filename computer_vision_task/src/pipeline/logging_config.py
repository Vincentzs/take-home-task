"""Structured (JSONL) logging to a file plus human-readable console output.

Used only by the runner / CLI; stages do not log directly.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, ClassVar


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.time(),
            "level": record.levelname,
            "stage": getattr(record, "stage", None),
            "page": getattr(record, "page", None),
            "code": getattr(record, "code", None),
            "message": record.getMessage(),
            "duration_ms": getattr(record, "duration_ms", None),
        }
        if record.exc_info is not None:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


class HumanFormatter(logging.Formatter):
    LEVEL_PREFIX: ClassVar[dict[str, str]] = {
        "DEBUG": "·",
        "INFO": "✓",
        "WARNING": "!",
        "ERROR": "✗",
    }

    def format(self, record: logging.LogRecord) -> str:
        prefix = self.LEVEL_PREFIX.get(record.levelname, "?")
        stage = getattr(record, "stage", None)
        page = getattr(record, "page", None)
        loc = ""
        if stage is not None:
            loc = f" [{stage}"
            if page is not None:
                loc += f" p{page}"
            loc += "]"
        return f"{prefix}{loc} {record.getMessage()}"


def configure(log_path: Path, console_level: int = logging.INFO, debug: bool = False) -> None:
    """Configure the root logger with file (JSONL) + console (human) handlers."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("pipeline")
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    file_handler.setFormatter(JsonLineFormatter())
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if debug else console_level)
    console_handler.setFormatter(HumanFormatter())
    root.addHandler(console_handler)


def event(
    stage: str,
    message: str,
    *,
    page: int | None = None,
    code: str | None = None,
    duration_ms: float | None = None,
    level: int = logging.INFO,
) -> None:
    logger = logging.getLogger("pipeline")
    logger.log(
        level,
        message,
        extra={"stage": stage, "page": page, "code": code, "duration_ms": duration_ms},
    )

from __future__ import annotations

"""Simple JSON logging setup."""

import json
import logging
from datetime import datetime
from pathlib import Path


class JsonFormatter(logging.Formatter):
    """Format log records as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        data = {
            "time": self.formatTime(record, "%Y-%m-%d %H:%M:%S"),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        return json.dumps(data, ensure_ascii=False)


def setup_logger() -> Path:
    """Configure root logger to write a JSON log file."""
    log_dir = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{datetime.now():%y%m%d_%H%M}.json"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(JsonFormatter())
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        root.addHandler(file_handler)
        root.addHandler(stream_handler)
    return log_file

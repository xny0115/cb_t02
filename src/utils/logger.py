from __future__ import annotations

"""Application logging utilities."""

import logging
from logging.handlers import RotatingFileHandler
from logging import Formatter
from pathlib import Path
from datetime import datetime
import json


LOG_DIR = Path("logs")
LOG_PATH = LOG_DIR / f"{datetime.now():%y%m%d_%H%M}.json"


class JsonFormatter(Formatter):
    """Simple JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        data = {
            "time": self.formatTime(record, "%Y-%m-%d %H:%M:%S"),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if record.exc_info:
            data["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False)


def setup_logger() -> Path:
    """Configure root logger."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        LOG_PATH,
        maxBytes=1_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    formatter = JsonFormatter()
    handler.setFormatter(formatter)
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.addHandler(handler)
    root.addHandler(stream)
    return LOG_PATH


def log_gpu_memory() -> None:
    """Debug log of current GPU memory usage."""
    try:
        import torch

        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1024**2
            logging.getLogger(__name__).debug("GPU memory %.1fMB", mem)
    except Exception:
        logging.getLogger(__name__).debug("GPU memory check failed")

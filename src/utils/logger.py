from __future__ import annotations

"""Application logging utilities."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOG_DIR = Path("logs")
LOG_PATH = LOG_DIR / "app.log"
FMT = "[%(asctime)s] %(levelname)s %(module)s:%(funcName)s - %(message)s"


def setup_logger() -> Path:
    """Configure root logger."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        LOG_PATH,
        maxBytes=1_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    formatter = logging.Formatter(FMT)
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

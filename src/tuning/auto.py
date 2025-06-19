"""Automatic hyperparameter tuning utilities."""

from __future__ import annotations

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional
    psutil = None
import torch
from ..config import Config


class AutoTuner:
    """Estimate reasonable hyperparameters based on hardware and data."""

    def __init__(self, dataset_size: int) -> None:
        self.dataset_size = dataset_size
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        if psutil is not None:
            self.mem_total = psutil.virtual_memory().total // (1024**3)
        else:
            self.mem_total = 0

    def suggest_config(self) -> Config:
        batch_size = 4
        if self.device == "cuda":
            batch_size = min(32, max(4, self.mem_total // 4))
        lr = 3e-4 if self.dataset_size > 1000 else 1e-3
        epochs = 10 if self.dataset_size > 500 else 20
        cfg = Config(
            batch_size=batch_size,
            learning_rate=lr,
            num_epochs=epochs,
        )
        return cfg

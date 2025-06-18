"""Automatic hyperparameter tuning utilities."""

from __future__ import annotations

import psutil
import torch


class AutoTuner:
    """Estimate reasonable hyperparameters based on hardware and data."""

    def __init__(self, dataset_size: int) -> None:
        self.dataset_size = dataset_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mem_total = psutil.virtual_memory().total // (1024**3)

    def suggest(self) -> dict:
        batch_size = 4
        if self.device == "cuda":
            batch_size = min(32, max(4, self.mem_total // 4))
        lr = 3e-4 if self.dataset_size > 1000 else 1e-3
        epochs = 10 if self.dataset_size > 500 else 20
        return {
            "batch_size": batch_size,
            "learning_rate": lr,
            "epochs": epochs,
            "device": self.device,
        }

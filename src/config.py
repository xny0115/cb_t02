from __future__ import annotations

"""Application configuration management."""

from pathlib import Path
from typing import Any, Dict
import json
from dataclasses import dataclass

from .utils.persist import load_json

CONFIG_PATH = Path("configs/current.json")

DEFAULT_CONFIG: Dict[str, Any] = {
    "epochs": 20,
    "batch_size": 32,
    "lr": 1e-3,
    "dropout_ratio": 0.1,
    "warmup_steps": 0,
    "max_sequence_length": 128,
    "num_heads": 4,
    "num_encoder_layers": 2,
    "num_decoder_layers": 2,
    "model_dim": 128,
    "ff_dim": 512,
    "top_k": 10,
    "top_p": 0.9,
    "no_repeat_ngram": 2,
    "temperature": 0.7,
    "early_stopping_patience": 5,
    "verbose": False,
}


@dataclass
class Config:
    num_epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    dropout_ratio: float = 0.1
    warmup_steps: int = 0
    max_sequence_length: int = 128
    num_heads: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    model_dim: int = 128
    ff_dim: int = 512
    top_k: int = 10
    top_p: float = 0.9
    no_repeat_ngram: int = 2
    temperature: float = 0.7
    early_stopping_patience: int = 5
    verbose: bool = False


def load_config() -> Dict[str, Any]:
    """Load configuration from disk."""
    data = load_json(CONFIG_PATH)
    cfg = DEFAULT_CONFIG.copy()
    if data:
        cfg.update(data)
    return cfg


def save_config(cfg: Dict[str, Any]) -> None:
    """Persist configuration to disk."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(CONFIG_PATH, "w"), indent=2)

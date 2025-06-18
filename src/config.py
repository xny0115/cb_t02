from __future__ import annotations

"""Application configuration management."""

from dataclasses import dataclass, asdict
from pathlib import Path
import json

CONFIG_PATH = Path("configs/config.json")


@dataclass
class Config:
    """Model and training configuration."""

    num_epochs: int = 10
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
    temperature: float = 0.7
    early_stopping_patience: int = 5


def load_config() -> Config:
    """Load configuration from disk."""
    if CONFIG_PATH.exists():
        data = json.loads(CONFIG_PATH.read_text("utf-8"))
        return Config(**data)
    return Config()


def save_config(cfg: Config) -> None:
    """Persist configuration to disk."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))

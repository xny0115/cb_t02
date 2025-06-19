from __future__ import annotations
"""Helper utilities for ChatbotService."""
from typing import Any, Dict
from ..config import Config

_CFG_MAP = {
    'num_epochs': ('epochs', 20),
    'batch_size': ('batch_size', 32),
    'learning_rate': ('lr', 1e-3),
    'dropout_ratio': ('dropout_ratio', 0.1),
    'warmup_steps': ('warmup_steps', 0),
    'max_sequence_length': ('max_sequence_length', 128),
    'num_heads': ('num_heads', 4),
    'num_encoder_layers': ('num_encoder_layers', 2),
    'num_decoder_layers': ('num_decoder_layers', 2),
    'model_dim': ('model_dim', 128),
    'ff_dim': ('ff_dim', 512),
    'top_k': ('top_k', 10),
    'temperature': ('temperature', 0.7),
    'early_stopping_patience': ('early_stopping_patience', 5),
    'verbose': ('verbose', False),
}

def to_config(data: Dict[str, Any]) -> Config:
    """Convert config dict to Config dataclass."""
    return Config(**{k: data.get(src, d) for k, (src, d) in _CFG_MAP.items()})


def simple_fallback(prompt: str) -> str:
    """Return short apology."""
    return "죄송합니다. 답변을 준비하지 못했습니다." if "?" in prompt else "답변을 생성하지 못했습니다."

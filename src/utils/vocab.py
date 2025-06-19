from __future__ import annotations

"""Vocabulary helpers."""

from pathlib import Path
from typing import Iterable

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional
    torch = None

from ..data.loader import QADataset


def build_vocab(dataset: QADataset) -> dict[str, int]:
    """Create vocabulary dictionary."""
    tokens: set[str] = set()
    for q, a in dataset:
        tokens.update(q.split())
        tokens.update(a.split())
    vocab = {t: i + 2 for i, t in enumerate(sorted(tokens))}
    vocab["<pad>"] = 0
    vocab["<eos>"] = 1
    return vocab


def encode(text: str, vocab: dict[str, int]):
    """Convert text to tensor of token ids or list if torch missing."""
    ids = [vocab.get(t, 0) for t in text.split()] + [vocab["<eos>"]]
    if torch is None:
        return ids
    return torch.tensor(ids, dtype=torch.long)

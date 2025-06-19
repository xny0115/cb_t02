from __future__ import annotations

"""Vocabulary helpers."""

from pathlib import Path
from typing import Iterable

import torch

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
    """Convert text to tensor of token ids."""
    ids = [vocab.get(t, 0) for t in text.split()] + [vocab["<eos>"]]
    return torch.tensor(ids, dtype=torch.long)


def decode(ids: Iterable[int], vocab: dict[str, int]) -> str:
    """Convert tensor ids back to string."""
    rev = {v: k for k, v in vocab.items()}
    words = []
    for i in ids:
        if i == vocab["<eos>"]:
            break
        token = rev.get(int(i), "")
        if token and token not in ("<pad>", "<eos>"):
            words.append(token)
    return " ".join(words)


class Tokenizer:
    """Tiny tokenizer built from a dataset."""

    def __init__(self, vocab: dict[str, int]) -> None:
        self.vocab = vocab

    def encode(self, text: str) -> torch.Tensor:
        return encode(text, self.vocab)

    def decode(self, ids: Iterable[int]) -> str:
        return decode(ids, self.vocab)

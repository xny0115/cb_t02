from __future__ import annotations

"""Vocabulary helpers and encoder utilities."""

from typing import Iterable, List, Dict

import torch

from ..data.morph import analyze
from ..data.loader import QADataset


_VALID_POS_PREFIX = ("N", "V", "XR")


def _filter_tokens(tokens: List[Dict[str, str]]) -> List[str]:
    """Return lemmas filtered by POS prefixes."""
    lemmas: List[str] = []
    for tok in tokens:
        pos = tok.get("pos", "")
        lemma = tok.get("lemma") or tok.get("text", "")
        if not lemma:
            continue
        if any(pos.startswith(p) for p in _VALID_POS_PREFIX):
            lemmas.append(lemma)
    return lemmas


def build_vocab(dataset: QADataset) -> dict[str, int]:
    """Create vocabulary with domain and concept tokens."""
    tokens: set[str] = set()
    for pair in dataset.pairs:
        tokens.update(_filter_tokens(pair.tokens_q))
        tokens.update(_filter_tokens(pair.tokens_a))
        tokens.update(pair.concepts)
        tokens.add(f"[도메인:{pair.domain}]")
    vocab = {tok: i + 2 for i, tok in enumerate(sorted(tokens))}
    vocab["<pad>"] = 0
    vocab["<eos>"] = 1
    return vocab


def encode_tokens(
    tokens: List[Dict[str, str]],
    concepts: List[str],
    domain: str,
    vocab: dict[str, int],
) -> torch.Tensor:
    """Encode tokens with domain and concepts using ``vocab``."""

    words = _filter_tokens(tokens)
    for c in concepts:
        if c not in words:
            words.append(c)
    prefix = f"[도메인:{domain}]"
    seq = [prefix] + words
    ids = [vocab.get(w, 0) for w in seq] + [vocab["<eos>"]]
    return torch.tensor(ids, dtype=torch.long)


def encode(text: str, vocab: dict[str, int]) -> torch.Tensor:
    """Backward compatible text encoder."""
    tokens = analyze(text)
    return encode_tokens(tokens, [], "", vocab)


def decode(ids: Iterable[int], vocab: dict[str, int]) -> str:
    """Decode ids back to whitespace joined string."""
    rev = {v: k for k, v in vocab.items()}
    words: List[str] = []
    for i in ids:
        if i == vocab["<eos>"]:
            break
        token = rev.get(int(i), "")
        if token and token not in ("<pad>", "<eos>"):
            words.append(token)
    return " ".join(words)


class Tokenizer:
    """Tiny tokenizer with optional rich token input."""

    def __init__(self, vocab: dict[str, int]) -> None:
        self.vocab = vocab

    def encode(self, text: str) -> torch.Tensor:
        return encode(text, self.vocab)

    def encode_tokens(
        self, tokens: List[Dict[str, str]], concepts: List[str], domain: str
    ) -> torch.Tensor:
        return encode_tokens(tokens, concepts, domain, self.vocab)

    def decode(self, ids: Iterable[int]) -> str:
        return decode(ids, self.vocab)


from __future__ import annotations

"""Morphological analysis utilities using KoNLPy Komoran."""

import logging
from typing import List, Dict

try:
    from konlpy.tag import Komoran

    _komoran = Komoran()
except Exception as exc:  # pragma: no cover - best effort
    _komoran = None
    logging.getLogger(__name__).warning("Komoran init failed: %s", exc)


def analyze(text: str) -> List[Dict[str, str]]:
    """Return list of morphological tokens for ``text``."""
    if not text:
        return []
    if _komoran is None:
        return [{"text": t, "lemma": t, "pos": "UNK"} for t in text.split()]
    tokens: List[Dict[str, str]] = []
    for word, pos in _komoran.pos(text):
        tokens.append({"text": word, "lemma": word, "pos": pos})
    return tokens


def to_str(tokens: List[Dict[str, str]]) -> str:
    """Convert token dicts to whitespace joined string."""
    return " ".join(t.get("lemma") or t.get("text", "") for t in tokens if t)

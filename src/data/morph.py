from __future__ import annotations

"""Morphological analysis utilities using KoNLPy Komoran.

This module normalizes text with ``soynlp`` before analysis and
applies simple post-processing to remove punctuation tokens.
"""

import logging
from typing import List, Dict

from soynlp.normalizer import emoticon_normalize, repeat_normalize
from soynlp.tokenizer import RegexTokenizer

try:
    from konlpy.tag import Komoran

    _komoran = Komoran()
except Exception as exc:  # pragma: no cover - best effort
    _komoran = None
    logging.getLogger(__name__).warning("Komoran init failed: %s", exc)

_regex_tokenizer = RegexTokenizer()


def _normalize(text: str) -> str:
    """Return text normalized with ``soynlp`` utilities."""
    text = emoticon_normalize(text, num_repeats=2)
    return repeat_normalize(text, num_repeats=2)


def _postprocess(tokens: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Remove punctuation tokens."""
    skip = {"SP", "SSO", "SSC", "SC", "SE", "SF", "SY"}
    return [t for t in tokens if t.get("pos") not in skip]


def analyze(text: str) -> List[Dict[str, str]]:
    """Return list of morphological tokens for ``text``."""
    if not text:
        return []
    text = _normalize(text)
    parts: List[tuple[str, str]]
    if _komoran is not None:
        try:
            parts = _komoran.pos(text)
        except Exception as exc:  # pragma: no cover - best effort
            logging.getLogger(__name__).warning("komoran error: %s", exc)
            parts = [(t, "UNK") for t in _regex_tokenizer.tokenize(text, flatten=True)]
    else:
        parts = [(t, "UNK") for t in _regex_tokenizer.tokenize(text, flatten=True)]

    tokens: List[Dict[str, str]] = [
        {"text": w, "lemma": w, "pos": p} for w, p in parts
    ]
    return _postprocess(tokens)


def to_str(tokens: List[Dict[str, str]]) -> str:
    """Convert token dicts to whitespace joined string."""
    return " ".join(t.get("lemma") or t.get("text", "") for t in tokens if t)

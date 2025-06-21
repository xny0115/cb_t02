from __future__ import annotations

"""Token-to-sentence restorer based on simple POS rules."""

from typing import Dict, List


def _has_batchim(word: str) -> bool:
    """Return True if ``word`` ends with a final consonant."""
    if not word:
        return False
    code = ord(word[-1])
    return 0xAC00 <= code <= 0xD7A3 and (code - 0xAC00) % 28 != 0


def _choose_particle(word: str, josa_type: str) -> str:
    """Return the particle variant for ``word`` based on ``josa_type``."""
    if josa_type == "JKO":
        return "을" if _has_batchim(word) else "를"
    return "은" if _has_batchim(word) else "는"


def restore_sentence(tokens: List[Dict[str, str]]) -> str:
    """Assemble a sentence from morphological tokens."""
    words: List[str] = []
    n = len(tokens)
    for idx, tok in enumerate(tokens):
        lemma = tok.get("lemma", "")
        pos = tok.get("pos", "")
        next_pos = tokens[idx + 1]["pos"] if idx + 1 < n else None

        if pos == "NNG":
            particle_type = "JKO" if next_pos == "VV" else "JKS"
            particle = _choose_particle(lemma, particle_type)
            words.append(lemma + particle)
        elif pos in {"VV", "VA"}:
            words.append(lemma + "다")
        else:
            continue

    return " ".join(words)


__all__ = ["restore_sentence"]

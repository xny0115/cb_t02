from __future__ import annotations

"""Simple token-based sentence restorer."""

from typing import List, Dict


def _has_batchim(word: str) -> bool:
    """Return True if ``word`` ends with a final consonant."""
    if not word:
        return False
    code = ord(word[-1])
    if 0xAC00 <= code <= 0xD7A3:
        return (code - 0xAC00) % 28 > 0
    return False


def _ends_with_ao(word: str) -> bool:
    """Return True if ``word`` ends with ``ㅏ`` or ``ㅗ`` class vowel."""
    if not word:
        return False
    code = ord(word[-1])
    if 0xAC00 <= code <= 0xD7A3:
        idx = code - 0xAC00
        jong = idx % 28
        jung = ((idx - jong) // 28) % 21
        return jung in {0, 2, 8, 12}
    return False


def _should_attach(prev: str, curr: str) -> bool:
    """Return True if tokens should be concatenated."""
    if prev.startswith("N") and curr.startswith("J"):
        return True
    if prev.startswith("V") and curr.startswith(("E", "X", "V")):
        return True
    if prev.startswith("X") and curr.startswith(("J", "E", "V")):
        return True
    return False


def _clean_tokens(tokens: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Remove repeated tokens and long noun sequences."""
    res: List[Dict[str, str]] = []
    prev_lemma = ""
    noun_run = 0
    for tok in tokens:
        lemma = tok.get("lemma") or tok.get("text", "")
        pos = tok.get("pos", "")
        if not lemma:
            continue
        if lemma == prev_lemma:
            continue
        if pos.startswith("N"):
            noun_run += 1
            if noun_run > 2:
                continue
        else:
            noun_run = 0
        res.append({"lemma": lemma, "pos": pos})
        prev_lemma = lemma
    return res


def _combine(tokens: List[Dict[str, str]]) -> List[str]:
    """Combine tokens into Eojeol words."""
    if not tokens:
        return []
    words: List[str] = []
    prev = tokens[0]
    buffer = prev["lemma"]
    for tok in tokens[1:]:
        lemma = tok["lemma"]
        pos = tok["pos"]
        if _should_attach(prev["pos"], pos):
            buffer += lemma
        else:
            words.append(buffer)
            buffer = lemma
        prev = tok
    words.append(buffer)
    return words


_TEMPLATE_MAP = {
    "meat": "{N} {V} {A}.",
    "history": "역사적으로 {N} {V} {A}.",
    "science": "과학적으로 {N} {V} {A}.",
    "default": "{N} {V} {A}.",
}


def _select_template(domain: str, concepts: List[str]) -> str:
    if domain == "고기" or "고기" in concepts:
        return _TEMPLATE_MAP["meat"]
    if domain == "역사" or "역사" in concepts:
        return _TEMPLATE_MAP["history"]
    if domain == "과학" or "과학" in concepts:
        return _TEMPLATE_MAP["science"]
    return _TEMPLATE_MAP["default"]


def restore_sentence(tokens: List[Dict[str, str]], concepts: List[str] | None = None, domain: str | None = None) -> str:
    """Return restored sentence from morphological tokens."""
    if not tokens:
        return ""
    cleaned = _clean_tokens(tokens)
    words = _combine(cleaned)
    subject = next((t["lemma"] for t in cleaned if t["pos"].startswith("N")), "")
    verb = next((t["lemma"] for t in cleaned if t["pos"].startswith("V")), "")
    adj = next((t["lemma"] for t in cleaned if t["pos"].startswith("VA")), "")
    if verb == subject:
        verb = next((t["lemma"] for t in cleaned if t["pos"].startswith("V") and t["lemma"] != subject), verb)
    if adj in {subject, verb}:
        adj = next(
            (t["lemma"] for t in cleaned if t["pos"].startswith("VA") and t["lemma"] not in {subject, verb}),
            adj,
        )
    if not subject:
        subject = words[0]
    if not verb:
        verb = "하"
    if not adj:
        adj = "좋"
    particle = "은" if _has_batchim(subject) else "는"
    if verb == "하":
        verb_part = "해야"
    else:
        verb_part = verb + ("아야" if _ends_with_ao(verb) else "어야")
    adj_part = adj + "다"
    template = _select_template(domain or "", concepts or [])
    sentence = template.format(N=subject + particle, V=verb_part, A=adj_part)
    return sentence


__all__ = ["restore_sentence"]

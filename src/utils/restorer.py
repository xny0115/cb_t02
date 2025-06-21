from __future__ import annotations

"""Token-to-sentence restorer based on simple POS rules."""

from typing import Dict, List

TEMPLATES: Dict[str, List[Dict[str, List[str] | str]]] = {
    "음식": [
        {"condition": ["단백질", "고기"], "template": "{X}에는 단백질이 많아"},
        {"condition": ["채식"], "template": "{X}는 채식 메뉴야"},
    ],
    "여행": [
        {"condition": ["휴양"], "template": "{X}에서 휴양을 즐길 수 있어"},
    ],
    "건강": [
        {"condition": ["운동"], "template": "{X}는 운동에 도움이 돼"},
    ],
    "기술": [
        {"condition": ["인공지능"], "template": "{X}에 인공지능이 적용돼"},
    ],
    "문화": [
        {"condition": ["역사"], "template": "{X}는 역사적 의미가 있어"},
    ],
}


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


def restore_with_template(
    tokens: List[Dict[str, str]], concepts: List[str], domain: str
) -> str:
    """Return a sentence assembled using domain templates."""

    first_nng = next((t.get("lemma", "") for t in tokens if t.get("pos") == "NNG"), "")
    for item in TEMPLATES.get(domain, []):
        cond = set(item.get("condition", []))
        if cond & set(concepts):
            return item["template"].format(X=first_nng)
    return "적절한 문장을 찾을 수 없습니다."


__all__ = ["restore_sentence", "restore_with_template"]

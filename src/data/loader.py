"""Dataset loading utilities."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Sample:
    """Raw QA sample as loaded from JSON."""

    question: dict
    answer: dict
    concepts: List[str]
    domain: str


def load_all(path: Path) -> List[Sample]:
    """Load all JSON files under ``path`` and merge into a list."""

    files = sorted(path.glob("*.json"))
    samples: List[Sample] = []
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for item in json.load(f):
                samples.append(
                    Sample(
                        question=item.get("question", {}),
                        answer=item.get("answer", {}),
                        concepts=item.get("concepts", []),
                        domain=item.get("domain", ""),
                    )
                )
    logger.info("loaded %d samples from %d files", len(samples), len(files))
    return samples

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional
    pd = None


@dataclass
class QAPair:
    """Simple question-answer pair."""

    question: str
    answer: str
    tokens_q: List[dict]
    tokens_a: List[dict]
    concepts: List[str]
    domain: str


class QADataset:
    """Dataset loader for JSON QA pairs.

    This class accepts either a single JSON file or a directory containing
    multiple JSON files. All files are merged into one dataset.
    """

    def __init__(self, path: Path) -> None:
        self.paths: List[Path] = []
        if path.is_file():
            self.paths = [path]
        elif path.is_dir():
            self.paths = sorted(path.glob("*.json"))
        else:
            raise FileNotFoundError(path)

        self.pairs: List[QAPair] = []
        self.load()

    def load(self) -> None:
        for file_path in self.paths:
            if not file_path.exists():
                raise FileNotFoundError(file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                self.pairs.append(
                    QAPair(
                        question=item["question"]["text"],
                        answer=item["answer"]["text"],
                        tokens_q=item["question"].get("tokens", []),
                        tokens_a=item["answer"].get("tokens", []),
                        concepts=item.get("concepts", []),
                        domain=item.get("domain", ""),
                    )
                )
        if len(self.pairs) < 100:
            print(
                f"Warning: dataset has only {len(self.pairs)} pairs (<100)."
            )

    def to_dataframe(self) -> Any:
        """Convert dataset to pandas DataFrame or fallback object."""

        records = [
            {
                "question": p.question,
                "answer": p.answer,
                "concepts": ",".join(p.concepts),
                "domain": p.domain,
            }
            for p in self.pairs
        ]
        if pd is None:
            class Dummy:
                def __init__(self, rec):
                    self._rec = rec

                @property
                def empty(self) -> bool:
                    return not self._rec

            return Dummy(records)
        return pd.DataFrame(records)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        pair = self.pairs[idx]
        return pair.question, pair.answer

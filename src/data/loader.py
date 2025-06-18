"""Dataset loading utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd


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

    def to_dataframe(self) -> pd.DataFrame:
        """Convert dataset to pandas DataFrame."""

        records = [
            {
                "question": p.question,
                "answer": p.answer,
                "concepts": ",".join(p.concepts),
                "domain": p.domain,
            }
            for p in self.pairs
        ]
        return pd.DataFrame(records)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        pair = self.pairs[idx]
        return pair.question, pair.answer

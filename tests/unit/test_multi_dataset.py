from pathlib import Path
import json
from src.data.loader import load_all


def test_multi_dataset(tmp_path):
    data_dir = tmp_path / "datas"
    data_dir.mkdir()
    for i in range(3):
        fp = data_dir / f"{i}.json"
        with open(fp, "w", encoding="utf-8") as f:
            json.dump([
                {
                    "question": {"text": f"q{i}"},
                    "answer": {"text": "a"},
                }
            ], f)
    samples = load_all(data_dir)
    assert len(samples) == 3


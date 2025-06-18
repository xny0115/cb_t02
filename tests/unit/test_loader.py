import pytest
from pathlib import Path

from src.data import QADataset


def test_dataset_loading():
    path = Path("datas") / "qa_test_10.json"
    ds = QADataset(path)
    assert len(ds) == 9
    df = ds.to_dataframe()
    assert not df.empty

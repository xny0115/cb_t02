import pytest
from pathlib import Path

from src.data import QADataset


def test_dataset_loading_dir():
    path = Path("datas")
    ds = QADataset(path)
    assert len(ds) >= 110
    df = ds.to_dataframe()
    assert not df.empty

import json
import logging
from pathlib import Path
from src.service.core import ChatbotService


def test_restart_training(tmp_path, caplog):
    svc = ChatbotService()
    svc.model_path = tmp_path / "current.pth"
    meta = svc.model_path.with_suffix(".meta.json")
    svc.update_config({"epochs": 1})
    svc.train(Path("datas"))
    assert meta.exists()
    first = json.load(open(meta))
    assert first["epochs_done"] == 1
    caplog.set_level(logging.INFO)
    svc.train(Path("datas"))
    again = json.load(open(meta))
    assert again["epochs_done"] == 1
    assert any("retrain requested" in r.message for r in caplog.records)

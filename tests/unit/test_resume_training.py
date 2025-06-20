import json
import logging
from src.service.core import ChatbotService
from pathlib import Path


def test_resume_training(tmp_path, caplog):
    svc = ChatbotService()
    svc.model_path = tmp_path / "current.pth"
    meta = svc.model_path.with_suffix(".meta.json")
    svc.update_config({"epochs": 2})
    svc.train(Path("datas"))
    assert meta.exists()
    m = json.load(open(meta))
    assert m["last_epoch"] == 1
    svc.update_config({"epochs": 4})
    caplog.set_level(logging.INFO)
    svc.train(Path("datas"))
    assert any("resume training from epoch 2" in r.message for r in caplog.records)

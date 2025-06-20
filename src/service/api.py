from __future__ import annotations
import json
from pathlib import Path
from fastapi import FastAPI
from .core import ChatbotService

app = FastAPI()
svc = ChatbotService()
STATUS_FILE = Path("training_status.json")

@app.get("/training/status")
def training_status() -> dict:
    if STATUS_FILE.exists():
        return json.loads(STATUS_FILE.read_text())
    return {"running": False, "current_epoch": 0, "total_epochs": 0, "loss": 0.0}

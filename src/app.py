from __future__ import annotations

"""FastAPI server providing training and inference APIs."""

from pathlib import Path
from threading import Thread, Lock
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import Config, load_config, save_config
from .training import train, infer

app = FastAPI()
_state_lock = Lock()
_status: Dict[str, Any] = {"running": False, "message": "idle", "log": []}
_cfg = load_config()


class TrainRequest(BaseModel):
    data_path: str = "."


class InferRequest(BaseModel):
    question: str


@app.get("/config")
def get_config() -> Dict[str, Any]:
    """Return current configuration."""
    return {"success": True, "data": _cfg.__dict__, "error": None}


@app.put("/config")
def update_config(conf: Dict[str, Any]) -> Dict[str, Any]:
    """Update configuration and persist to disk."""
    global _cfg
    for k, v in conf.items():
        if hasattr(_cfg, k):
            setattr(_cfg, k, v)
    save_config(_cfg)
    return {"success": True, "data": _cfg.__dict__, "error": None}


def _train_thread(path: Path) -> None:
    def cb(epoch: int, total: int, loss: float) -> None:
        with _state_lock:
            _status["message"] = f"{epoch}/{total} loss={loss:.4f}"
            _status["log"].append(_status["message"])

    try:
        train(path, _cfg, progress_cb=cb)
        with _state_lock:
            _status["message"] = "done"
    except Exception as exc:  # pragma: no cover - high level
        with _state_lock:
            _status["message"] = f"error: {exc}"
    finally:
        with _state_lock:
            _status["running"] = False


@app.post("/train")
def start_train(req: TrainRequest) -> Dict[str, Any]:
    """Launch training in a background thread."""
    with _state_lock:
        if _status["running"]:
            raise HTTPException(status_code=409, detail="Training already running")
        _status.update({"running": True, "message": "starting", "log": []})
    thread = Thread(target=_train_thread, args=(Path("datas") / req.data_path,), daemon=True)
    thread.start()
    return {"success": True, "data": None, "error": None}


@app.get("/status")
def get_status() -> Dict[str, Any]:
    """Return training status and logs."""
    with _state_lock:
        return {"success": True, "data": _status.copy(), "error": None}


@app.post("/infer")
def inference(req: InferRequest) -> Dict[str, Any]:
    """Return model answer for a question."""
    try:
        answer = infer(req.question, _cfg)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"success": True, "data": {"answer": answer}, "error": None}

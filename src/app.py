from __future__ import annotations

"""FastAPI server providing training and inference APIs."""

from pathlib import Path
from threading import Thread, Lock
from typing import Any, Dict

from .config import load_config, save_config
from .training import train, infer
from .data.loader import QADataset


class Backend:
    """Simple in-process API for the webview UI."""

    def __init__(self) -> None:
        self._cfg = load_config()
        self._state_lock = Lock()
        self._status: Dict[str, Any] = {
            "running": False,
            "message": "idle",
            "log": [],
        }

    def get_config(self) -> Dict[str, Any]:
        """Return current configuration."""
        return {"success": True, "data": self._cfg.__dict__, "error": None}

    def update_config(self, conf: Dict[str, Any]) -> Dict[str, Any]:
        """Update configuration and persist to disk."""
        for key, val in conf.items():
            if hasattr(self._cfg, key):
                setattr(self._cfg, key, val)
        save_config(self._cfg)
        return {"success": True, "data": self._cfg.__dict__, "error": None}

    def _train_thread(self, path: Path) -> None:
        def cb(epoch: int, total: int, loss: float) -> None:
            with self._state_lock:
                self._status["message"] = f"{epoch}/{total} loss={loss:.4f}"
                self._status["log"].append(self._status["message"])

        try:
            train(path, self._cfg, progress_cb=cb)
            with self._state_lock:
                self._status["message"] = "done"
        except Exception as exc:  # pragma: no cover - high level
            with self._state_lock:
                self._status["message"] = f"error: {exc}"
        finally:
            with self._state_lock:
                self._status["running"] = False

    def start_train(self, data_path: str = ".") -> Dict[str, Any]:
        """Launch training in a background thread."""
        with self._state_lock:
            if self._status["running"]:
                return {
                    "success": False,
                    "data": None,
                    "error": "Training already running",
                }
            self._status.update({"running": True, "message": "starting", "log": []})
        thread = Thread(
            target=self._train_thread,
            args=(Path("datas") / data_path,),
            daemon=True,
        )
        thread.start()
        return {"success": True, "data": None, "error": None}

    def get_status(self) -> Dict[str, Any]:
        """Return training status and logs."""
        with self._state_lock:
            return {"success": True, "data": self._status.copy(), "error": None}

    def inference(self, question: str) -> Dict[str, Any]:
        """Return model answer for a question."""
        try:
            answer = infer(question, self._cfg)
        except Exception as exc:  # pragma: no cover - high level
            return {"success": False, "data": None, "error": str(exc)}
        return {"success": True, "data": {"answer": answer}, "error": None}

    def delete_model(self) -> Dict[str, Any]:
        """Remove saved model file if present."""
        path = Path("models") / "transformer.pt"
        try:
            path.unlink(missing_ok=True)
        except Exception as exc:  # pragma: no cover - OS level errors
            return {"success": False, "data": None, "error": str(exc)}
        return {"success": True, "data": None, "error": None}

    def get_dataset_info(self, data_path: str = ".") -> Dict[str, Any]:
        """Return basic dataset statistics."""
        try:
            ds = QADataset(Path("datas") / data_path)
        except Exception as exc:  # pragma: no cover - file errors
            return {"success": False, "data": None, "error": str(exc)}
        return {"success": True, "data": {"size": len(ds)}, "error": None}

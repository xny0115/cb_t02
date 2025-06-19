from __future__ import annotations

"""Service layer managing training and inference."""

from pathlib import Path
from threading import Thread, Lock
from typing import Any, Dict
import logging
import psutil

try:
    import pynvml  # type: ignore
except Exception:  # pragma: no cover - optional
    pynvml = None

from ..config import Config, load_config, save_config
from ..training import train, infer
from ..data.loader import QADataset
from ..tuning.auto import AutoTuner

logger = logging.getLogger(__name__)


class ChatbotService:
    """Central application service."""

    def __init__(self) -> None:
        self.cfg = load_config()
        self._status: Dict[str, Any] = {"running": False, "message": "idle", "log": []}
        self._lock = Lock()
        self._thread: Thread | None = None
        self._auto_tune()
        logger.info("Service initialized")

    def _auto_tune(self) -> None:
        """Update config suggestion based on dataset and hardware."""
        try:
            ds = QADataset(Path("datas"))
            sugg = AutoTuner(len(ds)).suggest_config()
            for k, v in sugg.__dict__.items():
                if hasattr(self.cfg, k):
                    setattr(self.cfg, k, v)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Auto tune failed: %s", exc)

    def get_config(self) -> Config:
        return self.cfg

    def set_config(self, data: Dict[str, Any]) -> Config:
        for k, v in data.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)
        save_config(self.cfg)
        return self.cfg

    def _train_thread(self, path: Path) -> None:
        def progress(epoch: int, total: int, loss: float) -> None:
            with self._lock:
                msg = f"{epoch}/{total} loss={loss:.4f}"
                self._status["message"] = msg
                self._status["log"].append(msg)

        try:
            logger.info("Training %s", path)
            train(path, self.cfg, progress_cb=progress)
            with self._lock:
                self._status["message"] = "done"
        except Exception as exc:  # pragma: no cover
            with self._lock:
                self._status["message"] = f"error: {exc}"
            logger.error("Training failed: %s", exc)
        finally:
            with self._lock:
                self._status["running"] = False

    def start_training(self, data_path: str = ".") -> None:
        with self._lock:
            if self._status["running"]:
                raise RuntimeError("training already running")
            self._status.update({"running": True, "message": "starting", "log": []})
        self._thread = Thread(target=self._train_thread, args=(Path("datas") / data_path,), daemon=True)
        self._thread.start()

    def stop_training(self) -> None:
        if self._thread and self._thread.is_alive():
            logger.info("Stop requested but not implemented")

    def infer(self, question: str) -> str:
        return infer(question, self.cfg)

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            data = self._status.copy()
        data["cpu_usage"] = psutil.cpu_percent()
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                data["gpu_usage"] = float(util.gpu)
            except Exception:  # pragma: no cover
                data["gpu_usage"] = None
            finally:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
        else:
            data["gpu_usage"] = None
        return data

    def delete_model(self) -> Dict[str, Any]:
        path = Path("models") / "transformer.pt"
        try:
            path.unlink(missing_ok=True)
        except Exception as exc:  # pragma: no cover
            logger.error("Delete model failed: %s", exc)
            return {"success": False, "data": None, "error": str(exc)}
        return {"success": True, "data": {"message": "model deleted"}, "error": None}

    def get_dataset_info(self, data_path: str = ".") -> Dict[str, Any]:
        try:
            ds = QADataset(Path("datas") / data_path)
        except Exception as exc:  # pragma: no cover
            return {"success": False, "data": None, "error": str(exc)}
        return {"success": True, "data": {"size": len(ds)}, "error": None}

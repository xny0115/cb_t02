from __future__ import annotations

"""FastAPI server providing training and inference APIs."""

from pathlib import Path
from threading import Thread, Lock
from typing import Any, Dict

import logging
import psutil
try:
    import pynvml  # type: ignore
except Exception:  # pragma: no cover - optional dep
    pynvml = None

from .config import load_config, save_config
from .training import train, infer
from .data.loader import QADataset
from .tuning.auto import AutoTuner
from .utils.logger import setup_logger

logger = logging.getLogger(__name__)


class Backend:
    """Simple in-process API for the webview UI."""

    def __init__(self) -> None:
        setup_logger()
        self._cfg = load_config()
        self._auto_tune()
        self._state_lock = Lock()
        self._status: Dict[str, Any] = {
            "running": False,
            "message": "idle",
            "log": [],
        }
        logger.info("Backend initialized")

    def _auto_tune(self) -> None:
        """Adjust config based on dataset size and hardware."""
        try:
            ds = QADataset(Path("datas"))
            tuner = AutoTuner(len(ds))
            params = tuner.suggest()
            self._cfg.batch_size = params["batch_size"]
            self._cfg.learning_rate = params["learning_rate"]
            self._cfg.num_epochs = params["epochs"]
            logger.info("Auto tuned: %s", params)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Auto tune failed: %s", exc)

    def get_config(self) -> Dict[str, Any]:
        """Return current configuration."""
        logger.info("Config requested")
        return {"success": True, "data": self._cfg.__dict__, "error": None}

    def update_config(self, conf: Dict[str, Any]) -> Dict[str, Any]:
        """Update configuration and persist to disk."""
        for key, val in conf.items():
            if hasattr(self._cfg, key):
                setattr(self._cfg, key, val)
        save_config(self._cfg)
        logger.info("Config updated")
        return {"success": True, "data": self._cfg.__dict__, "error": None}

    def _train_thread(self, path: Path) -> None:
        def cb(epoch: int, total: int, loss: float) -> None:
            with self._state_lock:
                self._status["message"] = f"{epoch}/{total} loss={loss:.4f}"
                self._status["log"].append(self._status["message"])

        try:
            logger.info("Training started: %s", path)
            train(path, self._cfg, progress_cb=cb)
            with self._state_lock:
                self._status["message"] = "done"
            logger.info("Training completed")
        except Exception as exc:  # pragma: no cover - high level
            with self._state_lock:
                self._status["message"] = f"error: {exc}"
            logger.error("Training failed: %s", exc)
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
        logger.info("Start train requested for %s", data_path)
        thread = Thread(
            target=self._train_thread,
            args=(Path("datas") / data_path,),
            daemon=True,
        )
        thread.start()
        return {
            "success": True,
            "data": {"message": "training started"},
            "error": None,
        }

    def get_status(self) -> Dict[str, Any]:
        """Return training status, logs and resource usage."""
        with self._state_lock:
            status = self._status.copy()
        status["cpu_usage"] = psutil.cpu_percent()
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                status["gpu_usage"] = float(util.gpu)
            except Exception:  # pragma: no cover - GPU optional
                status["gpu_usage"] = None
            finally:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
        else:
            status["gpu_usage"] = None
        return {"success": True, "data": status, "error": None}

    def inference(self, question: str) -> Dict[str, Any]:
        """Return model answer for a question."""
        try:
            logger.info("Inference requested: %s", question)
            answer = infer(question, self._cfg)
        except Exception as exc:  # pragma: no cover - high level
            logger.error("Inference failed: %s", exc)
            return {"success": False, "data": None, "error": str(exc)}
        logger.info("Inference result: %s", answer)
        return {"success": True, "data": {"answer": answer}, "error": None}

    def delete_model(self) -> Dict[str, Any]:
        """Remove saved model file if present."""
        path = Path("models") / "transformer.pt"
        try:
            logger.info("Delete model requested")
            path.unlink(missing_ok=True)
            logger.info("Model file deleted")
        except Exception as exc:  # pragma: no cover - OS level errors
            logger.error("Delete model failed: %s", exc)
            return {"success": False, "data": None, "error": str(exc)}
        return {
            "success": True,
            "data": {"message": "model deleted"},
            "error": None,
        }

    def get_dataset_info(self, data_path: str = ".") -> Dict[str, Any]:
        """Return basic dataset statistics."""
        try:
            ds = QADataset(Path("datas") / data_path)
            logger.info("Dataset info requested for %s", data_path)
        except Exception as exc:  # pragma: no cover - file errors
            logger.error("Dataset info failed: %s", exc)
            return {"success": False, "data": None, "error": str(exc)}
        return {"success": True, "data": {"size": len(ds)}, "error": None}

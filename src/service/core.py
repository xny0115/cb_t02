from __future__ import annotations
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Dict, Tuple
import json
import logging
import time
import torch
try:
    import psutil  # type: ignore
except Exception:
    psutil = None
try:
    import pynvml  # type: ignore
except Exception:
    pynvml = None
from ..config import CONFIG_PATH, load_config
from .utils import to_config, _CFG_MAP
from ..training import train
from ..utils.logger import LOG_PATH, log_gpu_memory
from ..utils.vocab import Tokenizer
from .loader import auto_tune, load_model
logger = logging.getLogger(__name__)
class ChatbotService:
    def __init__(self) -> None:
        self._config = load_config()
        self.training = False
        self.model_path = Path("models") / "current.pth"
        self.model_exists = self.model_path.exists()
        self._status_msg = "idle"
        self._progress = 0.0
        self._last_loss = 0.0
        self._tokenizer: Tokenizer | None = None
        self._model = None
        self._lock = Lock(); self._thread: Thread | None = None
        auto_tune(self._config)
        if self.model_exists:
            self._tokenizer, self._model = load_model(self.model_path)
        logger.info("Service initialized")
    def get_config(self) -> Dict[str, Any]:
        return self._config.copy()
    def update_config(self, cfg: Dict[str, Any]) -> Tuple[bool, str]:
        """Update config dict and persist with type casting."""
        try:
            with self._lock:
                # map dataclass-style keys to internal names
                for key, (src, _) in _CFG_MAP.items():
                    if key in cfg:
                        cfg[src] = cfg[key]
                self._config.update(cfg)
                # force type for critical params
                self._config["epochs"] = int(cfg.get("epochs", 20))
                self._config["batch_size"] = int(cfg.get("batch_size", 4))
                self._config["lr"] = float(cfg.get("lr", 1e-4))
                json.dump(self._config, open(CONFIG_PATH, "w"), indent=2)
            return True, "saved"
        except Exception as exc:
            logger.exception("config save failed")
            return False, str(exc)
    def train(self, path: Path) -> None:
        """Run the training process synchronously."""
        cfg = self._config
        epochs = int(cfg.get("epochs", 20))
        logger.info("training epochs=%d", epochs)
        last = time.time()
        def progress(epoch: int, total: int, loss: float) -> None:
            nonlocal last
            with self._lock:
                self._status_msg = f"{epoch}/{total} loss={loss:.4f}"
                self._progress = epoch / total
                self._last_loss = loss
            if time.time() - last >= 1:
                last = time.time()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Device selected: %s", device)
        log_gpu_memory()
        try:
            train(path, cfg, progress_cb=progress, model_path=self.model_path)
            if not self.model_path.exists() or self.model_path.stat().st_size < 1_000_000:
                raise RuntimeError("모델 저장 실패: 생성 실패 또는 용량 미달")
            msg = "done"
            self.model_exists = True
            logger.info("Training complete")
            self._tokenizer, self._model = load_model(self.model_path)
        except Exception as exc:
            logger.exception("Training failed")
            with self._lock:
                self._status_msg = f"error: {exc}"
                self._progress = 0.0
                self.training = False
            self.model_exists = False
            return
        with self._lock:
            self.training = False
            self._status_msg = msg
            self._progress = 1.0 if msg == "done" else 0.0
    def start_training(self, data_path: str = ".") -> Dict[str, Any]:
        logging.getLogger().setLevel(
            logging.DEBUG if self._config.get("verbose", False) else logging.INFO
        )
        with self._lock:
            if self.training:
                logger.warning("Training already running")
                return {"success": False, "msg": "already_training", "data": None}
            self.training = True
            self._status_msg = "starting"
            self._progress = 0.001
        log_gpu_memory()
        self._thread = Thread(target=self.train, args=(Path("datas") / data_path,), daemon=True)
        self._thread.start()
        return {"success": True, "msg": "started", "data": None}
    def stop_training(self) -> None:
        if self._thread and self._thread.is_alive():
            logger.info("Stop requested, waiting for thread")
            self._thread.join()
    def infer(self, text: str) -> Dict[str, Any]:
        logger.info("infer | model_exists=%s", self.model_exists)
        if not self.model_exists or self._model is None or self._tokenizer is None:
            return {"success": False, "msg": "model_missing", "data": None}
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Device selected: %s", device)
            if hasattr(self._model, "to"):
                self._model.to(device).eval()
            tokens = self._tokenizer.encode(text).unsqueeze(1).to(device)
            out = self._model.generate(tokens, max_new_tokens=64)
            ids = out.view(-1).tolist()[1:]
            decoded = self._tokenizer.decode(ids).strip()
            if not decoded:
                logger.warning("empty answer fallback used | prompt=%s", text[:40])
                decoded = self._simple_fallback(text)
            logger.debug("infer result: %s", decoded[:60])
            return {"success": True, "msg": "", "data": decoded}
        except Exception as exc:
            logger.exception("infer failed")
            return {"success": False, "msg": f"error: {exc}", "data": None}

    def _simple_fallback(self, prompt: str) -> str:
        """Return a short apology message."""
        if "?" in prompt:
            return "죄송합니다. 답변을 준비하지 못했습니다."
        return "답변을 생성하지 못했습니다."
    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            msg = self._status_msg
            running = self.training
            progress = self._progress
            last_loss = self._last_loss
        data = {
            "training": running,
            "progress": progress,
            "status_msg": msg,
            "last_loss": last_loss,
            "model_exists": self.model_exists,
        }
        if psutil is not None:
            data["cpu_usage"] = psutil.cpu_percent()
        else:
            data["cpu_usage"] = 0.0
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                data["gpu_usage"] = float(util.gpu)
            except Exception:
                data["gpu_usage"] = None
            finally:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
        else:
            data["gpu_usage"] = None
        try:
            lines = LOG_PATH.read_text(encoding="utf-8").splitlines()
            data["logs"] = "\n".join(lines[-2000:])
        except Exception:
            data["logs"] = ""
        data["current_config"] = self._config.copy()
        return {"success": True, "msg": "ok", "data": data}
    def delete_model(self) -> bool:
        if self.training:
            self.stop_training()
        if self.model_path.exists():
            try:
                self.model_path.unlink()
                self.model_exists = False
                self._status_msg = "model_deleted"
                logger.info("Model deleted")
                return True
            except Exception:
                logger.exception("Delete model failed")
                return False
        return False

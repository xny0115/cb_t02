from __future__ import annotations
"""Service layer managing training and inference."""
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Dict
import logging
import time
import torch
try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional
    psutil = None
try:
    import pynvml  # type: ignore
except Exception:  # pragma: no cover - optional
    pynvml = None
from ..config import Config, load_config, save_config
from ..training import train
from ..data.loader import QADataset
from ..tuning.auto import AutoTuner
from ..utils.logger import LOG_PATH, log_gpu_memory
from ..utils.vocab import Tokenizer, build_vocab
logger = logging.getLogger(__name__)
class ChatbotService:
    """Central application service."""
    def __init__(self) -> None:
        self.cfg = load_config()
        self.training = False
        self.model_path = Path("models") / "current.pth"
        self.model_exists = self.model_path.exists()
        self._status_msg = "idle"
        self._progress = 0.0
        self._last_loss = 0.0
        self._tokenizer: Tokenizer | None = None
        self._model = None
        self._lock = Lock()
        self._thread: Thread | None = None
        self._auto_tune()
        if self.model_exists:
            self._load_model()
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
    def _load_model(self) -> None:
        """Load model and tokenizer from disk."""
        try:
            ds = QADataset(Path("datas"))
            vocab = build_vocab(ds)
            self._tokenizer = Tokenizer(vocab)
            from ..model.transformer import Seq2SeqTransformer
            self._model = Seq2SeqTransformer(vocab_size=len(vocab))
            self._model.load_state_dict(
                torch.load(self.model_path, map_location="cpu")
            )
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Model load failed: %s", exc)
    def get_config(self) -> Config:
        return self.cfg
    def set_config(self, data: Dict[str, Any]) -> Config:
        for k, v in data.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)
        save_config(self.cfg)
        return self.cfg
    def train(self, path: Path) -> None:
        """Run the training process synchronously."""
        last = time.time()
        def progress(epoch: int, total: int, loss: float) -> None:
            nonlocal last
            if time.time() - last >= 1:
                logger.info("Epoch %d/%d loss %.4f", epoch, total, loss)
                last = time.time()
            with self._lock:
                self._status_msg = f"{epoch}/{total} loss={loss:.4f}"
                self._progress = epoch / total
                self._last_loss = loss
        logger.info("Training started...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Device selected: %s", device)
        log_gpu_memory()
        try:
            train(path, self.cfg, progress_cb=progress, model_path=self.model_path)
            if not self.model_path.exists() or self.model_path.stat().st_size < 1_000_000:
                raise RuntimeError("모델 저장 실패: 생성 실패 또는 용량 미달")
            msg = "done"
            self.model_exists = True
            logger.info("Training complete")
            self._load_model()
        except Exception as exc:  # pragma: no cover
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
        if self._thread and self._thread.is_alive():  # pragma: no cover
            logger.info("Stop requested, waiting for thread")  # pragma: no cover
            self._thread.join()  # pragma: no cover
    def infer(self, text: str) -> Dict[str, Any]:
        if not self.model_exists or self._model is None or self._tokenizer is None:
            return {"success": False, "msg": "model_missing", "data": None}
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Device selected: %s", device)
            self._model.to(device).eval()
            tokens = self._tokenizer.encode(text).unsqueeze(1).to(device)
            output = self._model.generate(tokens, max_new_tokens=64)
            answer = self._tokenizer.decode(output.squeeze().tolist()[1:])
            return {"success": True, "msg": "", "data": answer}
        except Exception as exc:
            logger.exception("infer failed")
            return {"success": False, "msg": f"error: {exc}", "data": None}
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
        if pynvml is not None:  # pragma: no cover - optional GPU info
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
        try:
            lines = LOG_PATH.read_text(encoding="utf-8").splitlines()
            data["logs"] = "\n".join(lines[-2000:])
        except Exception:
            data["logs"] = ""
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
            except Exception:  # pragma: no cover
                logger.exception("Delete model failed")
                return False
        return False
    def get_dataset_info(self, data_path: str = ".") -> Dict[str, Any]:
        try:  # pragma: no cover
            ds = QADataset(Path("datas") / data_path)
        except Exception as exc:  # pragma: no cover
            return {"success": False, "data": None, "error": str(exc)}
        return {"success": True, "data": {"size": len(ds)}, "error": None}

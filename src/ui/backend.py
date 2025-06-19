from __future__ import annotations

"""pywebview API bridge."""

from typing import Any, Dict

from ..service import ChatbotService


class WebBackend:
    """Expose methods for the web UI."""

    def __init__(self) -> None:
        self.svc = ChatbotService()

    def get_config(self) -> Dict[str, Any]:
        cfg = self.svc.get_config()
        return {"success": True, "data": cfg.__dict__, "error": None}

    def set_config(self, conf: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.svc.set_config(conf)
        return {"success": True, "data": cfg.__dict__, "error": None}

    def start_training(self, data_path: str = ".") -> Dict[str, Any]:
        try:
            self.svc.start_training(data_path)
            return {"success": True, "data": {"message": "training started"}, "error": None}
        except Exception as exc:  # pragma: no cover
            return {"success": False, "data": None, "error": str(exc)}

    def stop_training(self) -> Dict[str, Any]:
        self.svc.stop_training()
        return {"success": True, "data": {"message": "stop requested"}, "error": None}

    def infer(self, question: str) -> Dict[str, Any]:
        try:
            ans = self.svc.infer(question)
            return {"success": True, "data": {"answer": ans}, "error": None}
        except Exception as exc:  # pragma: no cover
            return {"success": False, "data": None, "error": str(exc)}

    def get_status(self) -> Dict[str, Any]:
        return {"success": True, "data": self.svc.get_status(), "error": None}

    def delete_model(self) -> Dict[str, Any]:
        return self.svc.delete_model()

    def get_dataset_info(self, data_path: str = ".") -> Dict[str, Any]:
        return self.svc.get_dataset_info(data_path)

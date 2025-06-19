from __future__ import annotations

"""pywebview API bridge."""

from typing import Any, Dict

from ..service import ChatbotService


class WebBackend:
    """Expose methods for the web UI."""

    def __init__(self) -> None:
        self.svc = ChatbotService()

    def start_training(self) -> Dict[str, Any]:
        return self.svc.start_training()

    def delete_model(self) -> Dict[str, Any]:
        return self.svc.delete_model()

    def infer(self, text: str) -> Dict[str, Any]:
        return self.svc.infer(text)

    def get_status(self) -> Dict[str, Any]:
        return self.svc.get_status()

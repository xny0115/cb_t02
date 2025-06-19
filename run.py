from __future__ import annotations

"""Application entrypoint for the local webview."""

try:  # pragma: no cover - optional dependency
    import webview  # type: ignore
except Exception:  # pragma: no cover
    import time

    class _DummyWebview:
        def create_window(self, *args, **kwargs) -> None:
            pass

        def start(self, *args, **kwargs) -> None:
            while True:
                time.sleep(1)

    webview = _DummyWebview()

from src.utils.logger import setup_logger
from src.service import ChatbotService
from src.ui.backend import WebBackend


def main() -> None:
    """Launch the local webview."""
    setup_logger()
    svc = ChatbotService()
    api = WebBackend(svc)
    webview.create_window("Chatbot", "ui.html", js_api=api)
    webview.start(gui="edgechromium", http_server=False)


if __name__ == "__main__":
    main()

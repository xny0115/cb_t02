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


def _ensure_cuda_torch() -> None:
    """Install CUDA-enabled torch package if needed."""
    import importlib
    import subprocess
    import sys
    import torch

    if torch.cuda.is_available():
        return
    url = (
        "https://download.pytorch.org/whl/cu118/torch-2.3.0%2Bcu118-cp310-cp310-win_amd64.whl"
    )
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", url])
        importlib.reload(torch)
        assert torch.cuda.is_available(), "CUDA torch install failed"
    except Exception as exc:  # pragma: no cover - best effort
        print(f"CUDA torch install failed: {exc}")


def main() -> None:
    """Launch the local webview."""
    setup_logger()
    _ensure_cuda_torch()
    svc = ChatbotService()
    api = WebBackend(svc)
    webview.create_window("Chatbot", "ui.html", js_api=api)
    webview.start(gui="edgechromium", http_server=False)


if __name__ == "__main__":
    main()

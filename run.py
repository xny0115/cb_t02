from __future__ import annotations

"""Application entrypoint for the local webview."""

import webview

from src.utils.logger import setup_logger
from src.ui.backend import WebBackend


if __name__ == "__main__":
    setup_logger()
    webview.create_window("Chatbot", "ut.html", js_api=WebBackend())
    webview.start(gui="edgechromium", http_server=False)

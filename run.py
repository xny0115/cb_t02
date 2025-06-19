from __future__ import annotations

"""Application entrypoint for the local webview."""

import webview

from src.app import Backend
from src.utils.logger import setup_logger


if __name__ == "__main__":
    setup_logger()
    webview.create_window("Chatbot", "ui.html", js_api=Backend())
    webview.start()

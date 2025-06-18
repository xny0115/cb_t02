from __future__ import annotations

"""Application entrypoint for the local webview."""

import webview

from src.app import Backend


if __name__ == "__main__":
    webview.create_window("Chatbot", "ui.html", js_api=Backend())
    webview.start()

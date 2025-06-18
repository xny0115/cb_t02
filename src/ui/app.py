import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict

import webview

from run import train


@dataclass
class TrainingLog:
    messages: List[str] = field(default_factory=list)

    def add(self, text: str) -> None:
        self.messages.append(text)
        if len(self.messages) > 200:
            self.messages = self.messages[-200:]

    def tail(self) -> str:
        return "\n".join(self.messages[-20:])


class Backend:
    def __init__(self) -> None:
        self.is_training = False
        self.log = TrainingLog()
        self.status: str = "idle"

    def get_config_values(self) -> Dict[str, int | float | str]:
        return {
            "num_epochs": 10,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "dropout_ratio": 0.1,
        }

    def get_system_status(self) -> Dict[str, int]:
        # Dummy values; real implementation would query psutil or pynvml
        return {"cpu": 0, "gpu": 0}

    def start_training(self, data_path: str) -> bool:
        if self.is_training:
            return False
        self.is_training = True
        self.status = "running"
        thread = threading.Thread(target=self._train, args=(data_path,), daemon=True)
        thread.start()
        return True

    def _train(self, data_path: str) -> None:
        def cb(epoch: int, total: int, loss: float) -> None:
            self.log.add(f"Epoch {epoch}/{total}: loss={loss:.4f}")

        try:
            train_args = type("Args", (), {"data": data_path})
            train(train_args, progress_cb=cb)
            self.status = "done"
        except Exception as exc:  # pragma: no cover - UI feedback only
            self.status = f"error: {exc}"
        finally:
            self.is_training = False

    def get_status(self) -> Dict[str, str]:
        return {"status": self.status, "log": self.log.tail()}


def main() -> None:
    backend = Backend()
    window = webview.create_window(
        "Chatbot", "ui.html", width=1024, height=768, js_api=backend
    )
    webview.start(gui="edgechromium", http_server=True, func=None, debug=False)


if __name__ == "__main__":
    main()

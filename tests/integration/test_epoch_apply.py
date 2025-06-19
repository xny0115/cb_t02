from src.ui.backend import WebBackend
from src.service import ChatbotService
from src.utils.logger import setup_logger, LOG_PATH
import time


def test_epoch_apply(tmp_path):
    setup_logger()
    backend = WebBackend(ChatbotService())
    backend.set_config({'epochs': 5})
    backend.start_training()
    found = False
    for _ in range(60):
        text = LOG_PATH.read_text(encoding='utf-8')
        if 'Training complete' in text:
            found = True
            break
        time.sleep(0.1)
    backend.delete_model()
    assert found

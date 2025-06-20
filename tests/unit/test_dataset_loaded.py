import logging
from src.service.core import ChatbotService


def test_dataset_loaded(caplog):
    caplog.set_level(logging.INFO)
    svc = ChatbotService()
    size = len(svc._dataset)
    msgs = [r.message for r in caplog.records if 'dataset size=' in r.message]
    assert msgs and str(size) in msgs[0]

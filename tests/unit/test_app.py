from src.service.core import ChatbotService

svc = ChatbotService()


def test_config_roundtrip(tmp_path):
    cfg = svc.get_config()
    new_epochs = cfg['epochs'] + 1
    svc.update_config({'epochs': new_epochs})
    assert svc.get_config()['epochs'] == new_epochs

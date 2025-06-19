from src.service.core import ChatbotService

svc = ChatbotService()


def test_config_roundtrip(tmp_path):
    cfg = svc.get_config()
    new_epochs = cfg.num_epochs + 1
    svc.set_config({'num_epochs': new_epochs})
    assert svc.get_config().num_epochs == new_epochs

import time
from src.service.core import ChatbotService

def test_training_cycle(tmp_path):
    svc = ChatbotService()
    svc.cfg.num_epochs = 1
    res = svc.start_training()
    assert res["success"]
    time.sleep(3)
    assert svc.model_path.exists()
    assert svc.model_path.stat().st_size >= 1_000_000
    del_res = svc.delete_model()
    assert del_res["success"]
    assert not svc.model_path.exists()
    inf = svc.infer("hi")
    assert not inf["success"]


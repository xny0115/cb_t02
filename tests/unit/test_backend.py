from src.ui.backend import WebBackend
from src.config import load_config
import time


def test_backend_cycle(tmp_path):
    backend = WebBackend()
    cfg = load_config()
    cfg.num_epochs = 1
    backend.svc.set_config(cfg.__dict__)
    res = backend.start_training()
    assert res['success']
    while True:
        st = backend.get_status()
        if st['data']['message'] in ('done', '') or st['data']['message'].startswith('error'):
            break
        time.sleep(0.1)
    del_res = backend.delete_model()
    assert del_res['success']
    inf = backend.infer('hi')
    assert not inf['success']

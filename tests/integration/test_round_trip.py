from src.app import Backend
from src.config import load_config

backend = Backend()


def test_train_infer_cycle(tmp_path):
    # update config to run 1 epoch for speed
    cfg = load_config()
    cfg.num_epochs = 1
    backend.update_config(cfg.__dict__)
    backend.start_train('.')
    # wait for training to finish
    import time
    for _ in range(20):
        status = backend.get_status()['data']['message']
        if status.startswith('error') or status == 'done':
            break
        time.sleep(0.1)
    infer_res = backend.inference('인공지능이란 뭐야?')
    assert infer_res['success']
    assert 'answer' in infer_res['data']

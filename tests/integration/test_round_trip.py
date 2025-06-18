import json
from fastapi.testclient import TestClient
from src.app import app
from src.config import load_config

client = TestClient(app)


def test_train_infer_cycle(tmp_path):
    # update config to run 1 epoch for speed
    cfg = load_config()
    cfg.num_epochs = 1
    client.put('/config', data=json.dumps(cfg.__dict__))
    res = client.post('/train', json={'data_path': '.'})
    assert res.status_code == 200
    # wait for training to finish
    import time
    for _ in range(20):
        status = client.get('/status').json()['data']['message']
        if status.startswith('error') or status == 'done':
            break
        time.sleep(0.1)
    infer_res = client.post('/infer', json={'question': '인공지능이란 뭐야?'})
    assert infer_res.status_code == 200
    assert 'answer' in infer_res.json()['data']

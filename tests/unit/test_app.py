import json
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)


def test_config_roundtrip(tmp_path):
    resp = client.get('/config')
    assert resp.status_code == 200
    data = resp.json()['data']
    new_cfg = data.copy()
    new_cfg['num_epochs'] = data['num_epochs'] + 1
    res = client.put('/config', data=json.dumps(new_cfg))
    assert res.status_code == 200
    assert res.json()['data']['num_epochs'] == new_cfg['num_epochs']

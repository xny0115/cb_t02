from fastapi.testclient import TestClient
from src.service.api import app


def test_status_endpoint():
    client = TestClient(app)
    res = client.get('/training/status')
    assert res.status_code == 200
    data = res.json()
    for key in ['running', 'current_epoch', 'total_epochs', 'loss']:
        assert key in data

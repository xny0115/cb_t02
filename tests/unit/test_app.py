from src.app import Backend

backend = Backend()


def test_config_roundtrip(tmp_path):
    resp = backend.get_config()
    data = resp['data']
    new_cfg = data.copy()
    new_cfg['num_epochs'] = data['num_epochs'] + 1
    res = backend.update_config(new_cfg)
    assert res['data']['num_epochs'] == new_cfg['num_epochs']

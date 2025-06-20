import json
from pathlib import Path
from src.training import train
from src.config import Config

def test_resume(tmp_path):
    model_path = tmp_path / 'model.pth'
    cfg = Config(num_epochs=2, batch_size=2)
    train(Path('datas'), cfg, model_path=model_path, resume=True)
    meta = json.loads((model_path.with_suffix('.meta.json')).read_text())
    loss_before = meta['best_loss']
    cfg.num_epochs = 3
    train(Path('datas'), cfg, model_path=model_path, resume=True)
    meta2 = json.loads((model_path.with_suffix('.meta.json')).read_text())
    assert meta2['last_epoch'] == 3
    assert meta2['best_loss'] <= loss_before

def test_corrupt_meta(tmp_path):
    model_path = tmp_path / 'model.pth'
    meta_file = model_path.with_suffix('.meta.json')
    meta_file.write_text('{bad json')
    cfg = Config(num_epochs=1, batch_size=1)
    try:
        train(Path('datas'), cfg, model_path=model_path, resume=True)
    except SystemExit:
        assert True
    else:
        assert False

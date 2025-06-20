import json
from pathlib import Path
from src.training import train
from src.config import Config

def test_resume(tmp_path):
    model_path = tmp_path / 'model.pth'
    cfg = Config(num_epochs=2, batch_size=2)
    train(Path('datas'), cfg, model_path=model_path, resume=True)
    meta = json.loads((model_path.parent / 'ckpts/current.meta.json').read_text())
    loss_before = meta['last_loss']
    cfg.num_epochs = 3
    train(Path('datas'), cfg, model_path=model_path, resume=True)
    meta2 = json.loads((model_path.parent / 'ckpts/current.meta.json').read_text())
    assert meta2['last_epoch'] == 2
    assert meta2['last_loss'] <= loss_before

def test_corrupt_meta(tmp_path):
    model_path = tmp_path / 'model.pth'
    ckpt_dir = model_path.parent / 'ckpts'
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / 'current.meta.json').write_text('{bad json')
    cfg = Config(num_epochs=1, batch_size=1)
    try:
        train(Path('datas'), cfg, model_path=model_path, resume=True)
    except SystemExit:
        assert True
    else:
        assert False

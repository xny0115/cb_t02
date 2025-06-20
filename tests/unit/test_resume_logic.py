import json
from pathlib import Path
from src.training import train
from src.config import Config


def test_resume_logic(tmp_path):
    model_path = tmp_path / 'model.pth'
    cfg = Config(num_epochs=4, batch_size=2)
    train(Path('datas'), cfg, model_path=model_path, resume=True)
    meta_file = model_path.parent / 'ckpts' / 'current.meta.json'
    assert meta_file.exists()
    cfg.num_epochs = 10
    train(Path('datas'), cfg, model_path=model_path, resume=True)
    meta = json.loads(meta_file.read_text())
    assert meta['last_epoch'] == 10

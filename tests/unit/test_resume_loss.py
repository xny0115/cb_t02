import json
from pathlib import Path
from src.training import train
from src.config import Config


def test_resume_loss(tmp_path):
    model = tmp_path / 'model.pth'
    cfg = Config(num_epochs=3, batch_size=2)
    train(Path('datas'), cfg, model_path=model, resume=True)
    cfg.num_epochs = 5
    train(Path('datas'), cfg, model_path=model, resume=True)
    meta = json.loads((model.parent / 'ckpts/current.meta.json').read_text())
    assert meta['last_epoch'] == 5

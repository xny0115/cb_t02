from src.training_utils import migrate_optimizer_state, ensure_model_device
import torch


def test_migrate():
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.Adam(model.parameters())
    target = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(target)
    migrate_optimizer_state(opt, target)
    ensure_model_device(model, target)
    for state in opt.state.values():
        for v in state.values():
            if torch.is_tensor(v):
                assert v.device == target

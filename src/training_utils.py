from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable
from torch.utils.data import Dataset
from .data.loader import QADataset
from .utils.vocab import encode
import logging
import torch
log = logging.getLogger(__name__)

@dataclass
class EarlyStopping:
    patience: int
    best_loss: float | None = None
    counter: int = 0

    def step(self, loss: float) -> bool:
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class TorchQADataset(Dataset):
    def __init__(self, data: QADataset, vocab: dict[str, int]) -> None:
        if not vocab:
            raise ValueError("empty vocab")
        self.data = data
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        q, a = self.data[idx]
        if not q or not a:
            raise ValueError("invalid pair")
        return encode(q, self.vocab), encode(a, self.vocab)


def collate_fn(batch: Iterable[tuple[Any, Any]]):
    from torch import nn
    qs, as_ = zip(*batch)
    qs = nn.utils.rnn.pad_sequence(qs, padding_value=0)
    as_ = nn.utils.rnn.pad_sequence(as_, padding_value=0)
    return qs, as_

from pathlib import Path
import json
from typing import Any, Tuple
import torch
from torch import nn, optim


def load_checkpoint(ckpt_dir: Path, device: torch.device) -> Tuple[dict[str, Any], int]:
    """Load checkpoint data and starting epoch.

    Parameters
    ----------
    ckpt_dir : Path
        Directory containing checkpoint files.
    device : torch.device
        Target device for loading tensors.

    Returns
    -------
    Tuple[dict[str, Any], int]
        Checkpoint dictionary and epoch to resume from.
    """
    meta_file = ckpt_dir / "current.meta.json"
    if not meta_file.exists():
        return {}, 0
    try:
        meta = json.loads(meta_file.read_text())
        ckpt_path = ckpt_dir / f"ckpt_{meta['last_epoch']:04}.pt"
        ckpt = torch.load(ckpt_path, map_location=device)
    except Exception as exc:  # pragma: no cover - corruption handling
        raise SystemExit(f"checkpoint load failed: {exc}")
    start_epoch = int(meta.get("last_epoch", -1)) + 1
    return ckpt, start_epoch


def save_checkpoint(
    ckpt_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    loss: float,
) -> None:
    """Persist training state to disk.

    Parameters
    ----------
    ckpt_dir : Path
        Directory where checkpoint files are written.
    epoch : int
        Current training epoch.
    model : nn.Module
        Model to save.
    optimizer : optim.Optimizer
        Optimizer instance.
    scheduler : optim.lr_scheduler._LRScheduler
        Scheduler instance.
    loss : float
        Latest epoch loss.
    """
    ckpt_dir.mkdir(exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "loss": loss,
    }
    torch.save(ckpt, ckpt_dir / f"ckpt_{epoch:04}.pt")
    meta = {"last_epoch": epoch, "loss": loss}
    (ckpt_dir / "current.meta.json").write_text(json.dumps(meta))


def migrate_optimizer_state(optim: optim.Optimizer, device: torch.device) -> None:
    """Move all optimizer state tensors to the target device."""
    for state in optim.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device, non_blocking=True)


def ensure_model_device(model: nn.Module, device: torch.device) -> None:
    """Ensure every parameter resides on the given device."""
    off = [n for n, p in model.named_parameters() if p.device != device]
    if off:
        log.warning("Migrating %d parameters to %s", len(off), device)
        model.to(device, non_blocking=True)



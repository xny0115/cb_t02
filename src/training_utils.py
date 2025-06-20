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
from typing import Any
import torch
from torch import nn, optim




def migrate_optimizer_state(optim: optim.Optimizer, device: torch.device) -> None:
    """Move all optimizer state tensors to the target device."""
    for state in optim.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device, non_blocking=True)


def ensure_model_device(
    model: nn.Module, device: torch.device, once: bool = True
) -> None:
    """Ensure every parameter resides on the given device."""
    if once and getattr(model, "_device_checked", False):
        return
    off = [n for n, p in model.named_parameters() if p.device != device]
    if off:
        log.warning("Migrating %d parameters to %s", len(off), device)
        model.to(device, non_blocking=True)
    model._device_checked = True



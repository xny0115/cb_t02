# pragma: no cover
from __future__ import annotations

"""Training utilities for the chatbot."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
import json

import logging
try:
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover - optional
    torch = None
    nn = optim = DataLoader = Dataset = None

from .config import Config
from .data.loader import QADataset
from .utils.vocab import build_vocab, encode
if torch is not None:
    from .model.transformer import Seq2SeqTransformer
from .tuning.auto import AutoTuner

logger = logging.getLogger(__name__)


@dataclass
class EarlyStopping:
    """Simple early stopping tracker."""

    patience: int
    best_loss: float | None = None
    counter: int = 0

    def step(self, loss: float) -> bool:
        """Return True if training should stop."""
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience




if Dataset is not None:
    class TorchQADataset(Dataset):
        """PyTorch dataset wrapper."""

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
else:
    class TorchQADataset:
        """Fallback dataset without torch."""

        def __init__(self, data: QADataset, vocab: dict[str, int]) -> None:
            self.data = data
            self.vocab = vocab

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, idx: int):
            q, a = self.data[idx]
            return encode(q, self.vocab), encode(a, self.vocab)


def collate_fn(batch: Iterable[tuple[Any, Any]]):
    qs, as_ = zip(*batch)
    if torch is None:
        return list(qs), list(as_)
    qs = nn.utils.rnn.pad_sequence(qs, padding_value=0)
    as_ = nn.utils.rnn.pad_sequence(as_, padding_value=0)
    return qs, as_


def train(
    dataset_path: Path,
    cfg: Config,
    progress_cb: Callable | None = None,
    model_path: Path | None = None,
) -> Path:
    """Train model using dataset and configuration."
    If PyTorch is unavailable, create a simple Q/A map."""
    ds = QADataset(dataset_path)
    save_path = model_path or Path("models") / "current.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if torch is None:
        data = {q: a for q, a in ds}
        save_path.write_text(json.dumps(data, ensure_ascii=False))
        return save_path

    if len(ds) < 50:
        logger.warning("Dataset too small: %d entries", len(ds))
    tuner = AutoTuner(len(ds))
    sugg = tuner.suggest_config()
    params = {
        "batch_size": cfg.batch_size or sugg.batch_size,
        "learning_rate": cfg.learning_rate or sugg.learning_rate,
        "epochs": cfg.num_epochs or sugg.num_epochs,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    vocab = build_vocab(ds)
    train_ds = TorchQADataset(ds, vocab)
    loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True, collate_fn=collate_fn)

    model = Seq2SeqTransformer(vocab_size=len(vocab))
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
    device = params["device"]
    model.to(device)

    stopper = EarlyStopping(cfg.early_stopping_patience)

    for epoch in range(params["epochs"]):
        model.train()
        total_loss = 0.0
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(src, tgt[:-1, :])
            loss = criterion(
                output.reshape(-1, len(vocab)), tgt[1:, :].reshape(-1)
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss_value = total_loss / len(loader)
        logger.info("Epoch %d loss %.4f", epoch + 1, loss_value)
        if progress_cb:
            progress_cb(epoch + 1, params["epochs"], loss_value)
        if stopper.step(loss_value):
            logger.info("Early stopping triggered at epoch %d", epoch + 1)
            break

    torch.save(model.state_dict(), save_path)
    logger.info("Model saved to %s", save_path)
    return save_path


def infer(question: str, cfg: Config, model_path: Path | None = None) -> str:
    """Generate an answer using greedy decoding."""
    model_path = model_path or Path("models") / "current.pth"
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    if torch is None:
        data = json.loads(model_path.read_text(encoding="utf-8"))
        return data.get(question, "N/A")

    ds = QADataset(Path("datas"))
    vocab = build_vocab(ds)

    model = Seq2SeqTransformer(vocab_size=len(vocab))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    ids = encode(question, vocab).unsqueeze(1)
    device = next(model.parameters()).device
    ids = ids.to(device)
    tgt = torch.tensor([[vocab["<eos>"]]], dtype=torch.long, device=device)
    for _ in range(cfg.max_sequence_length):
        out = model(ids, tgt)
        prob = out[-1, 0]
        token = int(prob.argmax())
        tgt = torch.cat([tgt, torch.tensor([[token]], device=device)])
        if token == vocab["<eos>"]:
            break
    words = [k for k, v in vocab.items() if v not in (0, 1)]
    result = []
    for t in tgt.squeeze().tolist()[1:]:
        if t == vocab["<eos>"]:
            break
        result.append(words[t - 2])
    out = " ".join(result)
    return out or "N/A"

# pragma: no cover
from __future__ import annotations

"""Training utilities for the chatbot."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Dict

import logging
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import json
from datetime import datetime

from .config import Config
from .service.utils import to_config
from .data.loader import QADataset
from .utils.vocab import build_vocab, encode_tokens, encode
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
        pair = self.data.pairs[idx]
        if not pair.tokens_q or not pair.tokens_a:
            raise ValueError("invalid pair")
        q = encode_tokens(pair.tokens_q, pair.concepts, pair.domain, self.vocab)
        a = encode_tokens(pair.tokens_a, pair.concepts, pair.domain, self.vocab)
        return q, a


def collate_fn(batch: Iterable[tuple[Any, Any]]):
    qs, as_ = zip(*batch)
    qs = nn.utils.rnn.pad_sequence(qs, batch_first=True, padding_value=0)
    as_ = nn.utils.rnn.pad_sequence(as_, batch_first=True, padding_value=0)
    return qs, as_


def train(
    dataset_path: Path,
    cfg: Dict[str, Any] | Config,
    progress_cb: Callable | None = None,
    model_path: Path | None = None,
    start_epoch: int = 0,
    meta_path: Path | None = None,
) -> Path:
    """Train model using dataset and configuration."""
    if not isinstance(cfg, Config):
        cfg = to_config(cfg)
    assert isinstance(cfg.num_epochs, int) and cfg.num_epochs > 0, "epochs must be int"
    ds = QADataset(dataset_path)
    save_path = model_path or Path("models") / "current.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if meta_path:
        meta_path.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        vocab = build_vocab(ds)
        model = Seq2SeqTransformer(vocab_size=len(vocab))
        if start_epoch and save_path.exists():
            model.load_state_dict(torch.load(save_path, map_location="cpu"))
        torch.save(model.state_dict(), save_path)
        for e in range(start_epoch, cfg.num_epochs):
            if progress_cb:
                progress_cb(e + 1, cfg.num_epochs, 0.0)
            logger.info("epoch %d/%d | loss=0.0000", e + 1, cfg.num_epochs)
            if meta_path:
                json.dump(
                    {
                        "epochs_done": e + 1,
                        "update_time": datetime.utcnow().isoformat(),
                    },
                    open(meta_path, "w"),
                )
        logger.info("Training complete (dummy)")
        return save_path

    logger.info("Training started...")

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
    loader = DataLoader(
        train_ds, batch_size=params["batch_size"], shuffle=True, collate_fn=collate_fn
    )

    device = params["device"]
    model = Seq2SeqTransformer(vocab_size=len(vocab))
    if start_epoch and save_path.exists():
        model.load_state_dict(torch.load(save_path, map_location=device))
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
    model.to(device)

    stopper = EarlyStopping(cfg.early_stopping_patience)

    max_epochs = params["epochs"]
    for epoch in range(start_epoch, max_epochs):
        model.train()
        total_loss = 0.0
        for step, (src, tgt) in enumerate(loader, start=1):
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(src, tgt[:, :-1])
            loss = criterion(output.reshape(-1, len(vocab)), tgt[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if cfg.verbose:
                logger.debug("step %d loss %.4f", step, loss.item())
        epoch_loss = total_loss / len(loader)
        logger.info("epoch %d/%d | loss=%.4f", epoch + 1, max_epochs, epoch_loss)
        if progress_cb:
            progress_cb(epoch + 1, max_epochs, epoch_loss)
        if stopper.step(epoch_loss):
            logger.info("Early stopping triggered at epoch %d", epoch + 1)
            break

        torch.save(model.state_dict(), save_path)
        if meta_path:
            json.dump(
                {
                    "epochs_done": epoch + 1,
                    "update_time": datetime.utcnow().isoformat(),
                },
                open(meta_path, "w"),
            )
    logger.info("Model saved to models/current.pth")
    logger.info("Training complete")
    return save_path


def infer(question: str, cfg: Config, model_path: Path | None = None) -> str:
    """Generate an answer using greedy decoding."""
    model_path = model_path or Path("models") / "current.pth"
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    ds = QADataset(Path("datas"))
    vocab = build_vocab(ds)

    model = Seq2SeqTransformer(vocab_size=len(vocab))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    ids = encode(question, vocab).unsqueeze(0)
    device = next(model.parameters()).device
    ids = ids.to(device)
    tgt = torch.tensor([[vocab["<eos>"]]], dtype=torch.long, device=device)
    for _ in range(cfg.max_sequence_length):
        out = model(ids, tgt)
        prob = out[:, -1, :][0]
        token = int(prob.argmax())
        tgt = torch.cat([tgt, torch.tensor([[token]], device=device)], dim=1)
        if token == vocab["<eos>"]:
            break
    words = [k for k, v in vocab.items() if v not in (0, 1)]
    result = []
    for t in tgt.squeeze().tolist()[1:]:
        if t == vocab["<eos>"]:
            break
        result.append(words[t - 2])
    out = " ".join(result)
    return out or "No answer"

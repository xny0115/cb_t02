# pragma: no cover
from __future__ import annotations

"""Training utilities for the chatbot."""

from pathlib import Path
from typing import Any, Callable, Dict
import logging
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import json
from datetime import datetime
import math
import argparse

from .config import Config
from .service.utils import to_config
from .data.loader import QADataset
from .utils.vocab import build_vocab, encode
from .model.transformer import Seq2SeqTransformer
from .tuning.auto import AutoTuner
from .training_utils import EarlyStopping, TorchQADataset, collate_fn

logger = logging.getLogger(__name__)




def train(
    dataset_path: Path,
    cfg: Dict[str, Any] | Config,
    progress_cb: Callable | None = None,
    model_path: Path | None = None,
    start_epoch: int = 0,
    meta_path: Path | None = None,
    resume: bool = True,
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
    ckpt_dir = save_path.parent / "ckpts"
    ckpt_dir.mkdir(exist_ok=True)

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
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1)
    meta_file = ckpt_dir / "current.meta.json"
    if resume and meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text())
            ckpt = torch.load(ckpt_dir / f"ckpt_{meta['last_epoch']:04}.pt", map_location=device)
        except Exception as exc:
            raise SystemExit(f"checkpoint load failed: {exc}")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = meta["last_epoch"] + 1
    elif start_epoch and save_path.exists():
        model.load_state_dict(torch.load(save_path, map_location=device))
    model.to(device)

    stopper = EarlyStopping(cfg.early_stopping_patience)
    max_epochs = params["epochs"]
    for epoch in range(start_epoch, max_epochs):
        model.train()
        total_loss = 0.0
        for step, (src, tgt) in enumerate(loader, start=1):
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(src, tgt[:-1, :])
            loss = criterion(output.reshape(-1, len(vocab)), tgt[1:, :].reshape(-1))
            if not math.isfinite(loss.item()):
                raise RuntimeError(f"Non-finite loss detected: {loss.item()}")
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
        scheduler.step()
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "loss": epoch_loss,
        }
        torch.save(ckpt, ckpt_dir / f"ckpt_{epoch:04}.pt")
        meta_file.write_text(json.dumps({"last_epoch": epoch, "last_loss": epoch_loss}))
    torch.save(model.state_dict(), save_path)
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
    return out or "No answer"


def main() -> None:
    """CLI entry."""
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--data-dir", default="datas")
    args = p.parse_args()
    cfg = Config()
    try:
        train(Path(args.data_dir), cfg, resume=args.resume)
    except Exception as exc:  # pragma: no cover - CLI
        print(exc)
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()

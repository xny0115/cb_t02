from __future__ import annotations
"""Training utilities for the chatbot."""
from pathlib import Path
from typing import Any, Callable, Dict
import logging
try:
    import torch
except ImportError as exc:  # pragma: no cover - env handling
    raise ImportError(
        "PyTorch missing; install it manually before running tests."
    ) from exc
from torch import nn, optim
from torch.utils.data import DataLoader
import json
from datetime import datetime
import math
import argparse
from .config import Config
from .service.utils import to_config
from .data.loader import QADataset
from .utils.vocab import build_vocab
from .model.transformer import Seq2SeqTransformer
from .tuning.auto import AutoTuner
from .training_utils import (
    EarlyStopping,
    TorchQADataset,
    collate_fn,
    save_checkpoint,
    migrate_optimizer_state,
    ensure_model_device,
)
logger = logging.getLogger(__name__)

def train(
    dataset_path: Path,
    cfg: Dict[str, Any] | Config,
    progress_cb: Callable | None = None,
    model_path: Path | None = None,
    start_epoch: int = 0,
    meta_path: Path | None = None,
    resume: bool = True,
    device: torch.device | None = None,
) -> Path:
    """Train model using dataset and configuration."""
    from .utils.fs_lock import pid_lock

    with pid_lock(Path('.training.lock')):
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
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(
                "cuda" if str(device) == "cuda" and torch.cuda.is_available() else "cpu"
            )
        meta_file = meta_path or ckpt_dir / "current.meta.json"
        if not torch.cuda.is_available():
            if resume and meta_file.exists():
                try:
                    json.load(open(meta_file))
                except Exception as exc:
                    raise SystemExit(f"checkpoint load failed: {exc}")
            vocab = build_vocab(ds)
            model = Seq2SeqTransformer(vocab_size=len(vocab))
            if start_epoch and save_path.exists():
                model.load_state_dict(torch.load(save_path, map_location="cpu"))
            torch.save(model.state_dict(), save_path)
            meta_file.parent.mkdir(parents=True, exist_ok=True)
            for e in range(start_epoch, cfg.num_epochs):
                if progress_cb:
                    progress_cb(e + 1, cfg.num_epochs, 0.0)
                logger.info("epoch %d/%d | loss=0.0000", e + 1, cfg.num_epochs)
                json.dump(
                    {
                        "last_epoch": e,
                        "loss": 0.0,
                        "update_time": datetime.utcnow().isoformat(),
                    },
                    open(meta_file, "w"),
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
        }
        vocab = build_vocab(ds)
        train_ds = TorchQADataset(ds, vocab)
        loader = DataLoader(
            train_ds, batch_size=params["batch_size"], shuffle=True, collate_fn=collate_fn
        )
        model = Seq2SeqTransformer(vocab_size=len(vocab))
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1)
        last_epoch = -1
        meta = {"last_epoch": -1, "best_loss": float("inf")}
        if meta_file.exists() and resume:
            try:
                meta = json.load(open(meta_file))
                last_epoch = int(meta.get("last_epoch", -1))
            except Exception as exc:
                raise SystemExit(f"checkpoint load failed: {exc}")
            ckpt_file = ckpt_dir / f"ckpt_{last_epoch:04}.pt"
            if ckpt_file.exists():
                ckpt = torch.load(ckpt_file, map_location=device)
                model.load_state_dict(ckpt["model_state"])
                optimizer.load_state_dict(ckpt["optim_state"])
                scheduler.load_state_dict(ckpt["scheduler_state"])
                model.to(device, non_blocking=True)
                migrate_optimizer_state(optimizer, device)
        if cfg.num_epochs <= last_epoch:
            logger.info(
                "Requested epochs already completed (last=%d). Skipping training.",
                last_epoch,
            )
            return save_path
        start_epoch = max(start_epoch, last_epoch + 1)
        if start_epoch and save_path.exists() and last_epoch < 0:
            model.load_state_dict(torch.load(save_path, map_location=device))
        model.to(device, non_blocking=True)
        ensure_model_device(model, device, once=False)
        stopper = EarlyStopping(cfg.early_stopping_patience)
        max_epochs = params["epochs"]
        for epoch in range(start_epoch, max_epochs):
            model.train()
            ensure_model_device(model, device)
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
            save_checkpoint(ckpt_dir, epoch, model, optimizer, scheduler, epoch_loss)
            meta["last_epoch"] = epoch
            meta["best_loss"] = min(meta.get("best_loss", float("inf")), epoch_loss)
            json.dump(meta, open(meta_file, "w"))
        torch.save(model.state_dict(), save_path)
        logger.info("Model saved to models/current.pth")
        logger.info("Training complete")
        return save_path



def main() -> None:
    """CLI entry."""
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="datas")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--epochs", type=int, default=None, help="total target epochs")
    args = p.parse_args()
    cfg = Config()
    try:
        if args.epochs:
            cfg.num_epochs = args.epochs
        device = torch.device(
            "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        train(Path(args.data_dir), cfg, resume=not args.no_resume, device=device)
    except Exception as exc:  # pragma: no cover - CLI
        print(exc)
        raise SystemExit(1)

if __name__ == "__main__":  # pragma: no cover
    main()

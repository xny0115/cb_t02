"""Training entrypoint for QA chatbot."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from src.data.loader import QADataset
from src.model.transformer import Seq2SeqTransformer
from src.tuning.auto import AutoTuner


def build_vocab(dataset: QADataset) -> dict:
    """Build vocabulary from dataset."""
    tokens = set()
    for q, a in dataset:
        tokens.update(q.split())
        tokens.update(a.split())
    vocab = {t: i + 2 for i, t in enumerate(sorted(tokens))}
    vocab["<pad>"] = 0
    vocab["<eos>"] = 1
    return vocab


def encode(text: str, vocab: dict) -> torch.Tensor:
    ids = [vocab.get(t, 0) for t in text.split()] + [vocab["<eos>"]]
    return torch.tensor(ids, dtype=torch.long)


class QADatasetTorch(Dataset):
    """Torch dataset wrapper."""

    def __init__(self, dataset: QADataset, vocab: dict) -> None:
        self.dataset = dataset
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        q, a = self.dataset[idx]
        return encode(q, self.vocab), encode(a, self.vocab)


def collate_fn(batch):
    qs, as_ = zip(*batch)
    qs = nn.utils.rnn.pad_sequence(qs, padding_value=0)
    as_ = nn.utils.rnn.pad_sequence(as_, padding_value=0)
    return qs, as_


def train(args: argparse.Namespace) -> None:
    data_path = Path("datas") / args.dataset
    ds = QADataset(data_path)
    tuner = AutoTuner(len(ds))
    params = tuner.suggest()
    vocab = build_vocab(ds)
    train_ds = QADatasetTorch(ds, vocab)
    loader = DataLoader(
        train_ds,
        batch_size=params["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = Seq2SeqTransformer(vocab_size=len(vocab))
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

    device = params["device"]
    model.to(device)

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
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

    save_path = Path("models") / "transformer.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="qa_test_10.json", help="Dataset filename in datas/")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

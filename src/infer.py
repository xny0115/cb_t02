from __future__ import annotations
from pathlib import Path
import torch
from src.config import Config
from src.data.loader import QADataset
from src.utils.vocab import build_vocab, encode
from src.model.transformer import Seq2SeqTransformer


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

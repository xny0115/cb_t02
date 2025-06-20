"""Simple Transformer Seq2Seq model."""

from __future__ import annotations

import math
import logging
from typing import Tuple

import torch
from torch import nn


def _sanitize_probs(probs: torch.Tensor) -> torch.Tensor:
    """Ensure probability distribution sums to 1."""
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    if probs.sum() == 0:
        probs.fill_(1.0 / probs.numel())
        probs /= probs.sum()
    else:
        probs.div_(probs.sum())
    return probs

logger = logging.getLogger(__name__)


class Seq2SeqTransformer(nn.Module):
    """Minimal Seq2Seq Transformer implementation."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_ff: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        self.pos_decoder = PositionalEncoding(embed_dim, dropout)
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_ff,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(
        self, src: torch.Tensor, tgt: torch.Tensor
    ) -> torch.Tensor:
        src = self.embed(src) * math.sqrt(self.embed.embedding_dim)
        tgt = self.embed(tgt) * math.sqrt(self.embed.embedding_dim)
        src = self.pos_encoder(src)
        tgt = self.pos_decoder(tgt)
        output = self.transformer(src, tgt)
        return self.fc_out(output)

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        max_new_tokens: int = 64,
        eos_id: int = 1,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        no_repeat_ngram: int = 2,
    ) -> torch.Tensor:
        """Generate sequence with sampling and n-gram blocking."""
        self.eval()
        device = src.device
        ys = torch.tensor([[eos_id]], device=device)
        ngrams: set[tuple[int, ...]] = set()
        for step in range(max_new_tokens):
            out = self(src, ys)[-1, 0] / temperature
            if no_repeat_ngram > 1 and ys.size(0) >= no_repeat_ngram:
                prefix = ys[0, -no_repeat_ngram + 1 :].tolist()
                banned = [w for w in range(out.size(0)) if tuple(prefix + [w]) in ngrams]
                if banned:
                    out[banned] = -float("inf")
            k = min(top_k, out.size(0))
            topk_val, topk_idx = out.topk(k)
            probs = _sanitize_probs(torch.softmax(topk_val, -1))
            if 0.0 < top_p < 1.0:
                s_probs, s_idx = probs.sort(descending=True)
                cum = s_probs.cumsum(dim=-1)
                mask = cum > top_p
                if mask.all():
                    mask[-1] = False
                s_probs[mask] = 0
                s_probs = _sanitize_probs(s_probs)
                choice = torch.multinomial(s_probs, 1).item()
                idx = s_idx[choice].item()
                next_id = topk_idx[idx].view(1, 1)
            else:
                choice = torch.multinomial(probs, 1).item()
                next_id = topk_idx[choice].view(1, 1)
            ys = torch.cat([ys, next_id], dim=0)
            if no_repeat_ngram > 1 and ys.size(0) >= no_repeat_ngram:
                ngrams.add(tuple(ys[0, -no_repeat_ngram:].tolist()))
            if next_id.item() == eos_id:
                break
        return ys


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(
        self, emb_size: int, dropout: float = 0.1, max_len: int = 5000
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, emb_size, 2) * (-math.log(10000.0) / emb_size)
        )
        pe = torch.zeros(max_len, 1, emb_size)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)

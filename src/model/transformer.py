"""Simple Transformer Seq2Seq model."""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn


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

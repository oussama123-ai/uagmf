"""
Stage ⑤: Temporal Modelling — Two-layer Transformer with sinusoidal positional encoding.

Architecture (Table 2 from paper):
    - Layers: 2
    - Heads: 8
    - Positional encoding: Sinusoidal (fixed)
    - Dropout: 0.3
    - d_model: 256 (shared with fusion)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class TemporalTransformer(nn.Module):
    """
    Two-layer Transformer temporal model (Stage ⑤).

    Args:
        d_model:   Feature dimension (default 256).
        n_layers:  Number of transformer encoder layers (default 2).
        n_heads:   Multi-head attention heads (default 8).
        d_ff:      Feed-forward hidden dimension (default 512 = 2×d_model).
        dropout:   Dropout probability (default 0.3).
        max_len:   Maximum sequence length.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.3,
        max_len: int = 256,
    ) -> None:
        super().__init__()
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,    # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.pool = nn.Linear(d_model, d_model)   # learnable pooling

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:    (B, T, d_model) sequence of fused frame features.
            src_key_padding_mask: (B, T) bool — True = padded frame.

        Returns:
            pooled:   (B, d_model) temporal aggregation.
            last_attn: None (TransformerEncoder doesn't expose attn weights by default).
        """
        x = self.pos_enc(x)                          # (B, T, d_model)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)
        # Mean pooling (masked)
        if src_key_padding_mask is not None:
            mask = (~src_key_padding_mask).float().unsqueeze(-1)
            pooled = (x * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            pooled = x.mean(1)
        return self.pool(pooled), None

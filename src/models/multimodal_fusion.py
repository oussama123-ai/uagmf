"""
Stage ④: Cross-Attention Multimodal Fusion.

Implements Equations 4–5 and Algorithm 2 from the paper.

Cross-attention fuses physiological context into the visual stream:
    Attn(Q, K, V) = softmax(QKᵀ / √dk) V
    f_fused = Concat[Attn(fᵛ,fʰ,fʰ), Attn(fᵛ,fˢ,fˢ), Attn(fᵛ,fʳ,fʳ)] Wₒ

Absent-modality masking (Section 3.4.1):
    When modality m is absent, its attention logit receives an additive
    bias bₘ = −10⁶ (equivalent to PyTorch attn_mask), so exp(bₘ) → 0
    and the softmax normalises only over available modalities.
    This is numerically stable and prevents underflow.

Partial-modality handling (Algorithm 2):
    Step 1: Detect absent modalities Mabs
    Step 2: Zero-fill absent embeddings before projection
    Step 3: Apply log-space additive mask (−10⁶) after key projection
    Step 4: Fuse with masked attentions
    Step 5: Elevate uncertainty δᵤ = 0.08 × |Mabs|
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

PHYSIO_MODALITIES = ["hrv", "spo2", "resp"]
ABSENT_LOGIT_BIAS = -1e6   # additive mask for absent modalities (log-space)


class TCNBlock(nn.Module):
    """
    Temporal Convolutional Network block with residual connection.
    Exponential dilation factors: [1, 2, 4].
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=pad),
            nn.GroupNorm(1, out_ch),
            nn.ReLU(inplace=True),
        )
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.residual(x)


class ModalEncoder(nn.Module):
    """
    CNN-TCN encoder for a single modality (video or physiological signal).

    For video: takes pre-extracted visual features (from MobileNetV3-S backbone).
    For physiological: takes raw 1D time series.

    Output: (B, d_model) embedding.
    """

    def __init__(
        self,
        in_dim: int,
        d_model: int = 256,
        tcn_layers: int = 3,
        kernel_size: int = 3,
        dilations: List[int] = None,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        dilations = dilations or [1, 2, 4]
        assert len(dilations) == tcn_layers

        layers = []
        ch = in_dim
        for i in range(tcn_layers):
            layers.append(TCNBlock(ch, hidden_dim, kernel_size, dilations[i]))
            ch = hidden_dim
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, in_dim) or (B, in_dim, T)
        Returns:
            (B, d_model)
        """
        if x.dim() == 3 and x.size(-1) != self.tcn[0].conv[0].in_channels:
            x = x.transpose(1, 2)   # (B, in_dim, T)
        h = self.tcn(x)              # (B, hidden_dim, T)
        h = self.pool(h).squeeze(-1) # (B, hidden_dim)
        return self.proj(h)          # (B, d_model)


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention that uses visual stream as query and physiological
    embeddings as keys/values.

    Absent-modality logits are masked via additive −10⁶ bias.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        for m in [self.W_Q, self.W_K, self.W_V]:
            nn.init.xavier_uniform_(m.weight)

    def forward(
        self,
        query: torch.Tensor,           # (B, d_model) visual
        key_value: torch.Tensor,        # (B, d_model) physiological
        available: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            attended: (B, d_model)
            attn_weight: (B, d_head) for interpretability
        """
        Q = self.W_Q(query)
        K = self.W_K(key_value)
        V = self.W_V(key_value)

        scale = self.d_head ** 0.5
        score = (Q * K).sum(-1, keepdim=True) / scale   # (B, 1)

        # Absent-modality masking (Section 3.4.1):
        # equivalent to PyTorch attn_mask = −10⁶ for absent modalities
        if not available:
            score = score + ABSENT_LOGIT_BIAS

        attn = torch.sigmoid(score)  # (B, 1) weight ∈ [0,1]
        attn = self.dropout(attn)
        attended = attn * V

        return attended, attn


class MultimodalFusionLayer(nn.Module):
    """
    Full cross-attention fusion layer with multiple physiological modalities.
    Implements Eq. 5 from paper:
        f_fused = Concat[Attn(fᵛ,fʰ,fʰ), Attn(fᵛ,fˢ,fˢ), Attn(fᵛ,fʳ,fʳ)] Wₒ

    Args:
        d_model:           Feature dimension.
        n_heads:           Attention heads.
        n_physio:          Number of physiological modalities (default 3).
        dropout:           Attention dropout.
        modality_dropout:  Training-time modality dropout probability.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_physio: int = 3,
        dropout: float = 0.1,
        modality_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_physio = n_physio
        self.modality_dropout = modality_dropout

        # One cross-attention module per physiological modality
        self.cross_attns = nn.ModuleList([
            CrossAttentionFusion(d_model, n_heads, dropout)
            for _ in range(n_physio)
        ])

        # Output projection: concatenated attended features → d_model
        self.W_o = nn.Linear(n_physio * d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        f_v: torch.Tensor,
        physio_feats: List[Optional[torch.Tensor]],
        availability: Optional[List[bool]] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Implements Algorithm 2 (partial-modality inference).

        Args:
            f_v:          (B, d_model) visual embedding.
            physio_feats: List of n_physio tensors each (B, d_model) or None.
            availability: List of booleans indicating which modalities are present.

        Returns:
            f_fused:      (B, d_model)
            delta_u:      Uncertainty elevation scalar δᵤ = 0.08 × |Mabs|
        """
        B = f_v.size(0)
        if availability is None:
            availability = [f is not None for f in physio_feats]

        # Step 1: Detect absent modalities
        n_absent = sum(1 for a in availability if not a)

        # Training-time modality dropout (randomly mask modalities)
        if self.training and self.modality_dropout > 0:
            new_avail = []
            for a in availability:
                if a and torch.rand(1).item() < self.modality_dropout:
                    new_avail.append(False)
                    n_absent += 1
                else:
                    new_avail.append(a)
            availability = new_avail

        attended_list = []
        for i, (cross_attn, physio, avail) in enumerate(
            zip(self.cross_attns, physio_feats, availability)
        ):
            # Step 2: Zero-fill absent embeddings
            if physio is None or not avail:
                kv = torch.zeros(B, self.d_model, device=f_v.device)
            else:
                kv = physio

            # Step 3: Attention masking (−10⁶ for absent; see Section 3.4.1)
            attended, _ = cross_attn(f_v, kv, available=avail)
            attended_list.append(attended)

        # Step 4: Fusion (Eq. 5)
        concat = torch.cat(attended_list, dim=-1)   # (B, n_physio * d_model)
        f_fused = self.W_o(concat)                  # (B, d_model)
        f_fused = self.norm(f_v + f_fused)
        f_fused = self.norm2(f_fused + self.ffn(f_fused))

        # Step 5: Uncertainty elevation
        delta_u = 0.08 * n_absent   # δᵤ = 0.08 × |Mabs|

        return f_fused, delta_u


class CrossAttentionMultimodalFusion(nn.Module):
    """
    Full Stage ④ module with modality-specific encoders + 2-layer fusion.

    Physiological modalities: HRV (4 features), SpO₂ (1), Resp (1).
    Video features come from MobileNetV3-S (passed as pre-extracted tensors).

    Args:
        video_dim:    Dimension of pre-extracted video features.
        hrv_dim:      HRV feature dimension (default 4).
        d_model:      Fusion dimension (default 256).
        n_heads:      Attention heads (default 8).
        fusion_layers: Number of stacked fusion layers (default 2).
        modality_dropout: Prob. of dropping a modality during training.
    """

    def __init__(
        self,
        video_dim: int = 256,
        hrv_dim: int = 4,
        d_model: int = 256,
        n_heads: int = 8,
        fusion_layers: int = 2,
        dropout: float = 0.1,
        modality_dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # Modality-specific encoders
        self.video_proj = nn.Linear(video_dim, d_model)
        self.hrv_enc = ModalEncoder(hrv_dim, d_model)
        self.spo2_enc = ModalEncoder(1, d_model)
        self.resp_enc = ModalEncoder(1, d_model)

        # Stacked fusion layers
        self.fusion_layers = nn.ModuleList([
            MultimodalFusionLayer(d_model, n_heads, 3, dropout, modality_dropout)
            for _ in range(fusion_layers)
        ])

    def forward(
        self,
        video_feats: torch.Tensor,
        hrv: Optional[torch.Tensor] = None,
        spo2: Optional[torch.Tensor] = None,
        resp: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Args:
            video_feats: (B, video_dim) or (B, T, video_dim)
            hrv:   (B, T, 4) or None
            spo2:  (B, T, 1) or None
            resp:  (B, T, 1) or None

        Returns:
            f_fused: (B, d_model)
            delta_u: cumulative uncertainty elevation
        """
        # Encode video
        if video_feats.dim() == 3:
            f_v = self.video_proj(video_feats.mean(1))
        else:
            f_v = self.video_proj(video_feats)

        # Encode physiological signals
        availability = [hrv is not None, spo2 is not None, resp is not None]
        physio_feats = []
        for signal, enc in [(hrv, self.hrv_enc), (spo2, self.spo2_enc), (resp, self.resp_enc)]:
            if signal is not None:
                if signal.dim() == 2:
                    signal = signal.unsqueeze(-1)   # (B, T, 1)
                physio_feats.append(enc(signal))
            else:
                physio_feats.append(None)

        # Stacked cross-attention fusion
        total_delta_u = 0.0
        for layer in self.fusion_layers:
            f_v, delta_u = layer(f_v, physio_feats, availability)
            total_delta_u += delta_u

        return f_v, total_delta_u

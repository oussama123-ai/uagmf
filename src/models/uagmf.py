"""
UAG-MF: Full 7-Stage End-to-End Pipeline.

Stage ①: Multimodal inputs (video, HRV, SpO₂, Resp, optional audio)
Stage ②: Occlusion Detector (ResNet-18)
Stage ③: Generative Reconstruction (cGAN / VAE) → residual ρ
Stage ④: Cross-Attention Multimodal Fusion → δᵤ
Stage ⑤: Temporal Transformer
Stage ⑥: Dual UQ Layer (MC Dropout + Deep Ensemble) ← ρ, δᵤ
Stage ⑦: Output µ ± σ, alert if σ² > τ* = 0.35

Total parameters: 28.4M (inference: 16.1–23.9M depending on occlusion type).
See S3 Table for per-stage breakdown.

NOTE on PatchGAN Discriminator (0.6M params):
    Training only. NOT loaded at inference. See infer.py line 47.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .generative_reconstruction import GenerativeReconstruction
from .multimodal_fusion import CrossAttentionMultimodalFusion
from .occlusion_detector import OcclusionDetector
from .symbolic_engine import SymbolicConflictEngine
from .temporal_model import TemporalTransformer
from .uq_layer import DualUQLayer

try:
    import torchvision.models as tv_models
    _HAS_TORCHVISION = True
except ImportError:
    _HAS_TORCHVISION = False


class VideoEncoder(nn.Module):
    """MobileNetV3-Small video backbone (4.2M params)."""

    def __init__(self, pretrained: bool = True, output_dim: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        if _HAS_TORCHVISION:
            weights = (tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
                       if pretrained else None)
            backbone = tv_models.mobilenet_v3_small(weights=weights)
            self.features = backbone.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            in_dim = 576   # MobileNetV3-Small last channel
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(16, 576, 1), nn.ReLU(),
            )
            self.pool = nn.AdaptiveAvgPool2d(1)
            in_dim = 576

        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, output_dim),
            nn.GELU(),
        )
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B*T, 3, 112, 112) → (B*T, output_dim)"""
        h = self.pool(self.features(x)).flatten(1)
        return self.proj(h)


class UAGMF(nn.Module):
    """
    UAG-MF end-to-end pain estimation pipeline (28.4M parameters).

    Args:
        d_model:           Shared feature dimension (default 256).
        n_mc_samples:      MC Dropout samples S (default 50).
        n_ensemble:        Ensemble members K (default 5).
        alert_threshold:   τ* (default 0.35).
        rules_path:        Path to symbolic rules JSON.
        latent_dim:        VAE latent dimension (default 256).
        fusion_layers:     Cross-attention layers (default 2).
        temporal_layers:   Transformer layers (default 2).
        nrs_max:           Maximum NRS value (default 10.0).
    """

    def __init__(
        self,
        d_model: int = 256,
        n_mc_samples: int = 50,
        n_ensemble: int = 5,
        alert_threshold: float = 0.35,
        rules_path: str = "rules/symbolic_rules.json",
        latent_dim: int = 256,
        fusion_layers: int = 2,
        temporal_layers: int = 2,
        nrs_max: float = 10.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.nrs_max = nrs_max

        # ── Stage ① handled externally (data loader) ──────────────────────
        # ── Stage ②: Occlusion Detector ──────────────────────────────────
        self.occlusion_detector = OcclusionDetector(pretrained=True, seg_head=True)

        # ── Stage ③: Generative Reconstruction ────────────────────────────
        self.generative_recon = GenerativeReconstruction(latent_dim=latent_dim)

        # ── Video Encoder (MobileNetV3-Small) ─────────────────────────────
        self.video_encoder = VideoEncoder(output_dim=d_model)

        # ── Stage ④: Cross-Attention Multimodal Fusion ─────────────────────
        self.fusion = CrossAttentionMultimodalFusion(
            video_dim=d_model,
            hrv_dim=4,
            d_model=d_model,
            n_heads=8,
            fusion_layers=fusion_layers,
            dropout=0.1,
            modality_dropout=0.3,
        )

        # ── Stage ⑤: Temporal Transformer ─────────────────────────────────
        self.temporal = TemporalTransformer(
            d_model=d_model,
            n_layers=temporal_layers,
            n_heads=8,
            d_ff=d_model * 2,
            dropout=0.3,
        )

        # ── Stage ⑥: Dual UQ Layer ────────────────────────────────────────
        self.uq_layer = DualUQLayer(
            d_model=d_model,
            n_mc_samples=n_mc_samples,
            n_ensemble=n_ensemble,
            alert_threshold=alert_threshold,
            gamma=0.05,
            nrs_max=nrs_max,
        )

        # ── Symbolic Engine (post-hoc, no gradient) ───────────────────────
        self.symbolic_engine = SymbolicConflictEngine(
            rules_path=rules_path,
            alert_threshold=alert_threshold,
        )

        # ── Concept activation projector (12 concepts from d_model) ───────
        self.concept_proj = nn.Linear(d_model, 12)

    def forward(
        self,
        video: torch.Tensor,
        hrv: Optional[torch.Tensor] = None,
        spo2: Optional[torch.Tensor] = None,
        resp: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
        training_recon: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full 7-stage forward pass.

        Args:
            video:  (B, T, 3, H, W) — T frames, H=W=112.
            hrv:    (B, T, 4) or None.
            spo2:   (B, T, 1) or None.
            resp:   (B, T, 1) or None.
            return_intermediates: Include per-stage outputs for analysis.
            training_recon: Pass True during training to get GAN discriminator output.

        Returns:
            dict with at minimum:
                "mu":       (B,) NRS mean estimate ∈ [0, 10]
                "sigma":    (B,) NRS standard deviation
                "var":      (B,) predictive variance σ²
                "alert":    (B,) bool — True if σ² > τ* = 0.35
                "tier":     list of symbolic tier per sample
                "explanation": list of explanation strings
        """
        B, T, C, H, W = video.shape
        device = video.device

        # ── Stage ②: Detect occlusion on middle frame ─────────────────────
        mid_frame = video[:, T // 2]    # (B, 3, H, W)
        occ_out = self.occlusion_detector(mid_frame)
        occ_ratio = occ_out["occlusion_ratio"]   # (B,)
        occ_mask = occ_out["mask"]               # (B, 1, H/2, W/2)

        # ── Stage ③: Generative Reconstruction ────────────────────────────
        # Resize mask to match video spatial dims
        mask_full = F.interpolate(occ_mask, size=(H, W), mode="nearest")
        recon_out = self.generative_recon(
            mid_frame, mask_full, occ_ratio, training_mode=training_recon
        )
        x_hat = recon_out["x_hat"]        # (B, 3, H, W) reconstructed frame
        residual = recon_out["residual"]   # (B,) scalar ρ

        # Replace middle frame with reconstructed version
        video_recon = video.clone()
        video_recon[:, T // 2] = x_hat

        # ── Encode video frames ────────────────────────────────────────────
        video_flat = video_recon.view(B * T, C, H, W)   # (B*T, 3, H, W)
        video_feats = self.video_encoder(video_flat)     # (B*T, d_model)
        video_feats = video_feats.view(B, T, -1)         # (B, T, d_model)

        # ── Stage ④: Cross-Attention Multimodal Fusion ─────────────────────
        f_fused, delta_u = self.fusion(
            video_feats, hrv=hrv, spo2=spo2, resp=resp
        )  # (B, d_model), float

        # ── Stage ⑤: Temporal Transformer ─────────────────────────────────
        # Reshape: fused features need temporal sequence
        # Re-fuse per-frame video with physiological context
        video_seq = video_feats   # (B, T, d_model)
        temporal_out, _ = self.temporal(video_seq)   # (B, d_model)

        # Merge temporal output with fusion output
        combined = (temporal_out + f_fused) / 2.0   # (B, d_model)

        # ── Stage ⑥: Dual UQ Layer ────────────────────────────────────────
        uq_out = self.uq_layer(combined, residual=residual, delta_u=delta_u)

        # ── Symbolic Engine (post-hoc concept extraction) ─────────────────
        with torch.no_grad():
            concept_acts = torch.sigmoid(self.concept_proj(combined))   # (B, 12)
            symbolic = self.symbolic_engine.batch_evaluate(
                concept_acts, uq_out["var"]
            )

        # ── Stage ⑦: Output ────────────────────────────────────────────────
        output = {
            "mu": uq_out["mu"],
            "sigma": uq_out["sigma"],
            "var": uq_out["var"],
            "alert": uq_out["alert"],
            "tier": symbolic["tier"],
            "explanation": symbolic["explanation"],
        }

        if return_intermediates:
            output.update({
                "occlusion_type": occ_out["occlusion_class"],
                "occlusion_ratio": occ_ratio,
                "x_hat": x_hat,
                "residual": residual,
                "delta_u": delta_u,
                "f_fused": f_fused,
                "mu_mc": uq_out["mu_mc"],
                "var_mc": uq_out["var_mc"],
                "mu_ens": uq_out["mu_ens"],
                "var_ens": uq_out["var_ens"],
                "concept_acts": concept_acts,
                "recon_out": recon_out,
            })

        return output

    def count_parameters(self, inference_only: bool = False) -> int:
        """
        Count trainable parameters.

        Args:
            inference_only: If True, exclude PatchGAN discriminator
                            (training only, per S3 Table).
        """
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if inference_only:
            disc_params = sum(
                p.numel() for p in
                self.generative_recon.discriminator.parameters()
            )
            total -= disc_params
        return total

    @classmethod
    def from_checkpoint(cls, path: str, **kwargs) -> "UAGMF":
        """Load model from checkpoint. PatchGAN discriminator excluded (infer.py line 47)."""
        checkpoint = torch.load(path, map_location="cpu")
        model = cls(**kwargs)
        # LINE 47: Exclude discriminator from inference model state
        state_dict = {
            k: v for k, v in checkpoint["model_state_dict"].items()
            if not k.startswith("generative_recon.discriminator")
        }
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model

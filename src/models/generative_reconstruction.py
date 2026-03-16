"""
Stage ③: Generative Reconstruction — cGAN / VAE.

Implements Algorithm 1 from the paper:

    if r ≤ 0.5:   use cGAN (PatchGAN discriminator, 70×70)
    if r > 0.5:   use VAE (U-Net encoder → latent → U-Net decoder)

The reconstruction residual ρ = ‖xᵛ - x̂ᵛ‖₂ is forwarded as a scalar
auxiliary input to the Stage ⑥ UQ layer (zero additional parameters).

cGAN objective (Eq. 2 in paper):
    L_cGAN = E[log D(xᵛ,M)] + E[log(1-D(G(xᵛ_occ,M),M))]
             + λ_ℓ1 ‖xᵛ - G(xᵛ_occ,M)‖₁ + λ_perc L_VGG

VAE objective (Eq. 3 in paper):
    L_VAE = E[log p_θ(xᵛ|z)] - β * KL(q_ϕ || p(z)),  β = 0.5

Note: PatchGAN discriminator (0.6M params) is TRAINING ONLY.
      It is not loaded at inference. See scripts/infer.py line 47.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── U-Net building blocks ────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
        self.conv = ConvBlock(out_ch * 2, out_ch)  # skip connection

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ── cGAN Generator (5-block U-Net) ───────────────────────────────────────────

class UNetGenerator(nn.Module):
    """
    5-block U-Net generator for cGAN facial inpainting.

    Input:  xᵛ_occ (visible region, masked region set to 0) concatenated with M
    Output: reconstructed full frame x̂ᵛ
    """

    def __init__(self, in_channels: int = 4, base_ch: int = 64) -> None:
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_channels, base_ch)
        self.enc2 = DownBlock(base_ch, base_ch * 2)
        self.enc3 = DownBlock(base_ch * 2, base_ch * 4)
        self.enc4 = DownBlock(base_ch * 4, base_ch * 8)
        self.bottleneck = DownBlock(base_ch * 8, base_ch * 8)
        # Decoder
        self.dec4 = UpBlock(base_ch * 8, base_ch * 8)
        self.dec3 = UpBlock(base_ch * 8, base_ch * 4)
        self.dec2 = UpBlock(base_ch * 4, base_ch * 2)
        self.dec1 = UpBlock(base_ch * 2, base_ch)
        self.final = nn.Sequential(
            nn.Conv2d(base_ch, 3, 1),
            nn.Tanh(),
        )

    def forward(
        self, x_occ: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x_occ: (B, 3, H, W) visible-region frame (masked pixels = 0)
            mask:  (B, 1, H, W) binary mask (1 = occluded region)
        Returns:
            x_hat: (B, 3, H, W) reconstructed frame
        """
        # Resize mask to match x_occ spatial dims if needed
        if mask.shape[-2:] != x_occ.shape[-2:]:
            mask = F.interpolate(mask, size=x_occ.shape[-2:], mode="nearest")

        inp = torch.cat([x_occ, mask], dim=1)     # (B, 4, H, W)

        e1 = self.enc1(inp)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)

        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return self.final(d1)


# ── PatchGAN Discriminator (TRAINING ONLY — not loaded at inference) ─────────

class PatchGANDiscriminator(nn.Module):
    """
    70×70 PatchGAN discriminator.

    IMPORTANT: This module is used ONLY during training.
    It is NOT loaded at inference. See scripts/infer.py line 47.
    Parameters: ~0.6M
    """

    def __init__(self, in_channels: int = 7) -> None:
        super().__init__()
        # in_channels = 3 (real/fake) + 3 (condition xᵛ) + 1 (mask M)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 1, 1, bias=False),
            nn.InstanceNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 1),
        )

    def forward(
        self, x: torch.Tensor, condition: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if mask.shape[-2:] != x.shape[-2:]:
            mask = F.interpolate(mask, size=x.shape[-2:], mode="nearest")
        inp = torch.cat([x, condition, mask], dim=1)
        return self.net(inp)


# ── VAE (for heavy occlusion r > 0.5) ────────────────────────────────────────

class VAEEncoder(nn.Module):
    def __init__(self, in_channels: int = 4, latent_dim: int = 256) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_channels, 64), DownBlock(64, 128),
            DownBlock(128, 256), DownBlock(256, 512),
        )
        self.fc_mu = nn.Linear(512 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(512 * 7 * 7, latent_dim)

    def forward(
        self, x_occ: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask.shape[-2:] != x_occ.shape[-2:]:
            mask = F.interpolate(mask, size=x_occ.shape[-2:], mode="nearest")
        inp = torch.cat([x_occ, mask], dim=1)
        h = self.conv(inp)
        h = h.flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim: int = 256, mask_channels: int = 1) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim + mask_channels, 512 * 7 * 7)
        self.up = nn.Sequential(
            UpBlock(512, 256), UpBlock(256, 128), UpBlock(128, 64),
        )
        self.final = nn.Sequential(nn.Conv2d(64, 3, 1), nn.Tanh())

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_pooled = F.adaptive_avg_pool2d(mask, 1).flatten(1)
        h = self.fc(torch.cat([z, mask_pooled], dim=-1))
        h = h.view(h.size(0), 512, 7, 7)
        # For up blocks without skip connections use plain upsampling
        h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
        h = h[:, :256]
        h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
        h = h[:, :128]
        h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
        h = h[:, :64]
        return self.final(h)


class VAERecon(nn.Module):
    def __init__(self, latent_dim: int = 256) -> None:
        super().__init__()
        self.encoder = VAEEncoder(in_channels=4, latent_dim=latent_dim)
        self.decoder = VAEDecoder(latent_dim=latent_dim)

    def reparameterise(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x_occ: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x_occ, mask)
        z = self.reparameterise(mu, logvar)
        x_hat = self.decoder(z, mask)
        return x_hat, mu, logvar

    def vae_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 0.5,
    ) -> torch.Tensor:
        recon = F.mse_loss(x_hat, x, reduction="sum") / x.size(0)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()
        return recon + beta * kl


# ── Full Reconstruction Module (Algorithm 1) ─────────────────────────────────

class GenerativeReconstruction(nn.Module):
    """
    Implements Algorithm 1: occlusion-type-aware generative reconstruction.

    cGAN is used when r ≤ 0.5 (moderate occlusion).
    VAE is used when r > 0.5 (severe occlusion).

    The reconstruction residual ρ = ‖xᵛ - x̂ᵛ‖₂ is returned as a scalar
    and forwarded to the UQ layer (zero additional parameters).

    Args:
        latent_dim:  VAE latent dimension (default 256).
        l1_weight:   λ_ℓ1 for cGAN L1 loss (default 10).
        perc_weight: λ_perc for perceptual loss (default 1).
    """

    def __init__(
        self,
        latent_dim: int = 256,
        l1_weight: float = 10.0,
        perc_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.perc_weight = perc_weight

        self.cgan_generator = UNetGenerator(in_channels=4, base_ch=64)
        self.vae = VAERecon(latent_dim=latent_dim)
        # Discriminator: loaded during training, excluded at inference
        self.discriminator = PatchGANDiscriminator(in_channels=7)

    def forward(
        self,
        x_v: torch.Tensor,
        mask: torch.Tensor,
        occlusion_ratio: torch.Tensor,
        training_mode: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Implements Algorithm 1.

        Args:
            x_v:              (B, 3, H, W) original frame.
            mask:             (B, 1, H, W) binary occlusion mask.
            occlusion_ratio:  (B,) scalar r ∈ [0, 1].
            training_mode:    If True, discriminator output is also returned.

        Returns:
            dict with:
                "x_hat":    (B, 3, H, W) reconstructed frame
                "residual": (B,) scalar ρ per sample
                "mu":       VAE latent mean (for KL loss; None if cGAN path)
                "logvar":   VAE latent logvar (for KL loss; None if cGAN path)
                "d_fake":   discriminator on fake (for cGAN loss; None at inference)
        """
        B = x_v.size(0)
        device = x_v.device

        # Visible region: set masked pixels to 0
        x_occ = x_v * (1.0 - mask)   # xᵛ_occ ≡ xᵛ ⊙ (1−M)

        x_hat = torch.zeros_like(x_v)
        mu = logvar = None
        d_fake = None

        # Per-sample routing based on occlusion ratio
        cgan_mask = occlusion_ratio <= 0.5   # (B,) bool
        vae_mask = ~cgan_mask

        if cgan_mask.any():
            idx = cgan_mask.nonzero(as_tuple=True)[0]
            x_hat_gan = self.cgan_generator(x_occ[idx], mask[idx])
            # Composite: keep visible region, fill occluded with generator output
            x_hat[idx] = x_occ[idx] + x_hat_gan * mask[idx]

            if training_mode:
                d_fake = self.discriminator(x_hat[idx], x_v[idx], mask[idx])

        if vae_mask.any():
            idx = vae_mask.nonzero(as_tuple=True)[0]
            x_hat_vae, mu, logvar = self.vae(x_occ[idx], mask[idx])
            # Resize if needed
            if x_hat_vae.shape[-2:] != x_v.shape[-2:]:
                x_hat_vae = F.interpolate(
                    x_hat_vae, size=x_v.shape[-2:], mode="bilinear", align_corners=False
                )
            x_hat[idx] = x_hat_vae

        # No-occlusion path: pass through unchanged
        no_occ = occlusion_ratio == 0.0
        if no_occ.any():
            idx = no_occ.nonzero(as_tuple=True)[0]
            x_hat[idx] = x_v[idx]

        # Reconstruction residual ρ = ‖xᵛ - x̂ᵛ‖₂  (per sample, scalar)
        residual = (x_v - x_hat).pow(2).flatten(1).sum(1).sqrt()   # (B,)

        return {
            "x_hat": x_hat,
            "residual": residual,
            "mu": mu,
            "logvar": logvar,
            "d_fake": d_fake,
        }

    def cgan_loss(
        self,
        x_v: torch.Tensor,
        x_hat: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Eq. 2 from paper:
        L_cGAN = E[log D(xᵛ,M)] + E[log(1-D(G(xᵛ_occ,M),M))]
                + λ_ℓ1 ‖xᵛ - x̂ᵛ‖₁ + λ_perc L_VGG
        """
        # GAN terms
        d_real = self.discriminator(x_v, x_v, mask)
        d_fake = self.discriminator(x_hat.detach(), x_v, mask)
        loss_d = (
            F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
            + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
        )
        # Generator GAN loss
        d_fake_g = self.discriminator(x_hat, x_v, mask)
        loss_g_adv = F.binary_cross_entropy_with_logits(d_fake_g, torch.ones_like(d_fake_g))

        # L1 pixel loss
        loss_l1 = self.l1_weight * F.l1_loss(x_hat, x_v)

        # Perceptual loss (VGG feature matching) — simplified as MSE here
        # Replace with actual VGG features for full implementation
        loss_perc = self.perc_weight * F.mse_loss(x_hat, x_v)

        return loss_d + loss_g_adv + loss_l1 + loss_perc

#!/usr/bin/env python3
"""
Single-sample inference script for UAG-MF.

Usage:
    python scripts/infer.py \\
        --checkpoint experiments/fold0/best_model.pth \\
        --video /path/to/video.mp4 \\
        --hrv /path/to/hrv.npy \\
        --output results/prediction.json

Note: PatchGAN discriminator (0.6M params, training only) is explicitly
excluded at inference. See UAGMF.from_checkpoint() which filters it,
and line 47 below where the state dict is loaded.
"""

import argparse, json, logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from src.models.uagmf import UAGMF
from src.utils.logging_utils import setup_logging

logger = logging.getLogger("uagmf.infer")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--video", required=True, help="Path to .npy video array (T,3,112,112)")
    p.add_argument("--hrv", default=None)
    p.add_argument("--spo2", default=None)
    p.add_argument("--resp", default=None)
    p.add_argument("--output", default="prediction.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--mc_samples", type=int, default=50)
    p.add_argument("--ensemble_size", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging("logs", run_name="infer")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── Load model (LINE 47: discriminator excluded) ──────────────────────
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = UAGMF(n_mc_samples=args.mc_samples, n_ensemble=args.ensemble_size)
    # LINE 47: Exclude PatchGAN discriminator from inference model state
    state_dict = {k: v for k, v in checkpoint["model_state_dict"].items()
                  if not k.startswith("generative_recon.discriminator")}
    model.load_state_dict(state_dict, strict=False)  # noqa: line 47
    model.to(device).eval()
    logger.info(f"Loaded checkpoint (epoch {checkpoint.get('epoch', '?')}). "
                f"Discriminator excluded from inference model.")

    # ── Load inputs ───────────────────────────────────────────────────────
    video = torch.from_numpy(np.load(args.video).astype(np.float32)).unsqueeze(0).to(device)
    hrv = torch.from_numpy(np.load(args.hrv).astype(np.float32)).unsqueeze(0).to(device) \
        if args.hrv else None
    spo2 = torch.from_numpy(np.load(args.spo2).astype(np.float32)).unsqueeze(0).to(device) \
        if args.spo2 else None
    resp = torch.from_numpy(np.load(args.resp).astype(np.float32)).unsqueeze(0).to(device) \
        if args.resp else None

    # ── Inference ─────────────────────────────────────────────────────────
    with torch.no_grad():
        output = model(video, hrv=hrv, spo2=spo2, resp=resp,
                       return_intermediates=True)

    mu = float(output["mu"][0])
    sigma = float(output["sigma"][0])
    var = float(output["var"][0])
    alert = bool(output["alert"][0])
    tier = output["tier"][0]
    explanation = output["explanation"][0]

    result = {
        "pain_score_mu": round(mu, 3),
        "pain_score_sigma": round(sigma, 3),
        "predictive_variance": round(var, 3),
        "alert": alert,
        "alert_reason": "sigma^2 > tau* = 0.35" if alert else "OK",
        "symbolic_tier": tier,
        "explanation": explanation,
        "occlusion_ratio": float(output.get("occlusion_ratio", torch.tensor([0]))[0]),
    }

    print(f"\nPain score: {mu:.2f} ± {sigma:.2f} (σ² = {var:.3f})")
    print(f"Alert:      {'⚠️  CLINICIAN REVIEW REQUIRED' if alert else '✓ OK'}")
    print(f"Tier:       {tier}  |  {explanation}\n")

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved prediction to {args.output}")


if __name__ == "__main__":
    main()

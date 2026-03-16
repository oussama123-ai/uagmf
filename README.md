# UAG-MF: Uncertainty-Aware Generative Multimodal Fusion for Pain Estimation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c.svg)](https://pytorch.org/)

Official implementation of:

> **UAG-MF: Uncertainty-aware generative multimodal fusion for continuous pain estimation in non-verbal patients under clinical occlusions**  
> Oussama El Othmani, Sami Naouali  
> *PLOS Digital Health*, 2026  
> GitHub: https://github.com/oussama123-ai/uagmf

---

## Overview

UAG-MF is a 7-stage end-to-end framework for continuous, objective pain estimation in non-verbal patients (intubated, sedated, cognitively impaired). It addresses three core clinical problems:

| Problem | UAG-MF Solution |
|---------|----------------|
| **Occlusion** (masks, tubes, bandages cover 30–70% of face) | cGAN/VAE generative reconstruction |
| **Uncertainty** (point estimates without confidence mislead) | MC Dropout + Deep Ensemble → µ ± σ with human-in-the-loop alerts |
| **Missing modalities** (sensors fail or unavailable) | Explicit softmax-masking + modality-dropout penalty |

**Key Results (5-fold subject-independent CV):**

| Dataset | MSE ↓ | PCC ↑ | vs. CNN baseline |
|---------|-------|-------|-----------------|
| BioVid | **1.17 ± 0.05** | **0.87** | −51.3% MSE |
| UNBC-McMaster | **1.33 ± 0.02** | **0.88** | −52.2% MSE |
| EmoPain | **1.71 ± 0.02** | **0.88** | −44.8% MSE |

- ECE = 0.038 (well-calibrated uncertainty)  
- 93 ms inference on Jetson Nano (FP16/TensorRT 8.4)  
- 79.8% zero-shot accuracy on MD-NPL (neonatal OOD)

---

## Architecture (7 Stages)

```
① Multimodal Inputs
   Video [B×T×112×112×3] + HRV [B×T×4] + SpO₂/Resp [B×T]
        │
② Occlusion Detector (ResNet-18)
        │
③ Generative Reconstruction (cGAN / VAE)
   ├─ cGAN  if occlusion ratio r ≤ 0.5
   └─ VAE   if occlusion ratio r > 0.5
   ρ = ‖xᵛ − x̂ᵛ‖₂  ──────────────────────────────┐
        │                                           │ (residual)
④ Cross-Attention Multimodal Fusion                 │
   d=256, 8 heads; absent modalities masked         │
        │                                           │
⑤ Temporal Transformer (2L, 8H)                     │
        │                                           │
⑥ Dual UQ Layer ◄──────────────────────────────────┘
   MC Dropout (S=50) + Deep Ensemble (K=5)
   σ² ← σ² + γρ   (reconstruction residual inflation)
        │
⑦ Output: µ ± σ ∈ [0,10]
   Alert if σ² > τ* = 0.35
```

Total parameters: **28.4M** (inference: 16.1–23.9M depending on occlusion)

---

## Repository Structure

```
uagmf/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── occlusion_detector.py      # Stage ②: ResNet-18 occlusion classifier
│   │   ├── generative_reconstruction.py # Stage ③: cGAN + VAE
│   │   ├── multimodal_fusion.py       # Stage ④: Cross-attention fusion
│   │   ├── temporal_model.py          # Stage ⑤: Transformer / LSTM
│   │   ├── uq_layer.py                # Stage ⑥: MC Dropout + Deep Ensemble
│   │   ├── symbolic_engine.py         # 18-rule symbolic conflict resolution
│   │   └── uagmf.py                   # Full 7-stage pipeline
│   ├── data/
│   │   ├── __init__.py
│   │   ├── datasets.py                # BioVid, UNBC, EmoPain, MD-NPL loaders
│   │   ├── occlusion_augmentation.py  # Synthetic occlusion generation
│   │   └── preprocessing.py           # Feature extraction pipeline
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                 # Main training loop
│   │   ├── losses.py                  # Huber + reconstruction + UQ NLL
│   │   └── federated.py               # FedAvg + DP-SGD simulation
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                 # MSE, RMSE, MAE, PCC, ICC, QWK, ECE
│   │   └── visualisation.py           # Calibration, occlusion, temporal plots
│   └── utils/
│       ├── __init__.py
│       ├── physio.py                  # HRV, SpO₂, respiratory feature extraction
│       └── logging_utils.py
├── configs/
│   ├── default.yaml
│   ├── biovid.yaml
│   ├── unbc.yaml
│   └── emopain.yaml
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── infer.py                       # Single-sample inference (line 47: discriminator excluded)
│   ├── generate_occlusions.py         # Synthetic occlusion data generator
│   └── federated_sim.py               # Three-site federated simulation
├── rules/
│   └── symbolic_rules.json            # Complete 18-rule symbolic engine rule set
├── tests/
│   ├── test_models.py
│   ├── test_uq.py
│   ├── test_metrics.py
│   └── test_occlusion.py
├── docs/
│   └── data_format.md
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

---

## Installation

```bash
git clone https://github.com/oussama123-ai/uagmf.git
cd uagmf
pip install -r requirements.txt
pip install -e .
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 1.13, CUDA ≥ 11.7

---

## Datasets

| Dataset | Access | Used for |
|---------|--------|---------|
| [BioVid](https://www.nit.ovgu.de/BioVid.html) | Request form | Training / CV |
| [UNBC-McMaster](https://jeffcohn.net/Resources) | Request form | Training / CV (video only) |
| [EmoPain](https://www.ucl.ac.uk/uclic/research/affective-computing/datasets-automatic-affect-recognition/emopain-dataset) | Request form | Training / CV |
| [MD-NPL](https://data.mendeley.com/datasets/mdnpl-dataset) | Public | OOD evaluation only |

See [`docs/data_format.md`](docs/data_format.md) for expected directory layout.

---

## Training

### Single dataset

```bash
python scripts/train.py \
    --config configs/biovid.yaml \
    --data_root /path/to/data \
    --output_dir experiments/biovid_run1
```

### All 5 folds

```bash
for fold in 0 1 2 3 4; do
    python scripts/train.py \
        --config configs/default.yaml \
        --fold $fold \
        --data_root /path/to/data \
        --output_dir experiments/fold${fold}
done
```

### Federated simulation (3 sites: BioVid + UNBC + EmoPain)

```bash
python scripts/federated_sim.py \
    --config configs/default.yaml \
    --data_root /path/to/data \
    --output_dir experiments/federated \
    --rounds 50 --local_epochs 10 \
    --dp_epsilon 8.0 --dp_delta 1e-5
```

---

## Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint experiments/fold0/best_model.pth \
    --data_root /path/to/data \
    --dataset biovid \
    --output_dir results/fold0

# All folds + OOD on MD-NPL
python scripts/evaluate.py \
    --checkpoint experiments/fold0/best_model.pth \
    --all_folds \
    --ood_dataset mdnpl \
    --output_dir results/
```

---

## Inference

```python
from src.models.uagmf import UAGMF

model = UAGMF.from_checkpoint("experiments/best_model.pth")
model.eval()

# video: (1, T, 112, 112, 3) tensor; physio: dict of available signals
output = model(video=video_tensor, hrv=hrv_tensor, spo2=spo2_tensor)

print(f"Pain score: {output['mu']:.2f} ± {output['sigma']:.2f}")
print(f"Alert: {output['alert']}")   # True if σ² > τ* = 0.35
```

**Note on inference script:** The PatchGAN discriminator (0.6M params, training only) is explicitly excluded at inference. See `scripts/infer.py`, line 47.

---

## Uncertainty & Alerting

| Parameter | Value | Description |
|-----------|-------|-------------|
| MC Dropout samples S | 50 | Forward passes with dropout active |
| Ensemble members K | 5 | Independent models averaged |
| Alert threshold τ* | 0.35 | Youden-optimal on dev set (TPR=0.91, FPR=0.12) |
| Residual inflation γ | 0.05 | Couples reconstruction quality to UQ |
| Modality penalty δᵤ | 0.08/modality | Elevates σ² when sensors absent |

The alert threshold τ* = 0.35 (variance) is equivalent to σ > √0.35 ≈ 0.59 (SD units).

---

## Symbolic Rule Engine

The complete 18-rule set is in [`rules/symbolic_rules.json`](rules/symbolic_rules.json).

Three-tier structure:
- **Tier 1**: Concurrent physiological + facial + (optional) acoustic → high confidence
- **Tier 2**: Consensus from any 2 indicator domains → moderate confidence  
- **Tier 3**: Conflicting signals → escalate σ² by +0.15, trigger alert

Rules were designed with clinical domain experts against CPOT/FLACC behavioural indicators.

---

## Federated Security Design

> ⚠️ The federated configuration is a **simulation** using public benchmarks as proxy sites. No real hospital network was established.

| Control | Specification |
|---------|--------------|
| Algorithm | FedAvg, weighted by nₖ |
| Differential privacy | DP-SGD, ε = 8.0, δ = 10⁻⁵ |
| Gradient clipping | ℓ₂ ≤ 1.0 |
| Transport | TLS 1.3 (design-level) |
| Aggregation | SMPC (design-level) |
| Gradient anomaly | KL-divergence, 3σ exclusion |

ε = 8.0 is a relatively loose DP budget chosen to maintain utility across non-IID sites. See Section 6.5 (Privacy budget interpretation) in the paper.

---

## Edge Deployment

| Platform | Latency | Precision | Meets < 100 ms? |
|----------|---------|-----------|----------------|
| Jetson Nano | 93 ms | FP16/TensorRT | ✓ |
| Intel NUC 11 | 41 ms | FP16/TensorRT | ✓ |
| Raspberry Pi 4B | 278 ms | FP32 | ✗ |
| RTX 3090 | 18 ms | FP32 | ✓ |

---

## Citation

```bibtex
@article{elothmani2026uagmf,
  title   = {{UAG-MF}: Uncertainty-aware generative multimodal fusion for
             continuous pain estimation in non-verbal patients under clinical occlusions},
  author  = {El Othmani, Oussama and Naouali, Sami},
  journal = {PLOS Digital Health},
  year    = {2026},
  doi     = {10.1371/journal.pdig.XXXXXXX}
}
```

---

## Important Limitations

- All occlusion robustness results use **synthetically applied** geometric masks; validation under naturally occurring ICU occlusions requires prospective study.
- The federated deployment is a **simulation** across public benchmark datasets; real multi-site deployment requires dedicated IRB approval and governance.
- Validated on **adults only**; neonatal deployment requires dedicated training data and paediatric ethics review.
- ε = 8.0 DP budget should **not** be interpreted as strong privacy protection.
- UAG-MF is a **decision-support tool** only; it must not serve as the sole basis for analgesic administration.

---

## License

MIT License — see [LICENSE](LICENSE).

## Acknowledgements

We thank the creators of BioVid, UNBC-McMaster, EmoPain, and MD-NPL for making their datasets publicly available. Computational resources provided by the Military Research Center, Aouina, Tunisia and King Faisal University, Al Ahsa, Saudi Arabia.

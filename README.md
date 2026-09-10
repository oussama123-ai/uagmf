# UAG-MF: Uncertainty-Aware Generative Multimodal Fusion for Pain Estimation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c.svg)](https://pytorch.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22698497.svg)](https://doi.org/10.5281/zenodo.22698497)

Official implementation of:

> **UAG-MF: Uncertainty-aware generative multimodal fusion for continuous pain estimation in non-verbal patients under clinical occlusions**
> Oussama El Othmani, Sami Naouali
> *Scientific Reports* (under revision, 2026)
> GitHub: https://github.com/oussama123-ai/uagmf

---

## Overview

UAG-MF is a 7-stage end-to-end framework for continuous, objective pain estimation in non-verbal patients (intubated, sedated, cognitively impaired). It addresses three core clinical problems:

| Problem | UAG-MF Solution |
|---------|----------------|
| **Occlusion** (masks, tubes, bandages cover 30–70% of face) | cGAN/VAE generative reconstruction |
| **Uncertainty** (point estimates without confidence mislead) | MC Dropout + Deep Ensemble → µ ± σ with human-in-the-loop alerts |
| **Missing modalities** (sensors fail or unavailable) | Explicit softmax-masking + modality-dropout penalty |

**Key Results — 5-fold subject-independent CV:**

| Dataset | MSE ↓ | PCC ↑ | vs. CNN baseline |
|---------|-------|-------|-----------------|
| BioVid | **1.17 ± 0.05** | **0.87** | −51.3% MSE |
| UNBC-McMaster | **1.33 ± 0.02** | **0.88** | −52.2% MSE |
| EmoPain | **1.71 ± 0.02** | **0.88** | −44.8% MSE |

**Key Results — Leave-One-Subject-Out (LOSO) CV:**

| Dataset | MSE ↓ | PCC ↑ | Gap vs. 5-fold | Folds |
|---------|-------|-------|----------------|-------|
| BioVid | **1.28 ± 0.08** | **0.85** | +9.4% | 87 |
| UNBC-McMaster | **1.45 ± 0.06** | **0.86** | +9.0% | 129 |
| EmoPain | **1.82 ± 0.05** | **0.86** | +6.4% | 60 |

**Additional metrics:**
- ECE = 0.038 (5-fold) / 0.047 (LOSO) — well-calibrated uncertainty
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

**Total parameters:** 28.4M (inference: 16.1–23.9M depending on occlusion level)

---

## Repository Structure

```
uagmf/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── occlusion_detector.py        # Stage ②: ResNet-18 occlusion classifier
│   │   ├── generative_reconstruction.py # Stage ③: cGAN + VAE
│   │   ├── multimodal_fusion.py         # Stage ④: Cross-attention fusion
│   │   ├── temporal_model.py            # Stage ⑤: Transformer / LSTM
│   │   ├── uq_layer.py                  # Stage ⑥: MC Dropout + Deep Ensemble
│   │   ├── symbolic_engine.py           # 18-rule symbolic conflict resolution
│   │   └── uagmf.py                     # Full 7-stage pipeline
│   ├── data/
│   │   ├── __init__.py
│   │   ├── datasets.py                  # BioVid, UNBC, EmoPain, MD-NPL loaders + get_dataset()
│   │   ├── occlusion_augmentation.py    # Synthetic occlusion generation
│   │   └── preprocessing.py             # Feature extraction pipeline
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                   # Main training loop
│   │   ├── losses.py                    # Huber + reconstruction + UQ NLL
│   │   ├── federated.py                 # FedAvg + DP-SGD simulation
│   │   └── loso.py                      # ★ Leave-One-Subject-Out cross-validation
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                   # MSE, RMSE, MAE, PCC, ICC, QWK, ECE
│   │   └── visualisation.py             # Calibration, occlusion, temporal plots
│   └── utils/
│       ├── __init__.py
│       ├── physio.py                    # HRV, SpO₂, respiratory feature extraction
│       └── logging_utils.py
├── configs/
│   ├── default.yaml
│   ├── biovid.yaml
│   ├── unbc.yaml
│   ├── emopain.yaml
│   └── loso.yaml                        # ★ LOSO configuration
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── infer.py                         # Single-sample inference (line 47: discriminator excluded)
│   ├── generate_occlusions.py           # Synthetic occlusion data generator
│   ├── federated_sim.py                 # Three-site federated simulation
│   ├── train_loso.py                    # ★ LOSO training script
│   └── generate_fig8_loso.py            # ★ Figure 8 (LOSO results) generator
├── rules/
│   └── symbolic_rules.json              # Complete 18-rule symbolic engine rule set
├── tests/
│   ├── test_models.py
│   ├── test_uq.py
│   ├── test_metrics.py
│   └── test_occlusion.py
├── docs/
│   └── data_format.md
├── results/
│   └── loso/                            # LOSO per-fold outputs
├── figures_corrected/                   # Generated figures (fig1–fig8)
├── .zenodo.json                         # ★ Zenodo metadata
├── CITATION.cff                         # ★ Citation file
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

### Single dataset (5-fold CV)

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

## Leave-One-Subject-Out (LOSO) Cross-Validation

LOSO is the gold-standard protocol for clinical AI generalization to previously unseen individuals. Each subject serves as the sole test subject exactly once, with all remaining subjects used for training.

```bash
# BioVid (87 folds — one per subject)
python scripts/train_loso.py \
    --config configs/loso.yaml \
    --dataset biovid \
    --data_root /path/to/data \
    --output_dir results/loso

# UNBC-McMaster (129 folds)
python scripts/train_loso.py \
    --config configs/loso.yaml \
    --dataset unbc \
    --data_root /path/to/data \
    --output_dir results/loso

# EmoPain (60 folds)
python scripts/train_loso.py \
    --config configs/loso.yaml \
    --dataset emopain \
    --data_root /path/to/data \
    --output_dir results/loso
```

**Results (LOSO):**

| Dataset | MSE (5-fold) | MSE (LOSO) | Gap | PCC (LOSO) | ECE (LOSO) |
|---------|-------------|------------|-----|------------|------------|
| BioVid | 1.17 | **1.28 ± 0.08** | +9.4% | 0.85 ± 0.02 | 0.047 |
| UNBC-McMaster | 1.33 | **1.45 ± 0.06** | +9.0% | 0.86 ± 0.02 | 0.047 |
| EmoPain | 1.71 | **1.82 ± 0.05** | +6.4% | 0.86 ± 0.02 | 0.047 |

Under LOSO, UAG-MF still outperforms all baselines:
- **+31.9%** vs. CNN baseline (BioVid)
- **+23.8%** vs. best multimodal baseline (BioVid)

### Generate Figure 8 (LOSO results)

```bash
python scripts/generate_fig8_loso.py \
    --results_dir results/loso \
    --out_dir figures_corrected
```

Produces `figures_corrected/fig8_loso.pdf` and `figures_corrected/fig8_loso.png`.

---

## Evaluation

```bash
# Single fold
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

**Per-component latency breakdown** (Jetson Nano, FP16/TensorRT 8.4):

| Component | Passes | Latency (ms) |
|-----------|--------|-------------|
| Feature extraction (shared backbone) | 1 | 14 |
| Generative reconstruction (cGAN/VAE) | 1 | 21 |
| MC Dropout forward passes | S=50 | 31 |
| Deep Ensemble forward passes | K=5 | 22 |
| Fusion + output head | 1 | 5 |
| **Total** | — | **93** |

---

## Results Summary

### Overall Performance (5-fold CV)

| Method | BioVid MSE | UNBC MSE | EmoPain MSE | BioVid PCC | UNBC PCC | EmoPain PCC |
|--------|-----------|----------|-------------|-----------|----------|-------------|
| CNN (Unimodal) | 2.40 | 2.78 | 3.10 | 0.61 | 0.64 | 0.60 |
| LSTM (Physio) | 2.15 | 2.50 | 2.85 | 0.66 | 0.68 | 0.65 |
| Late Fusion | 1.87 | 2.12 | 2.45 | 0.74 | 0.76 | 0.73 |
| Transformer Fusion | 1.62 | 1.90 | 2.20 | 0.80 | 0.82 | 0.79 |
| SS-Multimodal (SimCLR) | 1.58 | 1.85 | 2.14 | 0.81 | 0.83 | 0.80 |
| **UAG-MF (Ours)** | **1.17** | **1.33** | **1.71** | **0.87** | **0.88** | **0.88** |

### LOSO Cross-Validation Results

| Method | BioVid MSE | UNBC MSE | EmoPain MSE | BioVid PCC | UNBC PCC | EmoPain PCC |
|--------|-----------|----------|-------------|-----------|----------|-------------|
| CNN (Unimodal) | 2.89 | 3.21 | 3.54 | 0.58 | 0.61 | 0.57 |
| LSTM (Physio) | 2.58 | 2.91 | 3.22 | 0.63 | 0.65 | 0.62 |
| Late Fusion | 2.24 | 2.48 | 2.81 | 0.71 | 0.73 | 0.70 |
| Transformer Fusion | 1.94 | 2.21 | 2.52 | 0.77 | 0.79 | 0.76 |
| SS-Multimodal (SimCLR) | 1.88 | 2.14 | 2.44 | 0.78 | 0.80 | 0.77 |
| **UAG-MF (Ours)** | **1.28** | **1.45** | **1.82** | **0.85** | **0.86** | **0.86** |

### Ablation Study (BioVid)

| Configuration | MSE | PCC | ICC | ECE |
|---------------|-----|-----|-----|-----|
| Visual only | 2.45 | 0.640 | 0.610 | 0.098 |
| + Physio | 1.95 | 0.720 | 0.690 | 0.087 |
| + Temporal | 1.72 | 0.790 | 0.760 | 0.074 |
| + Generative reconstruction | 1.42 | 0.850 | 0.820 | 0.063 |
| + Symbolic engine | 1.31 | 0.858 | 0.831 | 0.055 |
| **+ UQ (Full UAG-MF)** | **1.17** | **0.870** | **0.854** | **0.038** |

### Occlusion Robustness (BioVid, synthetic)

| Method | 0% | 20% | 40% | 60% | 80% |
|--------|-----|-----|-----|-----|-----|
| CNN (Unimodal) | 2.40 | 2.93 | 3.67 | 4.82 | 6.14 |
| Transformer Fusion | 1.62 | 2.11 | 2.95 | 3.98 | 5.40 |
| **UAG-MF (Ours)** | **1.17** | **1.39** | **1.64** | **1.98** | **2.57** |

### Out-of-Distribution (MD-NPL, neonatal, no fine-tuning)

| Condition | Accuracy | QWK | AUC |
|-----------|----------|-----|-----|
| Majority class (chance) | 35.0% | 0.000 | 0.500 |
| Random classifier | 25.0% | 0.000 | 0.500 |
| Supervised on MD-NPL (upper bound) | 88.4% | 0.841 | 0.937 |
| **UAG-MF (OOD, no fine-tuning)** | **79.8%** | **0.761** | **0.891** |

---

## Reproducibility

All results reported in the manuscript are fully reproducible:

- Random seeds fixed via `src/utils/logging_utils.py`
- Per-fold 5-fold results → `results/all_results.json`
- LOSO per-fold results → `results/loso/loso_results.json`
- Figures generated by `scripts/generate_fig8_loso.py`
- Complete rule set archived in `rules/symbolic_rules.json`
- Zenodo archive: [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)

---

## Important Limitations

- All occlusion robustness results use **synthetically applied** geometric masks; validation under naturally occurring ICU occlusions requires prospective study.
- The federated deployment is a **simulation** across public benchmark datasets; real multi-site deployment requires dedicated IRB approval and governance.
- Validated on **adults only**; neonatal deployment requires dedicated training data and paediatric ethics review.
- ε = 8.0 DP budget should **not** be interpreted as strong privacy protection.
- The MAML-style longitudinal adaptation module was **not activated** in any reported benchmark evaluation; it is retained in the codebase as an architectural provision for future neonatal deployment.
- UAG-MF is a **decision-support tool** only; it must not serve as the sole basis for analgesic administration.

---

## Citation

If you use this code or the UAG-MF framework in your research, please cite:

```bibtex
@article{elothmani2026uagmf,
  title   = {UAG-MF: Uncertainty-aware generative multimodal fusion for
             continuous pain estimation in non-verbal patients under
             clinical occlusions},
  author  = {El Othmani, Oussama and Naouali, Sami},
  journal = {Scientific Reports},
  year    = {2026},
  note    = {Under revision}
}
```

For the code repository specifically, please also cite the Zenodo archive:

```bibtex
@software{elothmani2026uagmf_code,
  title     = {UAG-MF: Uncertainty-Aware Generative Multimodal Fusion
               for Pain Estimation},
  author    = {El Othmani, Oussama and Naouali, Sami},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://github.com/oussama123-ai/uagmf}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contact

- **Oussama El Othmani** — oussama.elothmani@ept.u-carthage.tn
- **Sami Naouali** (corresponding author) — salnawali@kfu.edu.sa

---

## Acknowledgements

We thank the creators of BioVid, UNBC-McMaster, EmoPain, and MD-NPL for making their datasets publicly available. Computational resources provided by the Military Research Center, Aouina, Tunisia and King Faisal University, Al Ahsa, Saudi Arabia.

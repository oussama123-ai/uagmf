# UAG-MF: Uncertainty-Aware Generative Multimodal Fusion for Pain Estimation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c.svg)](https://pytorch.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

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

# Near–Real-Time Conflict-Related Fire Detection Using Unsupervised Deep Learning

[![tests](https://github.com/heykuldip/nrt-conflict-monitoring/actions/workflows/ci.yml/badge.svg)](https://github.com/heykuldip/nrt-conflict-monitoring/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23522614.svg)](https://doi.org/10.5281/zenodo.23522614)

##### Based on the RaVAEn model: https://github.com/spaceml-org/RaVAEn

**Official implementation of the paper:**

> **Near-real-time conflict-related fire detection in Sudan using unsupervised deep learning**
> *Science of Remote Sensing* 13 (2026) 100446. https://doi.org/10.1016/j.srs.2026.100446
> **Authors:** Kuldip Singh Atwal, Dieter Pfoser, Daniel Rothbart (George Mason University)

---

## Abstract

This repository contains the official implementation of the study **“Near–Real-Time Conflict-Related Fire Detection Using Unsupervised Deep Learning.”** We adapted and retrained the **RaVAEn** framework to detect conflict-related fires and burn scars in Sudan using high-resolution satellite imagery with a **resolution-adaptive Variational Autoencoder (VAE)**. It is archived on Zenodo as 10.5281/zenodo.23522615.

Unlike the original RaVAEn model designed for coarse-resolution disaster monitoring on-board satellites, this version is optimized for **3 m resolution PlanetScope imagery**, enabling detection of small, fragmented fires in active war zones. The model employs a lightweight, unsupervised approach that learns generalized visual primitives from diverse land-cover data (WorldFloods) and identifies fire-affected areas as anomalies in the latent space. Under favorable observational conditions, the system delivers actionable insights within **24–30 hours** of image acquisition.

---

## Key Features

- **Resolution-Adaptive Architecture:** Incorporates *Frequency Decomposition*, *Multi-scale Dilated Convolutions*, and *BlurPool (anti-aliased downsampling)* to handle domain shifts between training data (10 m Sentinel-2) and inference data (3 m PlanetScope).

- **Unsupervised Learning:** No labeled fire data required. The model learns nominal surface representations and flags deviations via latent-space cosine distance.

- **Lightweight & Fast:** Designed for operational near–real-time monitoring with end-to-end latency of approximately **24–30 hours**.

- **Data-Scarce Robustness:** Validated on verified conflict incidents in Sudan (e.g., El Fasher, Gandahar Market) where ground truth is inaccessible.

---

## Methodology

### 1. Preprocessing and Spectral Alignment

The pipeline processes **4-band PlanetScope imagery** (Red, Green, Blue, NIR).

- **Normalization:** Inputs are log-transformed to manage high dynamic range and scaled to the `[-1, 1]` interval using the 1st and 99th percentiles of the training data.

- **Spectral alignment:** PlanetScope bands are aligned to the Sentinel-2 reference with per-band gains and offsets from major-axis regression on near-coincident acquisitions.

- **Tiling:** Imagery is partitioned into non-overlapping **32 × 32 pixel tiles**.

---

### 2. Model Architecture
The core model, `ResolutionAdaptiveVAE` in `src/models/ae_vae_models/deeper_vae.py`, is a modified Variational Autoencoder that:

1. **Decomposes** inputs into low-frequency structural components and high-frequency texture components.
2. **Encodes** features using multi-scale atrous (dilated) convolutions to capture context at varying resolutions.
3. **Projects** representations into a **128-dimensional latent space**.

This design enables robustness to sensor resolution changes and texture fragmentation common in conflict environments.

---

### 3. Anomaly Detection

Inference compares **temporally paired tiles** (Pre-incident vs. Post-incident).

- **Metric:** Cosine distance between latent means <b>μ</b><sub>pre</sub> and <b>μ</b><sub>post</sub>.

- **Baseline:** Pixel-wise cosine distance, CVA, IR-MAD, dNDVI, and dBAI are included as scientific controls.

---

## Dataset

### Training Data — WorldFloods

The model is pre-trained on the **WorldFloods** dataset derived from Sentinel-2 imagery to learn diverse spectral and textural patterns.

- **Source:** https://github.com/spaceml-org/ml4floods
- **Adaptation:** Stochastic scale augmentation is applied during training to simulate the resolution shift from Sentinel-2 (10 m) to PlanetScope (3 m).

---

### Inference Data — PlanetScope

Commercial **PlanetScope Level 3B Ortho Scene (Analytic, 4-band)** imagery is used for inference.

- **Access:** Due to licensing restrictions, PlanetScope imagery for Sudan cannot be shared publicly. Researchers need to obtain licenses directly from Planet Labs.

- **Case Studies:** Metadata for the five Sudan conflict incidents (dates and AOIs) is provided in the manuscript.

---

## Usage

Tested with Python 3.9, PyTorch 2.1, PyTorch Lightning 1.3.8, Hydra 1.1, rasterio 1.2, scikit-learn 1.5, SciPy 1.10, and pandas 1.4; training logs to Weights & Biases (`wandb`). Run all commands from the repository root. Training expects RaVAEn's `worldfloods_change_no_duplicates` dataset in `data/`.

```bash
# Normalisation parameters from the training data
python -m scripts.compute_percentiles +dataset=alpha_multiscene +channels=rgb_nir +normalisation=raw \
    +training=da +transform=no_augmentation +out=config/normalisation/log_percentiles.yaml

# Training
python -m scripts.train_model +dataset=alpha_multiscene +channels=rgb_nir +normalisation=log_percentiles \
    +module=deeper_vae +training=da +transform=scale +project=conflict_monitoring

# Change map between two co-registered scenes
python -m scripts.infer CHECKPOINT.ckpt BEFORE.tif AFTER.tif change_map.tif \
    --normalisation config/normalisation/log_percentiles.yaml [--alignment gains_offsets.yaml]

# Evaluation for one site: metrics.csv (Tables 2–6, 8), effects.csv (Table 7), spearman.csv (Table 9)
python -m scripts.evaluate POST.tif LABELS.tif results/SITE --pre PRE.tif [PRE.tif ...] \
    --checkpoint CHECKPOINT.ckpt --normalisation config/normalisation/log_percentiles.yaml
```

The alignment file holds per-band `gains` and `offsets` from `src.data.alignment.fit_bands`. Several `--pre` scenes give the time-series configuration.

---

## Citation

If you use this code, model architecture, or methodology in your research, please cite:

```bibtex
@article{atwal2026near,
  title   = {Near-real-time conflict-related fire detection in Sudan using unsupervised deep learning},
  author  = {Atwal, Kuldip Singh and Pfoser, Dieter and Rothbart, Daniel},
  journal = {Science of Remote Sensing},
  volume  = {13},
  pages   = {100446},
  year    = {2026},
  doi     = {10.1016/j.srs.2026.100446}
}
```
---

## Acknowledgements

This work was conducted as part of the Sudan Conflict Observatory (SCO). The researchers involved in the SCO project include the authors of this article as well as the following analysts: Moneim Adam, Mathieu Bere, Hind Fadul, Anusha Srirenganathanmalarvizhi, Zifu Wang, David Wong and Chaowei Yang.

Computing resources were provided by the Office of Research Computing at George Mason University (https://orc.gmu.edu).

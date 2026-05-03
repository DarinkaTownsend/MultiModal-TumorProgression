# Safe Multimodal Tumor Progression Forecasting via Mask Prediction from Longitudinal Brain MRI

**CAP 5516 Medical Image Computing — Final Project**
University of Central Florida

**Obinsonne Servius** (ob675831@ucf.edu) · **Darinka Townsend** (dtownsend@ucf.edu)

---

## Overview

This project implements a dual-task multimodal framework for brain tumor analysis from longitudinal MRI.
Given a baseline brain MRI scan and associated clinical text, the system:

- **Severity classification** (Obinsonne Servius): predicts tumor severity as low, mid, or high using a ViT image encoder fused with a DistilBERT clinical text encoder via a learned reliability gate and Monte Carlo Dropout uncertainty estimation
- **Future mask forecasting** (Darinka Townsend): predicts the tumor segmentation mask at a future timepoint

---

## Dataset

We use the [UCSF Adult Longitudinal Post-Treatment Diffuse Glioma MRI Dataset](https://imagingdatasets.ucsf.edu/dataset/2), which contains longitudinal multimodal MRI scans (T1, T1ce, T2, FLAIR) from 286 adult glioma patients with expert segmentation masks and clinical metadata.

Severity labels are derived from WHO grade: grade 1 and 2 map to low (0), grade 3 to mid (1), grade 4 to high (2).

---

## File Structure

```
.
├── preprocess.py       # extracts 2D axial slices from NIfTI volumes, builds manifest.csv
├── dataset.py          # PyTorch Dataset class for severity classification
├── model.py            # ViT + DistilBERT + reliability-gated fusion + MC Dropout head
├── train.py            # training script (end-to-end, all parameters)
└── train_frozen.py     # training script with frozen encoders + early stopping
```

---

## Setup

Tested on Python 3.11.4, PyTorch 2.5.1 (cu121), UCF Newton HPC cluster (Tesla V100).

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy timm transformers scikit-learn tqdm matplotlib
```

---

## Preprocessing

```bash
python3 preprocess.py --dataDir /path/to/ucsf/dataset --outDir processed/
```

This extracts tumor-containing 2D axial slices and writes a `manifest.csv` with paths, severity labels, and clinical text strings. Clinical text includes age, sex, IDH status, MGMT, 1p/19q, ATRX, treatment, and interval between scans. Grade and diagnosis tokens are excluded to prevent label leakage.

---

## Training

**Full training (all parameters):**
```bash
python3 train.py --manifest processed/manifest_clean.csv --epochs 20 --batchSize 32 --lr 1e-4
```

**Frozen encoders with early stopping (recommended):**
```bash
python3 train_frozen.py --manifest processed/manifest_clean.csv --epochs 30 --patience 5
```

Outputs are saved to `output/`, including `best_model.pt` and `results.csv`.

---

## Results

Best model (epoch 2, full training):

| Metric | Score |
|--------|-------|
| Accuracy | 0.8192 |
| Macro F1 | 0.7425 |
| AUROC | 0.8395 |
| ECE | 0.1790 |

Training used a patient-level 80/20 train/test split across 286 patients (~19,000 total 2D slices).

---

## Newton HPC

To allocate a GPU node on the UCF Newton cluster:

```bash
srun --account=course_cap5516 --partition=normal --qos=course_cap5516 --gres=gpu:1 --mem=32G --time=8:00:00 --pty bash
module load python/python-3.11.4-gcc-12.2.0
source venv/bin/activate
```

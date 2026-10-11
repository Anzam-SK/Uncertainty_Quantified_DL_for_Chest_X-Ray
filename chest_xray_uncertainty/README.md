# Uncertainty-Quantified Deep Learning Framework for Reliable Chest X-Ray Diagnosis

**Author:** Anzam Shahriar Kabir (2022521460123)
**Thesis Project** — Deep Learning for Medical Image Analysis with Uncertainty Quantification

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Theoretical Background](#2-theoretical-background)
3. [System Requirements](#3-system-requirements)
4. [Installation Guide](#4-installation-guide)
5. [Dataset Setup](#5-dataset-setup)
6. [Project Architecture](#6-project-architecture)
7. [Module-by-Module Documentation](#7-module-by-module-documentation)
8. [Configuration Reference](#8-configuration-reference)
9. [Usage Guide](#9-usage-guide)
10. [Testing](#10-testing)
11. [Expected Results](#11-expected-results)
12. [Troubleshooting](#12-troubleshooting)
13. [References](#13-references)

---

## 1. Project Overview

### What This Project Does

This thesis project implements and evaluates **uncertainty-aware deep learning** for multi-label chest X-ray classification. It builds a complete pipeline — from data preprocessing through training, evaluation, and out-of-distribution (OOD) analysis — to answer a critical question:

> **Can Monte Carlo Dropout uncertainty estimates help clinicians identify when a deep learning model's predictions are unreliable?**

### Core Contribution

Standard deep learning models output a single prediction per input with no indication of confidence. This is dangerous in medical imaging, where an overconfident wrong prediction can lead to misdiagnosis. This project addresses this by:

1. **Implementing MC Dropout** — a Bayesian approximation technique that runs multiple stochastic forward passes through a dropout-enabled network to estimate prediction uncertainty.
2. **Comparing against a baseline** — a standard ResNet-50 without uncertainty quantification.
3. **Evaluating calibration** — measuring whether the model's confidence aligns with its actual accuracy using Expected Calibration Error (ECE).
4. **Testing OOD detection** — verifying that the model produces higher uncertainty on out-of-distribution data (Covid-X images) compared to in-distribution data (NIH ChestX-ray14).

### The Two Models

| Model | Architecture | Uncertainty Method | Purpose |
|-------|-------------|-------------------|---------|
| **Baseline** | ResNet-50 | None (standard inference) | Comparison reference |
| **MC Dropout** | DenseNet-121 | Monte Carlo Dropout (T=50 passes) | Uncertainty-quantified predictions |

### Disease Classes (14 total)

The model performs multi-label classification across 14 thoracic pathologies:

`Atelectasis` · `Consolidation` · `Infiltration` · `Pneumothorax` · `Edema` · `Emphysema` · `Fibrosis` · `Effusion` · `Pneumonia` · `Pleural_Thickening` · `Cardiomegaly` · `Nodule` · `Mass` · `Hernia`

Each image can have zero, one, or multiple diseases present (binary multi-label setting).

---

## 2. Theoretical Background

### Monte Carlo Dropout (MC Dropout)

MC Dropout was proposed by Gal & Ghahramani (2016) as a practical approximation to Bayesian neural networks. The key insight:

> **Keeping dropout active during inference and running multiple forward passes produces a distribution of predictions that approximates the posterior distribution.**

**How it works:**

1. **Training:** A standard neural network is trained with dropout (random neuron deactivation) as regularization.
2. **Inference:** Instead of disabling dropout (as is standard), dropout remains **active** during prediction.
3. **Multiple passes:** The same input is fed through the network **T times** (e.g., T=50), each time with different random dropout masks.
4. **Aggregation:** The T outputs are aggregated:
   - **Mean prediction** = average of T sigmoid outputs → final probability per class
   - **Uncertainty** = variance of T sigmoid outputs → epistemic uncertainty per class
   - **Predictive entropy** = entropy of the mean prediction → overall uncertainty per sample

**Why this matters for medical imaging:**

- A model that sees a clear pneumonia pattern will produce similar outputs across all T passes → **low uncertainty** → trustworthy prediction.
- A model that sees an ambiguous or unusual image will produce highly variable outputs across passes → **high uncertainty** → flag for human review.

### Calibration

A well-calibrated model's predicted probabilities should match actual frequencies. For example, if the model predicts 80% probability for a disease across 100 cases, roughly 80 of those cases should actually have the disease.

**Expected Calibration Error (ECE)** measures this by:
1. Dividing predictions into bins by confidence level
2. Computing the difference between average confidence and average accuracy in each bin
3. Taking the weighted average of these differences

Lower ECE = better calibration = more trustworthy probabilities.

### Out-of-Distribution Detection

OOD detection tests whether the model produces higher uncertainty on data from a different distribution than its training data. If MC Dropout works correctly:
- **In-distribution** (NIH ChestX-ray14 test set): Lower uncertainty
- **OOD** (Covid-X images): Higher uncertainty

The **separation ratio** (OOD mean uncertainty / in-distribution mean uncertainty) quantifies this. A ratio > 1.5 indicates meaningful OOD detection capability.

---

## 3. System Requirements

### Hardware

| Component | Minimum | Recommended | Tested Configuration |
|-----------|---------|-------------|---------------------|
| **GPU** | Any CUDA-capable GPU | NVIDIA RTX 3060+ (8GB+ VRAM) | NVIDIA RTX 4050 Laptop (6GB VRAM) |
| **CPU** | 4 cores | 8+ cores | Intel/AMD x86_64 |
| **RAM** | 8 GB | 16 GB | 16 GB |
| **Storage** | 100 GB free | 150 GB SSD | NVMe SSD |

### Software

| Component | Required Version | Notes |
|-----------|-----------------|-------|
| **Python** | 3.10 or 3.11 | 3.12+ may have compatibility issues with some dependencies |
| **CUDA** | 12.1 or higher | Required for GPU training; CUDA 11.x does NOT fully support RTX 4050 (sm_89) |
| **NVIDIA Driver** | 525.x or higher | Must support CUDA 12.1+ |
| **OS** | Windows 10/11, Linux, macOS | Windows tested; Linux recommended for production |

### GPU Compute Capability Reference

| GPU Series | Architecture | Compute Capability | CUDA 12.1 Support |
|-----------|-------------|-------------------|-------------------|
| RTX 40xx | Ada Lovelace | sm_89 | ✅ Yes |
| RTX 30xx | Ampere | sm_86 | ✅ Yes |
| RTX 20xx | Turing | sm_75 | ✅ Yes |
| GTX 16xx | Turing | sm_75 | ✅ Yes |

---

## 4. Installation Guide

### Step 1: Create Virtual Environment

```bash
# Navigate to project root
cd chest_xray_uncertainty

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Verify activation (prompt should show (venv))
python --version
pip --version
```

### Step 2: Install CUDA-Enabled PyTorch (Critical Order)

> **⚠️ IMPORTANT:** PyTorch with CUDA must be installed BEFORE other dependencies. Installing from `requirements.txt` first will pull the CPU-only version.

```bash
# For CUDA 12.1 (recommended for RTX 4050):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 12.4 (alternative if cu121 has issues):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**Verify CUDA is working:**

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); print('Compute capability:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'N/A')"
```

Expected output:
```
PyTorch: 2.x.x+cu121
CUDA available: True
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
Compute capability: (8, 9)
```

### Step 3: Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- **torch-uncertainty** — uncertainty quantification library
- **numpy, pandas, scikit-learn** — data processing and metrics
- **opencv-python** — image processing (CLAHE)
- **matplotlib, seaborn** — visualization
- **tqdm** — progress bars
- **PyYAML** — configuration file parsing
- **tensorboard** — training visualization
- **Pillow** — image loading
- **lightning** — PyTorch Lightning (torch-uncertainty dependency)

### Step 4: Verify Installation

```bash
python -c "import torch, cv2, sklearn, numpy, pandas, yaml; print('All imports OK')"
```

---

## 5. Dataset Setup

### NIH ChestX-ray14 Dataset

The primary dataset for this project.

**Source:** [NIH ChestX-ray14 on Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data) (~42 GB)

**What it contains:**
- 112,120 frontal-view chest X-ray images from 30,805 unique patients
- 14 disease labels (binary: present/absent) extracted from radiology reports via NLP
- Labels are **noisy** — NLP extraction is imperfect, which is why uncertainty quantification is valuable

**Setup:**

1. Download the dataset from Kaggle
2. Extract all images to `data/raw/images/` (flat directory of `.png` files)
3. Place the split files in `data/splits/`:
   - `train_val_list.txt` — list of training/validation image filenames (one per line)
   - `test_list.txt` — list of test image filenames (one per line)
4. Place `Data_Entry_2017.csv` in `data/` — contains the image-to-label mapping with columns:
   - `Image Index` — filename (e.g., `00000001_000.png`)
   - One column per disease class with binary values (0/1)

### Covid-X Dataset (OOD Evaluation)

Used to test whether the model produces higher uncertainty on out-of-distribution data.

**Source:** [COVID-Net GitHub](https://github.com/lindawangg/COVID-Net)

**Setup:**

1. Download Covid-X test images
2. Place images in `data/raw/covidx/`
3. Create `data/splits/covidx_test.csv` with a `filename` column listing all image filenames

### Expected Directory Layout After Setup

```
chest_xray_uncertainty/
├── data/
│   ├── raw/
│   │   ├── images/                    # ~112,120 NIH chest X-ray PNG images
│   │   │   ├── 00000001_000.png
│   │   │   ├── 00000001_001.png
│   │   │   └── ...
│   │   └── covidx/                    # Covid-X test images
│   │       ├── covid-image-1.png
│   │       └── ...
│   ├── splits/
│   │   ├── train_val_list.txt         # Training/validation split
│   │   ├── test_list.txt              # Test split
│   │   └── covidx_test.csv            # Covid-X file list
│   └── processed/                     # Created by preprocessing script
│       └── images/                    # CLAHE-enhanced images
├── experiments/
│   ├── checkpoints/                   # Saved model weights
│   │   ├── baseline/
│   │   │   └── best_model.pth
│   │   └── mc_dropout/
│   │       └── best_model.pth
│   ├── logs/                          # TensorBoard event files
│   └── results/                       # Evaluation outputs (JSON, NPY, PNG)
```

---

## 6. Project Architecture

### Directory Structure

```
chest_xray_uncertainty/
├── config/
│   └── default.yaml                   # All hyperparameters and paths
├── data/
│   ├── __init__.py
│   ├── dataset.py                     # Dataset classes and DataLoader factory
│   ├── transforms.py                  # CLAHE transform and augmentation pipelines
│   └── splits/                        # User-provided split files
├── models/
│   ├── __init__.py
│   ├── densenet.py                    # MC Dropout DenseNet-121 model
│   └── resnet_baseline.py             # Standard ResNet-50 baseline
├── uncertainty/
│   ├── __init__.py
│   ├── mc_dropout.py                  # MC Dropout inference functions
│   └── metrics.py                     # ECE, NLL, AUC computation
├── training/
│   ├── __init__.py
│   ├── trainer.py                     # Full training loop with early stopping
│   └── losses.py                      # Weighted BCE loss for class imbalance
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py                   # Test-set evaluation with MC Dropout
│   └── ood_evaluator.py               # OOD uncertainty analysis
├── visualization/
│   ├── __init__.py
│   └── plots.py                       # All plotting functions
├── scripts/
│   ├── preprocess.py                  # CLAHE preprocessing script
│   ├── train.py                       # Training entry point
│   ├── evaluate.py                    # Evaluation entry point
│   └── evaluate_ood.py                # OOD analysis entry point
├── tests/
│   ├── test_dataset.py                # Dataset unit tests (9 tests)
│   ├── test_model.py                  # Model unit tests (8 tests)
│   └── test_uncertainty.py            # Uncertainty unit tests (11 tests)
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

### Data Flow

```
Raw Images → [preprocess.py] → CLAHE-enhanced Images
                                      ↓
CLAHE Images → [dataset.py] → DataLoader (with augmentation)
                                      ↓
DataLoader → [trainer.py] → Model Training (with TensorBoard logging)
                                      ↓
Trained Model → [evaluator.py] → MC Dropout Inference (T=50 passes)
                                      ↓
Predictions + Uncertainty → [metrics.py] → AUC, ECE, NLL
                                      ↓
Results → [plots.py] → Reliability Diagrams, Uncertainty Distributions
                                      ↓
OOD Data → [ood_evaluator.py] → Separation Analysis
```

---

## 7. Module-by-Module Documentation

### 7.1 `config/default.yaml` — Configuration

Single source of truth for all hyperparameters. Every module reads from this file — no hardcoded values anywhere.

**Sections:**

- **`data`** — Dataset paths, image size, CLAHE parameters, DataLoader settings
- **`model`** — Backbone architecture, number of classes, dropout rate, pretrained flag, freeze epochs
- **`training`** — Epochs, batch size, learning rate, weight decay, early stopping, LR scheduler, mixed precision
- **`uncertainty`** — Number of MC Dropout passes (T=50 for full, T=20 for quick experiments)
- **`evaluation`** — Output directory, calibration bins
- **`disease_classes`** — Ordered list of 14 disease names

### 7.2 `data/transforms.py` — Image Transforms

**`ClaheTransform`** — Custom transform that applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance contrast in chest X-rays.

- Converts PIL Image to grayscale numpy array
- Applies `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))`
- Converts back to RGB PIL Image (grayscale stacked into 3 channels)
- CLAHE is applied BEFORE any augmentation or normalization

**`get_train_transforms(image_size)`** — Training pipeline with augmentation:
1. CLAHE enhancement
2. Random horizontal flip (p=0.5)
3. Random rotation (±10 degrees)
4. Resize to (image_size, image_size)
5. Convert to tensor
6. Normalize with ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

**`get_val_transforms(image_size)`** — Validation/test pipeline (no augmentation):
1. CLAHE enhancement
2. Resize to (image_size, image_size)
3. Convert to tensor
4. Normalize with ImageNet statistics

### 7.3 `data/dataset.py` — Dataset Classes

**`ChestXrayDataset`** — Main dataset for NIH ChestX-ray14.

- Loads images from disk using PIL
- Reads labels from CSV (binary multi-label: 14 classes)
- Returns `(image_tensor, label_tensor, filename)` tuples
- `get_pos_weights()` computes class-balanced weights for BCE loss: `pos_weight[i] = N_negative / N_positive`
- Handles missing images gracefully (returns zero tensor + zero labels, logs warning)

**`CovidXDataset`** — OOD dataset for Covid-X images.

- Loads images without labels (used only for uncertainty measurement)
- Returns `(image_tensor, filename)` tuples

**`create_dataloaders(config)`** — Factory function that creates train/val/test DataLoaders.

- Splits NIH dataset into train (80%) and val (20%) from the training list
- Uses separate test list for test set
- Applies appropriate transforms to each split
- Configurable batch size, num_workers, and pin_memory

### 7.4 `models/densenet.py` — MC Dropout Model

**`MCDropoutDenseNet`** — DenseNet-121 with MC Dropout for uncertainty quantification.

**Architecture:**
```
Input (3, 224, 224)
    ↓
DenseNet-121 Feature Extractor (pretrained on ImageNet)
    ↓
Global Average Pooling → 1024 features
    ↓
Dropout (p=0.3)
    ↓
Linear (1024 → 512) + ReLU
    ↓
Dropout (p=0.3)
    ↓
Linear (512 → 14) → Raw logits
```

**Key methods:**
- `freeze_backbone()` — Freezes all DenseNet feature extractor parameters (used during initial training)
- `unfreeze_backbone()` — Unfreezes feature extractor for fine-tuning
- `enable_mc_dropout()` — Sets ONLY Dropout layers to training mode while keeping everything else in eval mode. This is the critical method for MC Dropout inference.
- `forward(x)` — Returns raw logits (no sigmoid); loss function handles activation

### 7.5 `models/resnet_baseline.py` — Baseline Model

**`ResNetBaseline`** — Standard ResNet-50 without MC Dropout.

**Architecture:**
```
Input (3, 224, 224)
    ↓
ResNet-50 Feature Extractor (pretrained on ImageNet)
    ↓
Global Average Pooling → 2048 features
    ↓
Linear (2048 → 14) → Raw logits
```

- No dropout in the classifier head
- Same `freeze_backbone()` / `unfreeze_backbone()` interface
- Used as comparison baseline to quantify the value of uncertainty estimates

### 7.6 `uncertainty/mc_dropout.py` — MC Dropout Inference

**`mc_predict(model, image_batch, n_passes, device)`** — Core MC Dropout inference function.

**Algorithm:**
1. Set model to eval mode, then enable MC Dropout (dropout active, batch norm frozen)
2. Run `n_passes` forward passes on the same input (each pass uses different dropout mask)
3. Apply sigmoid to each pass output to get probabilities
4. Stack outputs: shape `(n_passes, batch_size, 14)`
5. Return:
   - `mean_probs`: mean across passes → final prediction `(batch_size, 14)`
   - `uncertainty`: variance across passes → epistemic uncertainty `(batch_size, 14)`

**`compute_predictive_entropy(mean_probs)`** — Alternative uncertainty measure.

- Computes `-sum(p * log(p + eps))` for each sample
- Returns shape `(batch_size,)` — scalar entropy per sample
- Higher entropy = more uncertain prediction

**`classify_uncertainty(variance, threshold)`** — Categorizes uncertainty level.

- LOW: mean variance < threshold/2
- MEDIUM: mean variance < threshold
- HIGH: mean variance >= threshold

### 7.7 `uncertainty/metrics.py` — Evaluation Metrics

**`compute_ece(probs, labels, n_bins)`** — Expected Calibration Error (implemented from scratch).

- Flattens predictions across all classes and samples
- Divides [0, 1] into `n_bins` equal-width bins
- For each bin: computes `|mean_confidence - mean_accuracy|` weighted by sample fraction
- Returns sum of weighted differences (lower is better, 0 = perfect calibration)

**`compute_nll(probs, labels)`** — Negative Log-Likelihood.

- `NLL = -mean(y * log(p) + (1-y) * log(1-p))`
- Lower is better; measures how well predicted probabilities match true labels

**`compute_auc_per_class(probs, labels, class_names)`** — Per-class AUC using sklearn.

- Returns dict mapping each disease name to its ROC-AUC score
- Handles classes with only one label present (returns None, logs warning)

**`compute_all_metrics(probs, labels, class_names, n_bins)`** — Combines all metrics.

Returns dict with: `auc_per_class`, `mean_auc`, `ece`, `nll`

### 7.8 `training/losses.py` — Loss Functions

**`get_weighted_bce_loss(pos_weights, device)`** — Weighted Binary Cross-Entropy.

- Uses `BCEWithLogitsLoss` with `pos_weight` to handle class imbalance
- `pos_weights` computed from dataset: `N_negative / N_positive` per class
- Moves weights to GPU device automatically

**`get_standard_bce_loss()`** — Unweighted BCE for baseline model.

### 7.9 `training/trainer.py` — Training Loop

**`Trainer`** — Complete training orchestrator.

**Initialization:**
- Adam optimizer with configurable learning rate and weight decay
- ReduceLROnPlateau scheduler (tracks validation AUC, reduces LR when plateauing)
- Mixed precision training via `torch.cuda.amp.GradScaler` and `torch.autocast`
- TensorBoard `SummaryWriter` for logging
- Early stopping based on validation AUC

**`train_one_epoch(epoch)`** — Single training epoch:
1. Forward pass with optional mixed precision (`autocast`)
2. Compute weighted BCE loss
3. Backward pass with gradient scaling
4. Optimizer step
5. Track training loss and AUC
6. Clear CUDA cache after each batch (memory management for 6GB VRAM)

**`validate(epoch)`** — Validation epoch:
1. Standard inference (no MC Dropout) for speed
2. Collect all predictions and labels
3. Compute validation loss, AUC, and ECE
4. Log to TensorBoard

**`train(freeze_epochs)`** — Full training loop:
1. Freeze backbone for first `freeze_epochs` epochs (train only classifier head)
2. Unfreeze backbone for remaining epochs (fine-tune entire network)
3. Early stopping: halt if validation AUC doesn't improve for `patience` epochs
4. Save best checkpoint when validation AUC improves
5. Return dict of all epoch metrics

**`save_checkpoint(epoch, val_auc)`** / **`load_checkpoint(path)`** — Model persistence.

### 7.10 `evaluation/evaluator.py` — Test-Set Evaluation

**`Evaluator`** — Runs MC Dropout evaluation on the test set.

**`run(n_passes)`** — Full evaluation pipeline:
1. For each batch: run `mc_predict` with T=n_passes
2. Collect mean probabilities, uncertainty variance, labels, filenames
3. Compute all metrics (AUC per class, mean AUC, ECE, NLL)
4. Compute per-sample mean uncertainty and predictive entropy
5. Return comprehensive results dict

**`save_results(results, output_path)`** — Saves metrics to JSON and arrays to NPY files.

### 7.11 `evaluation/ood_evaluator.py` — OOD Analysis

**`OODEvaluator`** — Measures uncertainty on out-of-distribution data.

**`run(n_passes)`** — OOD evaluation:
1. Run MC Dropout on both OOD (Covid-X) and in-distribution (NIH test subset) data
2. Collect mean uncertainty and entropy for both sets
3. Compute separation ratio: `OOD_mean_uncertainty / InDist_mean_uncertainty`
4. A ratio > 1.5 indicates the model correctly assigns higher uncertainty to unfamiliar data

### 7.12 `visualization/plots.py` — Visualization Functions

All functions save figures to disk AND return matplotlib Figure objects.

**`plot_reliability_diagram`** — Calibration visualization.
- X-axis: predicted confidence, Y-axis: actual accuracy
- Perfect calibration = diagonal line (y=x)
- Shows ECE value in title

**`plot_uncertainty_distribution`** — OOD vs in-distribution uncertainty.
- Overlapping histograms (blue=in-distribution, red=OOD)
- Vertical lines at means, annotated with separation ratio

**`plot_auc_per_class`** — Model comparison bar chart.
- Horizontal bars comparing baseline vs MC Dropout AUC per disease
- Green = MC better, Orange = baseline better

**`plot_uncertainty_vs_accuracy`** — Uncertainty-accuracy correlation.
- Scatter plot: X=uncertainty, Y=per-sample accuracy
- Expected: negative correlation (high uncertainty → low accuracy)
- Includes linear regression trend line

**`plot_training_curves`** — Training history from TensorBoard logs.
- Train/val AUC and loss over epochs
- One figure per model

### 7.13 Scripts

#### `scripts/preprocess.py` — Image Preprocessing

Applies CLAHE enhancement to all NIH ChestX-ray14 images.

```bash
# Test on 100 images first
python scripts/preprocess.py --config config/default.yaml --dry-run

# Process all images
python scripts/preprocess.py --config config/default.yaml
```

- Reads images from `data/raw/images/`
- Saves CLAHE-enhanced images to `data/processed/images/`
- Shows tqdm progress bar
- Reports total processed, failed, and output directory

#### `scripts/train.py` — Model Training

Main training entry point.

```bash
# Train MC Dropout DenseNet-121
python scripts/train.py --model densenet --experiment-name mc_dropout

# Train ResNet-50 baseline
python scripts/train.py --model resnet --experiment-name baseline

# Resume from checkpoint
python scripts/train.py --model densenet --experiment-name mc_dropout --resume experiments/checkpoints/mc_dropout/best_model.pth
```

**CLI Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `config/default.yaml` | Path to configuration file |
| `--model` | `densenet` | Model architecture: `densenet` or `resnet` |
| `--experiment-name` | (required) | Name for checkpoint/log subdirectory |
| `--resume` | None | Path to checkpoint to resume training |

**What it does:**
1. Loads configuration from YAML
2. Creates train/val/test DataLoaders
3. Instantiates model and moves to GPU
4. Computes class-balanced pos_weights from training data
5. Trains with early stopping and LR scheduling
6. Saves best checkpoint based on validation AUC
7. Prints final validation metrics

#### `scripts/evaluate.py` — Model Evaluation

Runs test-set evaluation with MC Dropout inference.

```bash
python scripts/evaluate.py \
    --checkpoint experiments/checkpoints/mc_dropout/best_model.pth \
    --model densenet \
    --mc-passes 50 \
    --output-dir experiments/results/mc_dropout
```

**CLI Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `config/default.yaml` | Path to configuration file |
| `--checkpoint` | (required) | Path to trained model `.pth` file |
| `--model` | (required) | `densenet` or `resnet` |
| `--mc-passes` | From config | Number of MC Dropout passes (use 1 for baseline) |
| `--output-dir` | From config | Directory to save results and plots |

**Outputs:**
- Formatted metrics table (AUC per class, mean AUC, ECE, NLL)
- Reliability diagram PNG
- AUC per class comparison bar chart
- Uncertainty vs accuracy scatter plot
- Results JSON and numpy arrays

#### `scripts/evaluate_ood.py` — OOD Uncertainty Analysis

Tests whether the model produces higher uncertainty on unfamiliar data.

```bash
python scripts/evaluate_ood.py \
    --checkpoint experiments/checkpoints/mc_dropout/best_model.pth \
    --mc-passes 50 \
    --output-dir experiments/results/ood
```

**Outputs:**
- OOD mean uncertainty vs in-distribution mean uncertainty
- Separation ratio (should be > 1.5)
- Uncertainty distribution histogram PNG

### 7.14 Tests

The test suite contains **28 unit tests** across three files. All tests use synthetic data — no real dataset required.

#### `tests/test_dataset.py` (9 tests)
- `test_clahe_output_is_pil` — CLAHE transform returns PIL Image
- `test_clahe_output_is_rgb` — Output has 3 channels
- `test_clahe_preserves_dimensions` — Spatial dimensions unchanged
- `test_train_transforms_output_shape` — Training transforms produce (3, 224, 224)
- `test_val_transforms_output_shape` — Validation transforms produce (3, 224, 224)
- `test_dataset_len` — Dataset length matches CSV rows
- `test_getitem_shapes` — Returns correct tensor shapes and dtypes
- `test_missing_image_handled` — Missing images don't crash; returns zeros
- `test_pos_weights_shape` — pos_weights has shape (14,) with positive values

#### `tests/test_model.py` (8 tests)
- `test_densenet_forward_shape` — Output shape is (batch, 14)
- `test_resnet_forward_shape` — Output shape is (batch, 14)
- `test_mc_dropout_stochastic` — MC mode produces different outputs each pass
- `test_standard_eval_deterministic` — Eval mode produces identical outputs
- `test_freeze_backbone` — Frozen parameters have requires_grad=False
- `test_unfreeze_backbone` — All parameters have requires_grad=True

#### `tests/test_uncertainty.py` (11 tests)
- `test_mc_predict_output_shapes` — Correct output shapes
- `test_mc_predict_probs_in_range` — Probabilities in [0, 1]
- `test_uncertainty_variance_nonneg` — Variance is non-negative
- `test_predictive_entropy_shape` — Correct entropy shape
- `test_classify_uncertainty_levels` — LOW/MEDIUM/HIGH classification
- `test_ece_perfect_calibration` — ECE < 0.01 for perfect calibration
- `test_ece_range` — ECE in [0, 1]
- `test_nll_finite` — NLL is finite
- `test_nll_perfect_prediction` — NLL ≈ 0 for perfect predictions
- `test_auc_per_class_returns_all_classes` — All 14 class names returned
- `test_compute_all_metrics_keys` — Correct dict keys

---

## 8. Configuration Reference

Complete reference for `config/default.yaml`:

### Data Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data.nih_image_dir` | `data/raw/images` | Path to NIH ChestX-ray14 images |
| `data.nih_train_csv` | `data/splits/train_val_list.txt` | Training/validation split file |
| `data.nih_test_csv` | `data/splits/test_list.txt` | Test split file |
| `data.covidx_image_dir` | `data/raw/covidx` | Path to Covid-X images |
| `data.covidx_csv` | `data/splits/covidx_test.csv` | Covid-X file list |
| `data.image_size` | `224` | Input image resolution (pixels) |
| `data.num_workers` | `4` | DataLoader worker threads (set to 0 on Windows if multiprocessing issues) |
| `data.clahe_clip_limit` | `2.0` | CLAHE contrast limiting threshold |
| `data.clahe_tile_size` | `8` | CLAHE grid size (8×8 tiles) |

### Model Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.backbone` | `densenet121` | Architecture: `densenet121` or `resnet50` |
| `model.num_classes` | `14` | Number of disease classes |
| `model.dropout_rate` | `0.3` | Dropout probability for MC Dropout |
| `model.pretrained` | `true` | Use ImageNet pretrained weights |
| `model.freeze_epochs` | `5` | Epochs to freeze backbone before fine-tuning |

### Training Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training.epochs` | `50` | Maximum training epochs |
| `training.batch_size` | `32` | Batch size (reduce to 16 if OOM on 6GB VRAM) |
| `training.learning_rate` | `0.0001` | Adam optimizer learning rate |
| `training.weight_decay` | `0.0001` | L2 regularization strength |
| `training.early_stopping_patience` | `10` | Stop if val AUC doesn't improve for N epochs |
| `training.lr_scheduler_patience` | `5` | Reduce LR if val AUC plateaus for N epochs |
| `training.lr_scheduler_factor` | `0.5` | LR reduction factor (multiply by this) |
| `training.checkpoint_dir` | `experiments/checkpoints` | Where to save model weights |
| `training.log_dir` | `experiments/logs` | TensorBoard log directory |
| `training.mixed_precision` | `true` | Use FP16 mixed precision (faster, less VRAM) |

### Uncertainty Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `uncertainty.mc_passes` | `50` | Number of MC Dropout forward passes (T) |
| `uncertainty.mc_passes_fast` | `20` | Reduced passes for quick experiments |

### Evaluation Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `evaluation.output_dir` | `experiments/results` | Where to save evaluation outputs |
| `evaluation.calibration_bins` | `15` | Number of bins for ECE computation |

---

## 9. Usage Guide

### Complete Workflow

#### Step 1: Preprocess Images

Apply CLAHE enhancement to improve contrast in chest X-rays:

```bash
# Test on 100 images first (recommended)
python scripts/preprocess.py --config config/default.yaml --dry-run

# Process all images
python scripts/preprocess.py --config config/default.yaml
```

This creates `data/processed/images/` with enhanced images. Training will use these if available.

#### Step 2: Train Baseline Model

Train the ResNet-50 baseline (no uncertainty quantification):

```bash
python scripts/train.py --model resnet --experiment-name baseline
```

Monitor training:
```bash
tensorboard --logdir experiments/logs
```

#### Step 3: Train MC Dropout Model

Train the DenseNet-121 with MC Dropout:

```bash
python scripts/train.py --model densenet --experiment-name mc_dropout
```

#### Step 4: Evaluate Models

Evaluate the MC Dropout model on the test set:

```bash
python scripts/evaluate.py \
    --checkpoint experiments/checkpoints/mc_dropout/best_model.pth \
    --model densenet \
    --mc-passes 50 \
    --output-dir experiments/results/mc_dropout
```

Evaluate the baseline:

```bash
python scripts/evaluate.py \
    --checkpoint experiments/checkpoints/baseline/best_model.pth \
    --model resnet \
    --mc-passes 1 \
    --output-dir experiments/results/baseline
```

#### Step 5: OOD Uncertainty Analysis

Test whether MC Dropout produces higher uncertainty on unfamiliar data:

```bash
python scripts/evaluate_ood.py \
    --checkpoint experiments/checkpoints/mc_dropout/best_model.pth \
    --mc-passes 50 \
    --output-dir experiments/results/ood
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_model.py -v

# Run specific test
python -m pytest tests/test_model.py::TestDenseNet::test_mc_dropout_stochastic -v
```

### Smoke Test

A quick end-to-end verification using synthetic data (no real dataset needed):

```bash
python scripts/smoke_test.py
```

This tests:
1. MCDropoutDenseNet forward pass
2. MC Dropout inference (T=10 passes)
3. MC Dropout stochasticity (outputs differ across passes)
4. Metrics computation (ECE, NLL)
5. ResNetBaseline forward pass

---

## 10. Testing

### Test Coverage

| Module | Test File | Tests | Coverage |
|--------|-----------|-------|----------|
| Data | `test_dataset.py` | 9 | CLAHE, transforms, dataset, pos_weights |
| Models | `test_model.py` | 8 | Forward pass, MC stochasticity, freeze/unfreeze |
| Uncertainty | `test_uncertainty.py` | 11 | MC predict, metrics, ECE, NLL, AUC |
| **Total** | | **28** | |

### Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Expected output: 28 passed
```

### Test Design Philosophy

- **No real data required** — all tests use synthetic tensors and temporary CSV files
- **Fast execution** — full suite completes in ~60 seconds on GPU
- **Isolated** — each test is independent, no shared state
- **Comprehensive** — tests cover shapes, ranges, stochasticity, determinism, and edge cases

---

## 11. Expected Results

### Model Performance Targets

| Metric | Baseline (ResNet-50) | MC Dropout (DenseNet-121) | Better |
|--------|---------------------|---------------------------|--------|
| **Mean AUC** | ~0.80 | ~0.82 | MC Dropout |
| **ECE** | ~0.08 | ~0.05 | MC Dropout (lower) |
| **NLL** | ~0.35 | ~0.30 | MC Dropout (lower) |
| **OOD Separation** | N/A | >1.5x | MC Dropout |

### Per-Class AUC Expectations

| Disease | Baseline AUC | MC Dropout AUC |
|---------|-------------|----------------|
| Cardiomegaly | ~0.88 | ~0.89 |
| Effusion | ~0.86 | ~0.87 |
| Hernia | ~0.85 | ~0.87 |
| Infiltration | ~0.72 | ~0.74 |
| Mass | ~0.80 | ~0.82 |
| Nodule | ~0.74 | ~0.76 |
| Atelectasis | ~0.78 | ~0.80 |
| Pneumothorax | ~0.84 | ~0.86 |
| Consolidation | ~0.79 | ~0.81 |
| Edema | ~0.87 | ~0.88 |
| Emphysema | ~0.89 | ~0.90 |
| Fibrosis | ~0.80 | ~0.82 |
| Pleural_Thickening | ~0.76 | ~0.78 |
| Pneumonia | ~0.74 | ~0.76 |

*Note: These are target values based on literature. Actual results depend on data splits, preprocessing, and training duration.*

### Training Time Estimates

| GPU | Model | Epochs | Time per Epoch | Total |
|-----|-------|--------|----------------|-------|
| RTX 4050 (6GB) | DenseNet-121 | 50 | ~15 min | ~12.5 hours |
| RTX 4050 (6GB) | ResNet-50 | 50 | ~12 min | ~10 hours |
| RTX 3080 (10GB) | DenseNet-121 | 50 | ~8 min | ~6.5 hours |

### VRAM Usage

| Batch Size | Model | VRAM (approx) |
|-----------|-------|----------------|
| 32 | DenseNet-121 | ~5.5 GB |
| 16 | DenseNet-121 | ~3.2 GB |
| 32 | ResNet-50 | ~4.8 GB |
| 16 | ResNet-50 | ~2.8 GB |

> **Note:** If you encounter CUDA out-of-memory errors on a 6GB GPU, reduce `training.batch_size` from 32 to 16 in `config/default.yaml`.

---

## 12. Troubleshooting

### CUDA Issues

**Problem:** `CUDA available: False` after installing PyTorch

**Solutions:**
1. Check NVIDIA driver version: `nvidia-smi` — must be >= 525.x for CUDA 12.1
2. Reinstall PyTorch with correct CUDA version:
   ```bash
   pip uninstall torch torchvision torchaudio -y
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
3. Try cu124 build instead:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

**Problem:** `CUDA out of memory` during training

**Solutions:**
1. Reduce batch size: set `training.batch_size: 16` in `config/default.yaml`
2. Enable mixed precision: set `training.mixed_precision: true`
3. Reduce image size: set `data.image_size: 192` (less VRAM, slight accuracy loss)

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'models'`

**Solution:** Run scripts from the project root directory, or set PYTHONPATH:
```bash
# Windows
$env:PYTHONPATH="E:\Thesis\Project\chest_xray_uncertainty"
# Linux/macOS
export PYTHONPATH="/path/to/chest_xray_uncertainty"
```

**Problem:** `ModuleNotFoundError: No module named 'lightning'`

**Solution:** Install PyTorch Lightning:
```bash
pip install lightning
```

### DataLoader Issues on Windows

**Problem:** Multiprocessing errors or hanging with `num_workers > 0`

**Solution:** Set `data.num_workers: 0` in `config/default.yaml`. This is a known Windows limitation when scripts are not guarded by `if __name__ == '__main__':`.

### Slow Training

**Problem:** Training is much slower than expected

**Checklist:**
1. Verify CUDA is being used: check `Device: cuda` in training output
2. Enable mixed precision: `training.mixed_precision: true`
3. Use SSD storage for dataset (HDD causes I/O bottleneck)
4. Increase `data.num_workers` to 4 or 8 (Linux only)

---

## 13. References

1. **Gal, Y., & Ghahramani, Z. (2016).** "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning." *Proceedings of the 33rd International Conference on Machine Learning (ICML 2016).*
   - Foundational paper establishing MC Dropout as a Bayesian approximation

2. **Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M. (2017).** "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases." *IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2017).*
   - Original NIH ChestX-ray14 dataset paper

3. **Mucsanyi, B., et al. (2025).** "Torch-Uncertainty: A Comprehensive Library for Uncertainty Quantification in Deep Learning." *Journal of Open Source Software.*
   - PyTorch library for uncertainty quantification used in this project

4. **Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017).** "Densely Connected Convolutional Networks." *CVPR 2017.*
   - DenseNet-121 architecture used as the MC Dropout backbone

5. **He, K., Zhang, X., Ren, S., & Sun, J. (2016).** "Deep Residual Learning for Image Recognition." *CVPR 2016.*
   - ResNet-50 architecture used as the baseline model

6. **Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).** "On Calibration of Modern Neural Networks." *ICML 2017.*
   - Calibration metrics and Expected Calibration Error formulation

7. **Wang, L., Lin, Z. Q., & Wong, A. (2020).** "COVID-Net: A Tailored Deep Convolutional Neural Network Design for Detection of COVID-19 Cases from Chest X-Ray Images." *Scientific Reports.*
   - Covid-X dataset used for OOD evaluation

---

## License

This project is developed for academic thesis purposes.

## Contact

**Anzam Shahriar Kabir** — Student ID: 2022521460123

# Progress Tracking for Methodology

This folder contains progress notes and outputs that will be used for writing the methodology section.

## Current Progress

- [2026-04-04]: Initial setup of progress tracking folder.
- [2026-04-04]: Ran unit tests for models (test_model.py). All 8 tests passed successfully.
- [2026-04-04]: Modified trainer.py to print training and validation metrics (loss, AUC, ECE) to stdout for real-time tracking during training.

## Outputs and Results

### Model Unit Tests Results (2026-04-04)
```
test_densenet_forward_shape (tests.test_model.TestDenseNet.test_densenet_forward_shape)
Forward pass should produce correct output shape. ... ok
test_freeze_backbone (tests.test_model.TestDenseNet.test_freeze_backbone)
freeze_backbone should set all feature parameters to requires_grad=False. ... ok
test_mc_dropout_stochastic (tests.test_model.TestDenseNet.test_mc_dropout_stochastic)
MC Dropout mode should produce different outputs for same input. ... ok
test_standard_eval_deterministic (tests.test_model.TestDenseNet.test_standard_eval_deterministic)
Standard eval mode should produce identical outputs for same input. ... ok
test_unfreeze_backbone (tests.test_model.TestDenseNet.test_unfreeze_backbone)
unfreeze_backbone should set all parameters to requires_grad=True. ... ok
test_freeze_backbone (tests.test_model.TestResNet.test_freeze_backbone)
freeze_backbone should freeze all non-fc parameters. ... ok
test_resnet_forward_shape (tests.test_model.TestResNet.test_resnet_forward_shape)
Forward pass should produce correct output shape. ... ok
test_unfreeze_backbone (tests.test_model.TestResNet.test_unfreeze_backbone)
unfreeze_backbone should unfreeze all parameters. ... ok

----------------------------------------------------------------------
Ran 8 tests in 6.251s

OK
```

### Configuration Changes (2026-04-05)
- Enforced CUDA requirement in all scripts (train.py, evaluate.py, evaluate_ood.py, smoke_test.py). Scripts now exit with a clear error if CUDA is unavailable instead of silently falling back to CPU.
- Confirmed terminal output already in place: trainer.py prints per-epoch Loss, AUC, ECE to stdout + tqdm batch progress bars.

---

### GPU Bottleneck Diagnosis & Fix (2026-04-17)
GPU utilization was 20–30% with only 1 GB/6 GB VRAM in use. Root causes identified and fixed:
- **Primary cause**: `num_workers` was hard-overridden to 0 on Windows in `data/dataset.py`, starving the GPU of data every batch.
- **Secondary cause**: `torch.cuda.empty_cache()` was called inside the training loop after every batch, forcing a GPU–CPU sync on each step.
- **Tertiary cause**: `batch_size: 10` was too small to saturate CUDA cores.

Fixes applied:
- Removed the Windows `num_workers=0` override — DataLoader now uses `num_workers` from config.
- Added `persistent_workers=True` and `prefetch_factor=2` to all three DataLoaders (train/val/test).
- Removed `torch.cuda.empty_cache()` from the per-batch training loop.
- Increased `batch_size` from 10 → 32 (later tuned further).

---

### cv2.CLAHE Pickle Error Fix (2026-04-17)
Training crashed with `TypeError: cannot pickle 'cv2.CLAHE'` when `num_workers > 0`.
- **Root cause**: `cv2.createCLAHE()` was stored on `self` in `ClaheTransform.__init__`. Windows DataLoader workers use `spawn` (not `fork`), so they pickle the entire dataset+transform object — and `cv2.CLAHE` is not picklable.
- **Fix**: Moved `cv2.createCLAHE()` inside `__call__`. The transform now creates a fresh CLAHE object per call, which is never stored and never pickled.
- Set `num_workers=0` as a safe fallback while testing; later raised to 2 after confirming fix.
- Added `torch.backends.cudnn.benchmark = True` to `scripts/train.py` for auto-tuned convolution kernels.

---

### Curriculum Training Strategy (2026-04-17)
Adopted a three-stage curriculum approach to reach full NIH dataset training:

**Stage 1 — 10k Run** (`thesis_10k_stable_run`)
- Dataset: 10,000 training images (slice of `train_val_list.txt`)
- Batch size: 32 | LR: 1e-4 | Scheduler: StepLR (×0.1 every 7 epochs)
- **Result: Best Val AUC = 0.7660**

**Stage 2 — 30k Run** (`thesis_30k_power_run`)
- Dataset: 30,000 training images | Batch: 48 | num_workers: 2
- Transfer learning from `thesis_10k_stable_run` weights (optimizer reset, epoch 0)
- `early_stopping_patience: 15` to survive StepLR drops
- **Result: Best Val AUC = 0.8114**

**Stage 3 — 112k Ultimate Run** (`thesis_112k_ultimate_run`)
- Dataset: Full `train_val_list.txt` (~86k train / ~21k val after 80/20 split)
- Transfer learning from `thesis_30k_power_run` weights
- LR reduced to 5e-5 (polishing phase) | `early_stopping_patience: 10`
- `test_list.txt` fully reserved — never used during training
- **Result: Best Val AUC = 0.8272**

All three stages used: MCDropout-DenseNet121, AMP (float16), `pin_memory=True`, `persistent_workers=True`, `cudnn.benchmark=True`.

---

### Evaluation Infrastructure (2026-04-17)
Fixed three bugs in existing evaluation scripts before they were run:
- `evaluate.py`: was passing `test_list.txt` as `labels_csv` instead of as `split_file` — would have crashed on `pd.read_csv()`.
- `evaluate_ood.py`: same `labels_csv` bug for the in-distribution dataset.
- `evaluate_ood.py`: `CovidXDataset` called with wrong kwarg `csv_path=` instead of `split_file=`.
- Added **Macro F1-Score** (threshold=0.5) to `compute_all_metrics()` in `uncertainty/metrics.py`.

Covid-X OOD paths also corrected in `config/default.yaml`:
- `covidx_image_dir`: `data/raw/covidx` → `data/Covid-X/test`
- `covidx_csv`: `data/splits/covidx_test.csv` → `data/Covid-X/test.txt`

---

### Final Reliability Report Pipeline (2026-04-17)
Created `scripts/reliability_report.py` — NIH-only comprehensive evaluation using T=50 MC Dropout passes on the held-out test set. Generates:
- **Metrics** (terminal + JSON): Mean AUC, Macro F1, ECE, NLL, Mean Entropy
- **Plot 1** — `reliability_diagram.png`: confidence vs accuracy calibration curve (300 dpi)
- **Plot 2** — `uncertainty_heatmap.png`: per-disease mean MC variance sorted descending
- **Plot 3** — `entropy_histogram.png`: predictive entropy distribution with KDE, mean/σ/P95 lines
- **Plot 4** — `uncertainty_accuracy_deciles.png`: samples grouped into 10 uncertainty deciles; accuracy decreases as uncertainty increases — empirically validates the Bayesian thesis claim

New plot functions added to `visualization/plots.py`:
- `plot_uncertainty_deciles()`, `plot_uncertainty_heatmap()`, `plot_entropy_histogram()`
- Added `dpi` parameter to `plot_reliability_diagram()` (default 150, thesis outputs use 300)

---

### Single-Image Inference Demo Script (2026-04-17)
Created `scripts/single_inference.py` for thesis defense demonstration.
- Input: any NIH image path (auto-resolves filename recursively across all 12 subdirectories)
- Runs T=50 MC Dropout passes on one image
- Terminal output: Top-3 predicted diseases with mean probability + variance, entropy score with LOW/MEDIUM/HIGH classification
- Saves 300 dpi figure: original X-ray (left) + all 14 disease probabilities with ±1 std error bars (right), Top-3 highlighted in red

---

### Training Graphs Dashboard (2026-04-17)
Created `graphs/generate_graphs.py` — reads all TensorBoard event logs and generates:
- Per-experiment training curves (Loss, AUC, ECE, LR) for all 8 runs
- Cross-experiment comparison bar charts (best val AUC + ECE)
- AUC–ECE scatter (trade-off across experiments)
- Summary table ranked by best val AUC
- `graphs/output/summary.csv` — machine-readable summary of all experiments

Run anytime after training: `python graphs/generate_graphs.py`

---

## Notes

- All model tests are passing, indicating the MCDropoutDenseNet and ResNetBaseline models are functioning correctly.
- Tests cover forward pass shapes, MC Dropout stochasticity, deterministic eval mode, and backbone freezing/unfreezing.
- Training script now prints epoch metrics to terminal for easy tracking: Loss, AUC (accuracy measure), and ECE (calibration metric).
- All scripts enforce CUDA — no silent CPU fallback. Project designed for RTX 4050 Laptop GPU (6GB VRAM).
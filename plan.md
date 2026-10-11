# Uncertainty-Quantified Deep Learning Framework for Reliable Chest X-Ray Diagnosis
## Project Build Plan — Anzam Shahriar Kabir (2022521460123)

> **Instructions for AI Agent (Opus 4.6)**
> Build this project top-to-bottom, file by file, in the order defined below.
> Each section specifies exact file paths, responsibilities, interfaces, and constraints.
> Do not skip sections. Do not combine files unless explicitly told to.
> After every file, confirm it is complete before moving to the next.

---

## Tech Stack

- **Language**: Python 3.10
- **Deep Learning**: PyTorch 2.x + Torchvision
- **Uncertainty**: Torch-Uncertainty
- **Data / Metrics**: NumPy, Pandas, Scikit-Learn, OpenCV (cv2)
- **Visualization**: Matplotlib, Seaborn
- **Experiment Tracking**: TensorBoard (torch.utils.tensorboard)
- **Config**: PyYAML
- **Utilities**: tqdm, Pillow

---

## Project Directory Layout

Build the following structure exactly. Create every `__init__.py` as an empty file.

```
chest_xray_uncertainty/
├── config/
│   └── default.yaml
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   ├── transforms.py
│   └── splits/              # empty dir — user fills with NIH CSV splits
├── models/
│   ├── __init__.py
│   ├── densenet.py
│   └── resnet_baseline.py
├── uncertainty/
│   ├── __init__.py
│   ├── mc_dropout.py
│   └── metrics.py
├── training/
│   ├── __init__.py
│   ├── trainer.py
│   └── losses.py
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py
│   └── ood_evaluator.py
├── visualization/
│   ├── __init__.py
│   └── plots.py
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── evaluate_ood.py
├── tests/
│   ├── test_dataset.py
│   ├── test_model.py
│   └── test_uncertainty.py
├── requirements.txt
└── README.md
```

---

## File 1 — `requirements.txt`

List all dependencies with pinned major versions:

```
torch>=2.1.0
torchvision>=0.16.0
torch-uncertainty>=0.3.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
opencv-python>=4.8.0
Pillow>=10.0.0
matplotlib>=3.7.0
seaborn>=0.13.0
tqdm>=4.65.0
PyYAML>=6.0
tensorboard>=2.14.0
```

---

## File 2 — `config/default.yaml`

Single source of truth for all hyperparameters and paths. Every module reads from this file — no hardcoded values anywhere else.

```yaml
data:
  nih_image_dir: "data/raw/images"
  nih_train_csv: "data/splits/train_val_list.txt"
  nih_test_csv: "data/splits/test_list.txt"
  covidx_image_dir: "data/raw/covidx"
  covidx_csv: "data/splits/covidx_test.csv"
  image_size: 224
  num_workers: 4
  clahe_clip_limit: 2.0
  clahe_tile_size: 8

model:
  backbone: "densenet121"          # or "resnet50"
  num_classes: 14
  dropout_rate: 0.3
  pretrained: true
  freeze_epochs: 5                 # epochs to freeze backbone, then unfreeze

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.0001
  weight_decay: 0.0001
  early_stopping_patience: 10
  lr_scheduler_patience: 5
  lr_scheduler_factor: 0.5
  checkpoint_dir: "experiments/checkpoints"
  log_dir: "experiments/logs"
  mixed_precision: true

uncertainty:
  mc_passes: 50                    # T in MC Dropout literature
  mc_passes_fast: 20               # for quick experiments

evaluation:
  output_dir: "experiments/results"
  calibration_bins: 15

disease_classes:
  - Atelectasis
  - Consolidation
  - Infiltration
  - Pneumothorax
  - Edema
  - Emphysema
  - Fibrosis
  - Effusion
  - Pneumonia
  - Pleural_Thickening
  - Cardiomegaly
  - Nodule
  - Mass
  - Hernia
```

---

## File 3 — `data/transforms.py`

Implement two transform pipelines — one for training (with augmentation) and one for validation/test (no augmentation). Both must apply CLAHE as the first step before any tensor conversion.

**`get_train_transforms(image_size: int) -> transforms.Compose`**
- Apply CLAHE (see ClaheTransform class below)
- `RandomHorizontalFlip(p=0.5)`
- `RandomRotation(degrees=10)`
- `Resize((image_size, image_size))`
- `ToTensor()`
- `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

**`get_val_transforms(image_size: int) -> transforms.Compose`**
- Apply CLAHE
- `Resize((image_size, image_size))`
- `ToTensor()`
- `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

**`class ClaheTransform`**
- Accepts a PIL Image
- Converts to grayscale numpy array
- Applies `cv2.createCLAHE(clipLimit, tileGridSize)` 
- Converts back to RGB PIL Image (stack grayscale into 3 channels)
- Constructor args: `clip_limit=2.0`, `tile_size=8`

---

## File 4 — `data/dataset.py`

**`class ChestXrayDataset(torch.utils.data.Dataset)`**

Constructor args:
- `image_dir: str` — path to folder of images
- `labels_csv: str` — path to CSV with columns: `Image Index`, plus one column per disease class (0/1 binary)
- `transform: Optional[Callable]` — transform pipeline
- `disease_classes: List[str]` — ordered list of 14 class names from config

Requirements:
- `__len__` returns number of samples
- `__getitem__(idx)` returns `(image_tensor, label_tensor, image_filename)`
  - `label_tensor` is a `torch.FloatTensor` of shape `(14,)` with binary values
  - Load image using PIL, apply transform, return tensor
- Include a `get_pos_weights()` method that returns a `torch.FloatTensor` of shape `(14,)` where `pos_weight[i] = (N_neg_i / N_pos_i)`. Used for weighted BCE loss.
- Handle missing images gracefully: log a warning and return a zero image + zero label (do not crash)

**`class CovidXDataset(torch.utils.data.Dataset)`**

Constructor args:
- `image_dir: str`
- `csv_path: str` — CSV with column `filename` (no labels needed for OOD testing)
- `transform: Optional[Callable]`

Requirements:
- `__getitem__` returns `(image_tensor, image_filename)`
- No label tensor needed — this dataset is used only for OOD uncertainty measurement

**`def create_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]`**

Returns `(train_loader, val_loader, test_loader)` using config values for batch_size, num_workers. Apply train transforms to train loader, val transforms to val/test loaders.

---

## File 5 — `models/densenet.py`

**`class MCDropoutDenseNet(nn.Module)`**

This is the core model. Builds on `torchvision.models.densenet121(pretrained=True)`.

Constructor args:
- `num_classes: int = 14`
- `dropout_rate: float = 0.3`
- `pretrained: bool = True`

Architecture modifications:
1. Load `densenet121(pretrained=pretrained)`
2. Replace `model.classifier` with:
   ```
   nn.Sequential(
       nn.Dropout(p=dropout_rate),
       nn.Linear(1024, 512),
       nn.ReLU(),
       nn.Dropout(p=dropout_rate),
       nn.Linear(512, num_classes)
   )
   ```
3. Do NOT modify the DenseNet feature extractor (backbone) layers. Only replace the classifier head.

**`def freeze_backbone(self)`** — freeze all parameters in `self.model.features`

**`def unfreeze_backbone(self)`** — unfreeze all parameters in `self.model.features`

**`def enable_mc_dropout(self)`** — set ONLY `nn.Dropout` modules to `training=True` while everything else stays in eval mode. This is the critical MC Dropout inference method. Implement as:
```python
def enable_mc_dropout(self):
    for module in self.modules():
        if isinstance(module, nn.Dropout):
            module.train()
```

**`def forward(self, x: torch.Tensor) -> torch.Tensor`**
- Returns raw logits of shape `(batch_size, num_classes)` — no sigmoid, loss function handles it

---

## File 6 — `models/resnet_baseline.py`

**`class ResNetBaseline(nn.Module)`**

Standard fine-tuned ResNet-50 WITHOUT MC Dropout. Used as the comparison baseline.

Constructor args:
- `num_classes: int = 14`
- `pretrained: bool = True`

Architecture:
1. Load `resnet50(pretrained=pretrained)`
2. Replace `model.fc` with `nn.Linear(2048, num_classes)`
3. No dropout in the classifier head

**`def freeze_backbone(self)` / `def unfreeze_backbone(self)`** — same pattern as densenet.py, freeze/unfreeze all layers except `fc`

**`def forward(self, x) -> torch.Tensor`** — returns raw logits `(batch_size, num_classes)`

---

## File 7 — `uncertainty/mc_dropout.py`

**`def mc_predict(model: nn.Module, image_batch: torch.Tensor, n_passes: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]`**

Core MC Dropout inference function. This is the most important function in the project.

Algorithm:
1. Call `model.eval()` then `model.enable_mc_dropout()` (assumes model has this method)
2. Run `n_passes` forward passes on the same `image_batch`
3. Apply `torch.sigmoid()` to each pass output to get probabilities
4. Stack all pass outputs into shape `(n_passes, batch_size, num_classes)`
5. Return:
   - `mean_probs`: shape `(batch_size, num_classes)` — mean across passes
   - `uncertainty`: shape `(batch_size, num_classes)` — variance across passes

No gradient computation during this function (`torch.no_grad()`).

**`def compute_predictive_entropy(mean_probs: torch.Tensor) -> torch.Tensor`**
- Computes entropy of the mean prediction: `-sum(p * log(p + eps))`
- Returns shape `(batch_size,)` — scalar entropy per sample
- This is an alternative uncertainty measure to variance

**`def classify_uncertainty(variance: torch.Tensor, threshold: float = 0.05) -> List[str]`**
- For each sample, returns `"LOW"`, `"MEDIUM"`, or `"HIGH"` based on mean variance across classes
- Thresholds: < `threshold/2` → LOW, < `threshold` → MEDIUM, >= `threshold` → HIGH

---

## File 8 — `uncertainty/metrics.py`

**`def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float`**

Computes Expected Calibration Error manually (do not rely on library for this — implement from scratch so you understand it).

Algorithm:
1. Flatten probs and labels to 1D (treat each class independently)
2. Divide [0, 1] into `n_bins` equal-width bins
3. For each bin: find samples whose predicted probability falls in that bin
4. Compute `|avg_confidence - avg_accuracy|` for that bin, weighted by fraction of samples
5. Return sum of weighted absolute differences

**`def compute_nll(probs: np.ndarray, labels: np.ndarray) -> float`**

Negative Log-Likelihood for binary multi-label predictions.
- `NLL = -mean(y * log(p + eps) + (1 - y) * log(1 - p + eps))`

**`def compute_auc_per_class(probs: np.ndarray, labels: np.ndarray, class_names: List[str]) -> Dict[str, float]`**

- Uses `sklearn.metrics.roc_auc_score` per class
- Returns dict: `{class_name: auc_value}`
- Handle classes with only one label present in test set (return `None` for those, log warning)

**`def compute_all_metrics(probs: np.ndarray, labels: np.ndarray, class_names: List[str], n_bins: int = 15) -> Dict`**

Combines all metrics into one dict:
```python
{
    "auc_per_class": {...},
    "mean_auc": float,
    "ece": float,
    "nll": float
}
```

---

## File 9 — `training/losses.py`

**`def get_weighted_bce_loss(pos_weights: torch.Tensor, device: torch.device) -> nn.BCEWithLogitsLoss`**

Returns a `BCEWithLogitsLoss` with `pos_weight` set to handle class imbalance.
- `pos_weights` comes from `dataset.get_pos_weights()`
- Move `pos_weights` to `device` before passing to loss

**`def get_standard_bce_loss() -> nn.BCEWithLogitsLoss`**

Returns `BCEWithLogitsLoss` with no weighting. Used for the baseline model.

---

## File 10 — `training/trainer.py`

**`class Trainer`**

Handles the full training loop for both models (baseline and MC Dropout model).

Constructor args:
- `model: nn.Module`
- `train_loader: DataLoader`
- `val_loader: DataLoader`
- `config: dict`
- `device: torch.device`
- `pos_weights: Optional[torch.Tensor] = None`

**Internal setup in `__init__`:**
- `optimizer = Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=...)`
- `scheduler = ReduceLROnPlateau(optimizer, patience=..., factor=..., mode='max')` — tracks val AUC
- `loss_fn` from `losses.py` (weighted if pos_weights provided)
- `scaler = torch.cuda.amp.GradScaler()` if `config['training']['mixed_precision']` is True
- `writer = SummaryWriter(log_dir=...)` for TensorBoard

**`def train_one_epoch(epoch: int) -> Dict[str, float]`**
- Loop over train_loader
- Forward pass → compute loss → backward → optimizer step
- Use `torch.cuda.amp.autocast()` if mixed_precision enabled
- Track and return `{"train_loss": float, "train_auc": float}`

**`def validate(epoch: int) -> Dict[str, float]`**
- Loop over val_loader in `model.eval()` mode (standard, not MC)
- Collect all probs and labels
- Compute loss, mean AUC using `uncertainty/metrics.py`
- Log to TensorBoard
- Return `{"val_loss": float, "val_auc": float, "val_ece": float}`

**`def train(freeze_epochs: int) -> Dict`**
- Calls `model.freeze_backbone()` for first `freeze_epochs` epochs, then `model.unfreeze_backbone()`
- Runs `train_one_epoch` + `validate` for all epochs
- Implements early stopping: stop if val_auc does not improve for `patience` epochs
- Saves best checkpoint via `save_checkpoint()` when val_auc improves
- Returns dict of all epoch metrics

**`def save_checkpoint(epoch: int, val_auc: float)`**
- Saves `model.state_dict()`, `optimizer.state_dict()`, epoch, val_auc to `checkpoint_dir/best_model.pth`

**`def load_checkpoint(path: str)`**
- Loads checkpoint, restores model and optimizer state

---

## File 11 — `evaluation/evaluator.py`

**`class Evaluator`**

Runs full evaluation on the test set using MC Dropout inference.

Constructor args:
- `model: nn.Module`
- `test_loader: DataLoader`
- `config: dict`
- `device: torch.device`

**`def run(n_passes: int) -> Dict`**

For each batch in test_loader:
1. Call `mc_predict(model, batch, n_passes, device)` from `uncertainty/mc_dropout.py`
2. Collect `mean_probs`, `uncertainty_variance`, `labels`, `filenames`

After full dataset:
3. Compute `compute_all_metrics(all_probs, all_labels, class_names)`
4. Compute per-sample mean uncertainty (mean variance across 14 classes)
5. Compute predictive entropy per sample

Return dict:
```python
{
    "metrics": {mean_auc, ece, nll, auc_per_class},
    "predictions": np.ndarray,         # shape (N, 14)
    "uncertainty_variance": np.ndarray, # shape (N, 14)
    "mean_uncertainty": np.ndarray,     # shape (N,)
    "entropy": np.ndarray,              # shape (N,)
    "labels": np.ndarray,               # shape (N, 14)
    "filenames": List[str]
}
```

**`def save_results(results: Dict, output_path: str)`**
- Save results dict to JSON (metrics only — numpy arrays saved separately as .npy files)

---

## File 12 — `evaluation/ood_evaluator.py`

**`class OODEvaluator`**

Measures uncertainty on out-of-distribution Covid-X images. No labels involved.

Constructor args:
- `model: nn.Module`
- `ood_loader: DataLoader`      # CovidXDataset loader
- `indist_loader: DataLoader`   # subset of ChestXray14 test loader (same size as OOD set)
- `config: dict`
- `device: torch.device`

**`def run(n_passes: int) -> Dict`**

For BOTH loaders (OOD and in-distribution):
1. Run `mc_predict` on each batch
2. Collect `mean_uncertainty` (mean variance across classes) per sample
3. Collect `entropy` per sample

Return dict:
```python
{
    "ood": {
        "mean_uncertainty": np.ndarray,   # shape (N_ood,)
        "entropy": np.ndarray,
        "mean_uncertainty_scalar": float,  # mean of the above
    },
    "in_dist": {
        "mean_uncertainty": np.ndarray,
        "entropy": np.ndarray,
        "mean_uncertainty_scalar": float,
    },
    "separation_ratio": float  # ood_mean / indist_mean — should be > 1.5 to be meaningful
}
```

---

## File 13 — `visualization/plots.py`

All functions save figures to `output_dir` AND return the `matplotlib.figure.Figure` object.

**`def plot_reliability_diagram(probs, labels, model_name, output_dir, n_bins=15)`**
- X-axis: Mean predicted confidence per bin
- Y-axis: Actual accuracy per bin
- Plot both the model's curve and the perfect calibration diagonal (y=x)
- Add ECE value in the plot title

**`def plot_uncertainty_distribution(ood_uncertainty, indist_uncertainty, output_dir)`**
- Overlapping histograms: in-distribution (blue) vs OOD (red)
- X-axis: Mean uncertainty (variance), Y-axis: density
- Add vertical lines at means of each distribution
- Annotate with separation_ratio

**`def plot_auc_per_class(auc_dict_baseline, auc_dict_mc, class_names, output_dir)`**
- Horizontal bar chart comparing baseline vs MC Dropout AUC for each of the 14 disease classes
- Color-code bars: green if MC >= baseline, orange if MC < baseline

**`def plot_uncertainty_vs_accuracy(mean_probs, labels, uncertainty, output_dir)`**
- Scatter plot: X-axis = mean uncertainty, Y-axis = per-sample accuracy (fraction of correct binary labels)
- Should show negative correlation: high uncertainty → lower accuracy
- Add trend line (linear regression via numpy)

**`def plot_training_curves(log_dir, output_dir)`**
- Read TensorBoard event files using `tensorboard.backend.event_processing.event_accumulator`
- Plot train/val AUC and train/val loss over epochs on 2 subplots
- One figure per model (baseline and MC Dropout)

---

## File 14 — `scripts/preprocess.py`

Standalone script. Run once before training.

**Behavior:**
1. Read `config/default.yaml`
2. Walk through `data.nih_image_dir`
3. For each image: apply CLAHE (using `ClaheTransform` from `data/transforms.py`) and save the processed image to `data/processed/images/` (preserve original filenames)
4. Show a `tqdm` progress bar
5. At the end, print: total images processed, any failed images, output directory

**CLI args** (use `argparse`):
- `--config`: path to config yaml (default: `config/default.yaml`)
- `--dry-run`: process only first 100 images for verification

---

## File 15 — `scripts/train.py`

Main training entry point. Trains ONE model (either baseline or MC Dropout).

**CLI args** (argparse):
- `--config`: path to config yaml
- `--model`: `"densenet"` or `"resnet"` (default: `"densenet"`)
- `--experiment-name`: string, used for checkpoint and log subdirectory naming
- `--resume`: path to checkpoint to resume from (optional)

**Behavior:**
1. Load config from yaml
2. Build dataloaders via `create_dataloaders(config)`
3. Instantiate model (`MCDropoutDenseNet` or `ResNetBaseline`) based on `--model` flag
4. Move model to device (`cuda` if available, else `cpu`)
5. Get `pos_weights` from train dataset
6. Instantiate `Trainer` and call `trainer.train(freeze_epochs=config['model']['freeze_epochs'])`
7. Print final best validation AUC and ECE at the end

---

## File 16 — `scripts/evaluate.py`

Runs full test-set evaluation on a trained checkpoint.

**CLI args** (argparse):
- `--config`
- `--checkpoint`: path to `.pth` file
- `--model`: `"densenet"` or `"resnet"`
- `--mc-passes`: integer (default from config)
- `--output-dir`: where to save results and plots

**Behavior:**
1. Load model from checkpoint
2. Build test dataloader
3. Run `Evaluator.run(n_passes)` — use `n_passes=1` for baseline (standard inference, no MC)
4. Print formatted metrics table: AUC per class + mean AUC, ECE, NLL
5. Call all relevant `visualization/plots.py` functions: reliability diagram, AUC per class bar chart, uncertainty vs accuracy scatter
6. Save all results to `output_dir`

---

## File 17 — `scripts/evaluate_ood.py`

Runs OOD uncertainty analysis.

**CLI args** (argparse):
- `--config`
- `--checkpoint`: trained MC Dropout model checkpoint
- `--mc-passes`
- `--output-dir`

**Behavior:**
1. Load trained `MCDropoutDenseNet` from checkpoint
2. Build `CovidXDataset` loader (OOD) and a random subset of ChestXray14 test loader (in-distribution), same N samples
3. Run `OODEvaluator.run(n_passes)`
4. Print: OOD mean uncertainty, in-dist mean uncertainty, separation ratio
5. Call `plot_uncertainty_distribution()` and save to output_dir

---

## File 18 — `tests/test_dataset.py`

Unit tests using `unittest`. Test WITHOUT needing real data (use synthetic dummy data).

**Tests to include:**

- `test_dataset_len`: Create a small CSV with 10 rows, assert `len(dataset) == 10`
- `test_getitem_shapes`: Assert `image_tensor.shape == (3, 224, 224)` and `label_tensor.shape == (14,)` and `label_tensor.dtype == torch.float32`
- `test_pos_weights_shape`: Assert `pos_weights.shape == (14,)` and all values are positive
- `test_missing_image_handled`: Add a row with a non-existent filename, assert no crash and returned label is all zeros
- `test_clahe_transform`: Feed a random PIL Image, assert output is also PIL Image with same dimensions

---

## File 19 — `tests/test_model.py`

Unit tests for model forward pass and MC Dropout behavior.

**Tests to include:**

- `test_densenet_forward_shape`: Random input `(2, 3, 224, 224)`, assert output shape `(2, 14)`
- `test_resnet_forward_shape`: Same as above for ResNet baseline
- `test_mc_dropout_stochastic`: In MC mode (`enable_mc_dropout()`), run the same input twice. Assert the two outputs are NOT equal (stochasticity is active)
- `test_standard_eval_deterministic`: In standard `model.eval()` mode, run same input twice. Assert outputs ARE equal (no stochasticity)
- `test_freeze_backbone`: After `freeze_backbone()`, assert `model.model.features` parameters have `requires_grad == False`
- `test_unfreeze_backbone`: After `unfreeze_backbone()`, assert all parameters have `requires_grad == True`

---

## File 20 — `tests/test_uncertainty.py`

Unit tests for uncertainty and metrics functions.

**Tests to include:**

- `test_mc_predict_output_shapes`: Feed a batch of 4 images, T=10. Assert `mean_probs.shape == (4, 14)` and `uncertainty.shape == (4, 14)`
- `test_mc_predict_probs_in_range`: Assert all values in `mean_probs` are in `[0, 1]`
- `test_uncertainty_variance_nonneg`: Assert all values in `uncertainty` are >= 0
- `test_ece_perfect_calibration`: Build synthetic probs/labels where calibration is perfect. Assert ECE < 0.01
- `test_ece_range`: Assert ECE is always in `[0, 1]`
- `test_nll_finite`: Assert NLL is finite for reasonable inputs
- `test_auc_per_class_returns_all_classes`: Assert returned dict has all 14 class names as keys

---

## File 21 — `README.md`

Write a complete, usage-ready README with the following sections:

1. **Project Overview** — one-paragraph description of the thesis goal
2. **Setup**
   - Prerequisites (Python 3.10, CUDA GPU recommended)
   - `pip install -r requirements.txt`
   - Dataset download instructions (NIH ChestX-ray14 from Kaggle, Covid-X from GitHub)
   - Dataset directory layout expected
3. **Usage**
   - Step 1: Preprocess — `python scripts/preprocess.py --config config/default.yaml`
   - Step 2: Train baseline — `python scripts/train.py --model resnet --experiment-name baseline`
   - Step 3: Train MC Dropout model — `python scripts/train.py --model densenet --experiment-name mc_dropout`
   - Step 4: Evaluate — `python scripts/evaluate.py --checkpoint experiments/checkpoints/mc_dropout/best_model.pth --model densenet`
   - Step 5: OOD analysis — `python scripts/evaluate_ood.py --checkpoint ...`
4. **Configuration** — explain key config.yaml fields
5. **Expected Results** — table with target AUC/ECE for both models
6. **Project Structure** — the directory tree
7. **References** — the key papers (Gal & Ghahramani 2016, Wang et al. 2017, Mucsanyi et al. 2025)

---

## Build Order & Dependency Rules

Build files in this exact order. Each file depends only on files already built.

```
1.  requirements.txt          (no dependencies)
2.  config/default.yaml       (no dependencies)
3.  data/transforms.py        (no dependencies)
4.  data/dataset.py           (depends on: transforms.py)
5.  models/densenet.py        (no dependencies)
6.  models/resnet_baseline.py (no dependencies)
7.  uncertainty/mc_dropout.py (depends on: models/)
8.  uncertainty/metrics.py    (no dependencies — pure numpy/sklearn)
9.  training/losses.py        (no dependencies)
10. training/trainer.py       (depends on: losses.py, metrics.py)
11. evaluation/evaluator.py   (depends on: mc_dropout.py, metrics.py)
12. evaluation/ood_evaluator.py (depends on: mc_dropout.py)
13. visualization/plots.py    (depends on: metrics.py)
14. scripts/preprocess.py     (depends on: transforms.py, config)
15. scripts/train.py          (depends on: dataset.py, models/, trainer.py)
16. scripts/evaluate.py       (depends on: evaluator.py, plots.py)
17. scripts/evaluate_ood.py   (depends on: ood_evaluator.py, plots.py)
18. tests/test_dataset.py     (depends on: dataset.py, transforms.py)
19. tests/test_model.py       (depends on: models/)
20. tests/test_uncertainty.py (depends on: mc_dropout.py, metrics.py)
21. README.md                 (depends on: everything)
```

---

## Global Coding Standards

Apply these rules to every file without exception:

- **Type hints** on all function signatures
- **Docstrings** on every class and public function (Google style)
- **No hardcoded paths or hyperparameters** — all values come from config dict
- **Device agnostic** — always use `device` variable, never assume CUDA
- **Logging** — use Python's `logging` module (not `print`) except in scripts
- **Error handling** — wrap all file I/O and model loading in try/except with meaningful messages
- **Reproducibility** — scripts/train.py must call `torch.manual_seed(42)` and `np.random.seed(42)` at startup
- **No silent failures** — if a critical config key is missing, raise `KeyError` with a clear message

---

## Validation Checklist (Run After Each File)

After building each file, verify:

- [ ] File imports cleanly with no errors (`python -c "import <module>"`)
- [ ] All type hints are present on public functions
- [ ] No hardcoded magic numbers (everything from config)
- [ ] Unit tests pass for that module (if test file exists)
- [ ] Docstrings are present on all classes and public functions

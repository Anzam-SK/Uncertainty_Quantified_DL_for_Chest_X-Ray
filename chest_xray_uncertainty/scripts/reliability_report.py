"""
NIH ChestX-ray14 Final Reliability Report
==========================================
Runs T=50 MC Dropout passes on the held-out NIH test split and produces:

Metrics (printed to terminal + saved as JSON):
  - AUC per class + Mean AUC
  - Macro F1-Score (threshold = 0.5)
  - ECE  (Expected Calibration Error)
  - NLL  (Negative Log-Likelihood)
  - Mean predictive entropy

Plots (saved to --output-dir at 300 dpi):
  1. reliability_diagram.png        — calibration quality (confidence vs accuracy)
  2. uncertainty_heatmap.png        — per-disease MC Dropout variance
  3. entropy_histogram.png          — predictive entropy distribution
  4. uncertainty_accuracy_deciles.png — accuracy degrades as uncertainty rises

Usage:
  python scripts/reliability_report.py \
      --checkpoint experiments/checkpoints/thesis_112k_ultimate_run/best_model.pth
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import ChestXrayDataset
from data.transforms import get_val_transforms
from models.densenet import MCDropoutDenseNet
from uncertainty.mc_dropout import compute_predictive_entropy, mc_predict
from uncertainty.metrics import compute_all_metrics
from visualization.plots import (
    plot_entropy_histogram,
    plot_reliability_diagram,
    plot_uncertainty_deciles,
    plot_uncertainty_heatmap,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NIH ChestX-ray14 Final Reliability Report"
    )
    parser.add_argument(
        '--config', type=str, default='config/default.yaml',
        help='Path to config YAML file',
    )
    parser.add_argument(
        '--checkpoint', type=str,
        default='experiments/checkpoints/thesis_112k_ultimate_run/best_model.pth',
        help='Path to model checkpoint (.pth)',
    )
    parser.add_argument(
        '--mc-passes', type=int, default=50,
        help='Number of MC Dropout forward passes (default: 50)',
    )
    parser.add_argument(
        '--batch-size', type=int, default=None,
        help='Override batch size for inference (default: from config)',
    )
    parser.add_argument(
        '--output-dir', type=str,
        default='experiments/results/reliability_report',
        help='Directory to save all outputs',
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_mc_inference(
    model: MCDropoutDenseNet,
    loader: DataLoader,
    n_passes: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run MC Dropout inference over the full test loader.

    Returns:
        all_mean_probs  : (N, 14)  — mean probability across T passes
        all_uncertainty : (N, 14)  — variance across T passes
        all_entropy     : (N,)     — predictive entropy per sample
        all_labels      : (N, 14)  — ground truth
    """
    all_mean_probs:  list[np.ndarray] = []
    all_uncertainty: list[np.ndarray] = []
    all_entropy:     list[np.ndarray] = []
    all_labels:      list[np.ndarray] = []

    pbar = tqdm(loader, desc=f"MC Dropout (T={n_passes})", unit="batch")
    for images, labels, _ in pbar:
        images = images.to(device)

        mean_probs, uncertainty = mc_predict(model, images, n_passes, device)
        entropy = compute_predictive_entropy(mean_probs)

        all_mean_probs.append(mean_probs.cpu().numpy())
        all_uncertainty.append(uncertainty.cpu().numpy())
        all_entropy.append(entropy.cpu().numpy())
        all_labels.append(labels.numpy())

    return (
        np.concatenate(all_mean_probs,  axis=0),
        np.concatenate(all_uncertainty, axis=0),
        np.concatenate(all_entropy,     axis=0),
        np.concatenate(all_labels,      axis=0),
    )


# ---------------------------------------------------------------------------
# Terminal report
# ---------------------------------------------------------------------------
def print_report(
    metrics: dict,
    mean_entropy: float,
    n_samples: int,
    n_passes: int,
    checkpoint: str,
    elapsed: float,
    disease_classes: list[str],
) -> None:
    sep = '=' * 65
    print(f"\n{sep}")
    print("  NIH ChestX-ray14  —  FINAL RELIABILITY REPORT")
    print(sep)
    print(f"  Checkpoint : {checkpoint}")
    print(f"  Test samples: {n_samples:,}  |  MC passes: {n_passes}  |  Time: {elapsed:.1f}s")
    print(sep)

    print(f"\n{'  AUC per Disease Class':}")
    print(f"  {'Class':<25} {'AUC':>8}")
    print(f"  {'-'*34}")
    for cls in disease_classes:
        auc = metrics['auc_per_class'].get(cls)
        val = f"{auc:.4f}" if auc is not None else "  N/A "
        print(f"  {cls:<25} {val:>8}")

    print(f"  {'-'*34}")
    print(f"  {'Mean AUC':<25} {metrics['mean_auc']:>8.4f}")
    print(f"\n  {'Macro F1  (thr=0.5)':<25} {metrics['macro_f1']:>8.4f}")
    print(f"  {'ECE':<25} {metrics['ece']:>8.4f}")
    print(f"  {'NLL':<25} {metrics['nll']:>8.4f}")
    print(f"  {'Mean Entropy':<25} {mean_entropy:>8.4f}")
    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # --- CUDA required ---
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This project requires a CUDA-capable GPU.")
        sys.exit(1)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- Config ---
    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_config     = config['data']
    model_config    = config['model']
    disease_classes = config['disease_classes']
    batch_size      = args.batch_size or config['training']['batch_size']
    n_passes        = args.mc_passes
    output_dir      = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # --- Test dataset (NIH test_list.txt only — never seen during training) ---
    logger.info("Loading NIH test set...")
    test_dataset = ChestXrayDataset(
        image_dir=data_config['nih_image_dir'],
        labels_csv=data_config['nih_labels_csv'],
        transform=get_val_transforms(data_config['image_size']),
        disease_classes=disease_classes,
        split_file=data_config['nih_test_csv'],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    logger.info(f"Test set: {len(test_dataset):,} images")

    # --- Model ---
    model = MCDropoutDenseNet(
        num_classes=model_config['num_classes'],
        dropout_rate=model_config['dropout_rate'],
        pretrained=False,
    )
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    checkpoint_data = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint_data['model_state_dict'])
    model = model.to(device)
    logger.info(f"Loaded checkpoint: {args.checkpoint}")
    logger.info(
        f"Checkpoint epoch={checkpoint_data.get('epoch', '?')} | "
        f"val_auc={checkpoint_data.get('val_auc', 0):.4f}"
    )

    # --- MC Dropout inference ---
    print(f"\nRunning {n_passes} MC Dropout passes on {len(test_dataset):,} test images...")
    t0 = time.time()
    mean_probs, uncertainty, entropy, labels = run_mc_inference(
        model, test_loader, n_passes, device
    )
    elapsed = time.time() - t0

    # --- Metrics ---
    metrics = compute_all_metrics(
        mean_probs, labels, disease_classes,
        n_bins=config['evaluation']['calibration_bins'],
    )
    mean_entropy = float(np.mean(entropy))

    # --- Terminal report ---
    print_report(
        metrics, mean_entropy,
        n_samples=len(test_dataset),
        n_passes=n_passes,
        checkpoint=args.checkpoint,
        elapsed=elapsed,
        disease_classes=disease_classes,
    )

    # --- Save JSON summary ---
    summary = {
        "checkpoint":    args.checkpoint,
        "n_test_samples": len(test_dataset),
        "mc_passes":     n_passes,
        "mean_auc":      round(metrics['mean_auc'],  4),
        "macro_f1":      round(metrics['macro_f1'],  4),
        "ece":           round(metrics['ece'],        4),
        "nll":           round(metrics['nll'],        4),
        "mean_entropy":  round(mean_entropy,          4),
        "auc_per_class": {
            k: round(v, 4) if v is not None else None
            for k, v in metrics['auc_per_class'].items()
        },
    }
    json_path = os.path.join(output_dir, 'reliability_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved: {json_path}")

    # --- Save raw arrays ---
    np.save(os.path.join(output_dir, 'mean_probs.npy'),   mean_probs)
    np.save(os.path.join(output_dir, 'uncertainty.npy'),  uncertainty)
    np.save(os.path.join(output_dir, 'entropy.npy'),      entropy)
    np.save(os.path.join(output_dir, 'labels.npy'),       labels)

    # --- Plots ---
    print("Generating plots...")

    plot_reliability_diagram(
        mean_probs, labels,
        model_name='MCDropout-DenseNet121 (T=50)',
        output_dir=output_dir,
        n_bins=config['evaluation']['calibration_bins'],
        dpi=300,
    )
    print("  [1/4] reliability_diagram.png")

    plot_uncertainty_heatmap(
        uncertainty, disease_classes,
        output_dir=output_dir,
        dpi=300,
    )
    print("  [2/4] uncertainty_heatmap.png")

    plot_entropy_histogram(
        entropy,
        output_dir=output_dir,
        dpi=300,
    )
    print("  [3/4] entropy_histogram.png")

    plot_uncertainty_deciles(
        mean_probs, labels, uncertainty,
        output_dir=output_dir,
        dpi=300,
    )
    print("  [4/4] uncertainty_accuracy_deciles.png")

    print(f"\nAll outputs saved to: {output_dir}\n")


if __name__ == '__main__':
    main()

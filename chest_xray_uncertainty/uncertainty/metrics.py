"""Uncertainty and calibration metrics for model evaluation."""

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Measures the difference between predicted confidence and actual accuracy
    across probability bins. Lower is better (0 = perfectly calibrated).

    Args:
        probs: Predicted probabilities, shape (N, num_classes).
        labels: Ground truth binary labels, shape (N, num_classes).
        n_bins: Number of equal-width bins for [0, 1].

    Returns:
        ECE value in [0, 1].
    """
    # Flatten to treat each class prediction independently
    probs_flat = probs.flatten()
    labels_flat = labels.flatten()

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        # Find samples in this bin
        in_bin = (probs_flat >= bin_edges[i]) & (probs_flat < bin_edges[i + 1])
        n_in_bin = in_bin.sum()

        if n_in_bin == 0:
            continue

        # Average confidence and accuracy in this bin
        avg_confidence = probs_flat[in_bin].mean()
        avg_accuracy = labels_flat[in_bin].mean()

        # Weighted absolute difference
        ece += (n_in_bin / len(probs_flat)) * abs(avg_confidence - avg_accuracy)

    return float(ece)


def compute_nll(probs: np.ndarray, labels: np.ndarray) -> float:
    """Compute Negative Log-Likelihood (NLL) for binary multi-label predictions.

    NLL = -mean(y * log(p + eps) + (1 - y) * log(1 - p + eps))

    Args:
        probs: Predicted probabilities, shape (N, num_classes).
        labels: Ground truth binary labels, shape (N, num_classes).

    Returns:
        Mean NLL value.
    """
    eps = 1e-10
    # Clip to avoid log(0)
    probs_clipped = np.clip(probs, eps, 1.0 - eps)

    nll = -np.mean(
        labels * np.log(probs_clipped) + (1 - labels) * np.log(1 - probs_clipped)
    )
    return float(nll)


def compute_auc_per_class(
    probs: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
) -> Dict[str, Optional[float]]:
    """Compute ROC AUC for each disease class.

    Args:
        probs: Predicted probabilities, shape (N, num_classes).
        labels: Ground truth binary labels, shape (N, num_classes).
        class_names: List of class names.

    Returns:
        Dict mapping class name to AUC value (or None if class has only one label).
    """
    auc_dict = {}

    for i, name in enumerate(class_names):
        class_labels = labels[:, i]
        class_probs = probs[:, i]

        # Check if class has both positive and negative samples
        unique_labels = np.unique(class_labels)
        if len(unique_labels) < 2:
            auc_dict[name] = None
            continue

        try:
            auc = roc_auc_score(class_labels, class_probs)
            auc_dict[name] = float(auc)
        except Exception:
            auc_dict[name] = None

    return auc_dict


def compute_all_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    n_bins: int = 15,
) -> Dict:
    """Compute all evaluation metrics.

    Args:
        probs: Predicted probabilities, shape (N, num_classes).
        labels: Ground truth binary labels, shape (N, num_classes).
        class_names: List of class names.
        n_bins: Number of bins for ECE computation.

    Returns:
        Dict with keys: auc_per_class, mean_auc, ece, nll.
    """
    auc_per_class = compute_auc_per_class(probs, labels, class_names)

    # Compute mean AUC (excluding None values)
    valid_aucs = [v for v in auc_per_class.values() if v is not None]
    mean_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0

    ece = compute_ece(probs, labels, n_bins=n_bins)
    nll = compute_nll(probs, labels)

    # Macro F1 at 0.5 threshold across all classes
    preds_binary = (probs >= 0.5).astype(int)
    macro_f1 = float(f1_score(labels, preds_binary, average='macro', zero_division=0))

    return {
        "auc_per_class": auc_per_class,
        "mean_auc": mean_auc,
        "macro_f1": macro_f1,
        "ece": ece,
        "nll": nll,
    }

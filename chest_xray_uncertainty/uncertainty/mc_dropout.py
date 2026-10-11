"""
Monte Carlo Dropout inference for uncertainty quantification.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


def mc_predict(
    model: nn.Module,
    image_batch: torch.Tensor,
    n_passes: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run Monte Carlo Dropout inference over T stochastic forward passes.

    Args:
        model: MCDropoutDenseNet instance with enable_mc_dropout() method.
        image_batch: Input tensor of shape (batch_size, 3, 224, 224).
        n_passes: Number of stochastic forward passes (T). Recommended: 50.
        device: torch.device — cuda or cpu.

    Returns:
        mean_probs: shape (batch_size, num_classes) — mean probability across T passes.
        uncertainty: shape (batch_size, num_classes) — variance across T passes.
    """
    model.eval()
    model.enable_mc_dropout()  # keep only Dropout layers in train mode

    all_passes = []

    with torch.no_grad():
        for _ in range(n_passes):
            logits = model(image_batch.to(device))
            probs = torch.sigmoid(logits)         # sigmoid NOT softmax — multi-label
            all_passes.append(probs.unsqueeze(0)) # (1, batch_size, num_classes)

    # Stack: (n_passes, batch_size, num_classes)
    stacked = torch.cat(all_passes, dim=0)

    mean_probs = stacked.mean(dim=0)   # (batch_size, num_classes)
    uncertainty = stacked.var(dim=0)   # (batch_size, num_classes) — epistemic uncertainty

    return mean_probs, uncertainty


def compute_predictive_entropy(mean_probs: torch.Tensor) -> torch.Tensor:
    """
    Compute predictive entropy for multi-label classification (sigmoid outputs).

    For multi-label problems, each class is an independent Bernoulli variable.
    The correct formulation is the SUM of binary entropies across all classes —
    NOT standard Shannon entropy (which assumes a categorical/softmax distribution
    where probabilities sum to 1).

    Correct formula per sample:
        H = -sum_c [ p_c * log(p_c) + (1 - p_c) * log(1 - p_c) ]

    Args:
        mean_probs: shape (batch_size, num_classes) — sigmoid probabilities in [0, 1].

    Returns:
        entropy: shape (batch_size,) — scalar entropy value per sample.
                 Higher = more uncertain overall.
    """
    eps = 1e-10  # numerical stability — prevents log(0)

    # Entropy of the positive outcome: p * log(p)
    entropy_pos = mean_probs * torch.log(mean_probs + eps)

    # Entropy of the negative outcome: (1-p) * log(1-p)
    entropy_neg = (1.0 - mean_probs) * torch.log(1.0 - mean_probs + eps)

    # Sum binary entropies across all 14 classes → scalar per sample
    return -(entropy_pos + entropy_neg).sum(dim=1)


def classify_uncertainty(
    variance: torch.Tensor,
    threshold: float = 0.05
) -> List[str]:
    """
    Classify each sample's uncertainty as LOW, MEDIUM, or HIGH.

    Uses MAXIMUM variance across classes (not mean) for clinical safety.
    Rationale: if the model is highly uncertain about even ONE disease class,
    the entire scan must be flagged for radiologist review — averaging would
    hide a dangerous spike in uncertainty for a single critical condition.

    Args:
        variance: shape (batch_size, num_classes) — per-class variance from mc_predict.
        threshold: float — boundary between MEDIUM and HIGH uncertainty.
                   MEDIUM/LOW boundary is threshold / 2.

    Returns:
        List of strings, one per sample: "LOW", "MEDIUM", or "HIGH".
    """
    # Take the WORST-CASE class variance per sample — not the average
    max_variance = variance.max(dim=1).values  # shape (batch_size,)

    labels = []
    for v in max_variance:
        v = v.item()
        if v < threshold / 2:
            labels.append("LOW")
        elif v < threshold:
            labels.append("MEDIUM")
        else:
            labels.append("HIGH")

    return labels
"""Loss functions for chest X-ray classification training."""

import torch
import torch.nn as nn


def get_weighted_bce_loss(
    pos_weights: torch.Tensor,
    device: torch.device,
) -> nn.BCEWithLogitsLoss:
    """Create weighted BCE loss for handling class imbalance.

    Args:
        pos_weights: Tensor of shape (num_classes,) with pos_weight per class.
            pos_weight[i] = N_neg_i / N_pos_i.
        device: Device to move weights to.

    Returns:
        BCEWithLogitsLoss with pos_weight set.
    """
    pos_weights = pos_weights.to(device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weights)


def get_standard_bce_loss() -> nn.BCEWithLogitsLoss:
    """Create standard (unweighted) BCE loss.

    Returns:
        BCEWithLogitsLoss with no weighting.
    """
    return nn.BCEWithLogitsLoss()

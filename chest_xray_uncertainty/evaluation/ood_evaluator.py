"""OOD (Out-of-Distribution) evaluator for uncertainty analysis on Covid-X data."""

import logging
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from uncertainty.mc_dropout import compute_predictive_entropy, mc_predict

logger = logging.getLogger(__name__)


class OODEvaluator:
    """Measures uncertainty on out-of-distribution Covid-X images.

    Compares model uncertainty between in-distribution (NIH ChestX-ray14)
    and out-of-distribution (Covid-X) data to assess whether the model
    produces higher uncertainty for unfamiliar inputs.

    Args:
        model: Trained MC Dropout model.
        ood_loader: DataLoader for OOD data (Covid-X).
        indist_loader: DataLoader for in-distribution data (subset of NIH test set).
        config: Configuration dictionary.
        device: Device to run inference on.
    """

    def __init__(
        self,
        model: nn.Module,
        ood_loader: DataLoader,
        indist_loader: DataLoader,
        config: dict,
        device: torch.device,
    ) -> None:
        self.model = model
        self.ood_loader = ood_loader
        self.indist_loader = indist_loader
        self.config = config
        self.device = device

    def run(self, n_passes: int = 50) -> Dict:
        """Run OOD uncertainty analysis.

        Args:
            n_passes: Number of MC Dropout passes.

        Returns:
            Dict with OOD and in-distribution uncertainty statistics.
        """
        # Evaluate OOD data
        ood_mean_uncertainty, ood_entropy = self._evaluate_loader(
            self.ood_loader, n_passes, "OOD"
        )

        # Evaluate in-distribution data
        indist_mean_uncertainty, indist_entropy = self._evaluate_loader(
            self.indist_loader, n_passes, "In-Dist"
        )

        # Compute separation ratio
        ood_mean = float(np.mean(ood_mean_uncertainty))
        indist_mean = float(np.mean(indist_mean_uncertainty))
        separation_ratio = ood_mean / max(indist_mean, 1e-10)

        results = {
            "ood": {
                "mean_uncertainty": ood_mean_uncertainty,
                "entropy": ood_entropy,
                "mean_uncertainty_scalar": ood_mean,
            },
            "in_dist": {
                "mean_uncertainty": indist_mean_uncertainty,
                "entropy": indist_entropy,
                "mean_uncertainty_scalar": indist_mean,
            },
            "separation_ratio": separation_ratio,
        }

        logger.info(
            f"OOD Analysis: "
            f"OOD mean uncertainty={ood_mean:.6f}, "
            f"In-dist mean uncertainty={indist_mean:.6f}, "
            f"Separation ratio={separation_ratio:.2f}"
        )

        return results

    def _evaluate_loader(
        self, loader: DataLoader, n_passes: int, label: str
    ) -> tuple:
        """Evaluate uncertainty on a single data loader.

        Args:
            loader: DataLoader to evaluate.
            n_passes: Number of MC Dropout passes.
            label: Label for progress bar.

        Returns:
            Tuple of (mean_uncertainty_array, entropy_array).
        """
        all_mean_uncertainty = []
        all_entropy = []

        self.model.eval()

        pbar = tqdm(loader, desc=f"MC Dropout ({label})")
        for batch in pbar:
            # Handle both (image, filename) and (image, label, filename) formats
            if len(batch) == 2:
                images, _ = batch
            else:
                images, _, _ = batch

            images = images.to(self.device)

            mean_probs, uncertainty = mc_predict(
                self.model, images, n_passes, self.device
            )

            # Mean variance across classes per sample
            mean_var = uncertainty.mean(dim=1).cpu().numpy()
            entropy = compute_predictive_entropy(mean_probs).cpu().numpy()

            all_mean_uncertainty.append(mean_var)
            all_entropy.append(entropy)

        return np.concatenate(all_mean_uncertainty), np.concatenate(all_entropy)

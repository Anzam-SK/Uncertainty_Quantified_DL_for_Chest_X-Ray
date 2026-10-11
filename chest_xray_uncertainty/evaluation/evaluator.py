"""Evaluator for running full test-set evaluation with MC Dropout inference."""

import json
import logging
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from uncertainty.mc_dropout import compute_predictive_entropy, mc_predict
from uncertainty.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


class Evaluator:
    """Runs full evaluation on the test set using MC Dropout inference.

    Args:
        model: Trained model (should have enable_mc_dropout method for MC models).
        test_loader: DataLoader for test data.
        config: Configuration dictionary.
        device: Device to run inference on.
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        config: dict,
        device: torch.device,
    ) -> None:
        self.model = model
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.disease_classes = config['disease_classes']

    def run(self, n_passes: int = 1) -> Dict:
        """Run evaluation on the full test set.

        Args:
            n_passes: Number of MC Dropout passes. Use 1 for standard inference.

        Returns:
            Dict with metrics, predictions, uncertainty, and filenames.
        """
        all_mean_probs: List[np.ndarray] = []
        all_uncertainty: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        all_filenames: List[str] = []

        self.model.eval()

        pbar = tqdm(self.test_loader, desc="Evaluating")
        for images, labels, filenames in pbar:
            images = images.to(self.device)

            if n_passes > 1:
                # MC Dropout inference
                mean_probs, uncertainty = mc_predict(
                    self.model, images, n_passes, self.device
                )
            else:
                # Standard inference
                with torch.no_grad():
                    logits = self.model(images)
                    mean_probs = torch.sigmoid(logits)
                    uncertainty = torch.zeros_like(mean_probs)

            all_mean_probs.append(mean_probs.cpu().numpy())
            all_uncertainty.append(uncertainty.cpu().numpy())
            all_labels.append(labels.numpy())
            all_filenames.extend(filenames)

        # Concatenate all batches
        all_mean_probs = np.concatenate(all_mean_probs, axis=0)
        all_uncertainty = np.concatenate(all_uncertainty, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Compute metrics
        metrics = compute_all_metrics(
            all_mean_probs, all_labels, self.disease_classes
        )

        # Compute per-sample uncertainty measures
        mean_uncertainty = all_uncertainty.mean(axis=1)  # Mean variance across classes
        entropy = compute_predictive_entropy(
            torch.tensor(all_mean_probs)
        ).numpy()

        results = {
            "metrics": metrics,
            "predictions": all_mean_probs,
            "uncertainty_variance": all_uncertainty,
            "mean_uncertainty": mean_uncertainty,
            "entropy": entropy,
            "labels": all_labels,
            "filenames": all_filenames,
        }

        logger.info(
            f"Evaluation complete: "
            f"Mean AUC={metrics['mean_auc']:.4f}, "
            f"ECE={metrics['ece']:.4f}, "
            f"NLL={metrics['nll']:.4f}"
        )

        return results

    @staticmethod
    def save_results(results: Dict, output_path: str) -> None:
        """Save evaluation results to disk.

        Saves metrics as JSON and numpy arrays as .npy files.

        Args:
            results: Results dict from run().
            output_path: Directory to save results to.
        """
        os.makedirs(output_path, exist_ok=True)

        # Save metrics as JSON
        metrics_path = os.path.join(output_path, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(results["metrics"], f, indent=2)

        # Save numpy arrays
        np.save(os.path.join(output_path, "predictions.npy"), results["predictions"])
        np.save(
            os.path.join(output_path, "uncertainty_variance.npy"),
            results["uncertainty_variance"],
        )
        np.save(
            os.path.join(output_path, "mean_uncertainty.npy"),
            results["mean_uncertainty"],
        )
        np.save(os.path.join(output_path, "entropy.npy"), results["entropy"])
        np.save(os.path.join(output_path, "labels.npy"), results["labels"])

        # Save filenames
        with open(os.path.join(output_path, "filenames.txt"), 'w') as f:
            f.write('\n'.join(results["filenames"]))

        logger.info(f"Results saved to {output_path}")

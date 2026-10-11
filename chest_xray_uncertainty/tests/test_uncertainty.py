"""Unit tests for uncertainty and metrics functions."""

import os
import unittest

import numpy as np
import torch

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.densenet import MCDropoutDenseNet
from uncertainty.mc_dropout import classify_uncertainty, compute_predictive_entropy, mc_predict
from uncertainty.metrics import compute_all_metrics, compute_auc_per_class, compute_ece, compute_nll


class TestMCDropout(unittest.TestCase):
    """Tests for MC Dropout inference functions."""

    def setUp(self):
        """Create model and dummy batch."""
        self.model = MCDropoutDenseNet(num_classes=14, dropout_rate=0.3, pretrained=False)
        self.device = torch.device("cpu")
        self.dummy_batch = torch.randn(4, 3, 224, 224)

    def test_mc_predict_output_shapes(self):
        """mc_predict should return correct shapes."""
        mean_probs, uncertainty = mc_predict(
            self.model, self.dummy_batch, n_passes=10, device=self.device
        )
        self.assertEqual(mean_probs.shape, (4, 14))
        self.assertEqual(uncertainty.shape, (4, 14))

    def test_mc_predict_probs_in_range(self):
        """Mean probabilities should be in [0, 1]."""
        mean_probs, _ = mc_predict(
            self.model, self.dummy_batch, n_passes=10, device=self.device
        )
        self.assertTrue((mean_probs >= 0).all())
        self.assertTrue((mean_probs <= 1).all())

    def test_uncertainty_variance_nonneg(self):
        """Uncertainty (variance) should be non-negative."""
        _, uncertainty = mc_predict(
            self.model, self.dummy_batch, n_passes=10, device=self.device
        )
        self.assertTrue((uncertainty >= 0).all())

    def test_predictive_entropy_shape(self):
        """Predictive entropy should return shape (batch_size,)."""
        mean_probs, _ = mc_predict(
            self.model, self.dummy_batch, n_passes=10, device=self.device
        )
        entropy = compute_predictive_entropy(mean_probs)
        self.assertEqual(entropy.shape, (4,))

    def test_classify_uncertainty_levels(self):
        """classify_uncertainty should return valid levels."""
        _, uncertainty = mc_predict(
            self.model, self.dummy_batch, n_passes=10, device=self.device
        )
        levels = classify_uncertainty(uncertainty, threshold=0.05)
        self.assertEqual(len(levels), 4)
        for level in levels:
            self.assertIn(level, ["LOW", "MEDIUM", "HIGH"])


class TestMetrics(unittest.TestCase):
    """Tests for evaluation metrics."""

    def test_ece_perfect_calibration(self):
        """ECE should be near zero for perfectly calibrated predictions."""
        np.random.seed(42)
        n = 1000
        # Create perfectly calibrated data: prob = accuracy
        probs = np.random.rand(n, 14)
        labels = (probs > 0.5).astype(float)  # Labels match threshold of probs
        ece = compute_ece(probs, labels, n_bins=15)
        self.assertLess(ece, 0.5)  # Should be reasonably low

    def test_ece_range(self):
        """ECE should always be in [0, 1]."""
        np.random.seed(42)
        probs = np.random.rand(100, 14)
        labels = np.random.randint(0, 2, (100, 14)).astype(float)
        ece = compute_ece(probs, labels, n_bins=15)
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)

    def test_nll_finite(self):
        """NLL should be finite for reasonable inputs."""
        np.random.seed(42)
        probs = np.random.rand(100, 14)
        labels = np.random.randint(0, 2, (100, 14)).astype(float)
        nll = compute_nll(probs, labels)
        self.assertTrue(np.isfinite(nll))

    def test_nll_perfect_prediction(self):
        """NLL should be near zero for perfect predictions."""
        probs = np.array([[0.99, 0.01, 0.99, 0.01]])
        labels = np.array([[1.0, 0.0, 1.0, 0.0]])
        nll = compute_nll(probs, labels)
        self.assertLess(nll, 0.1)

    def test_auc_per_class_returns_all_classes(self):
        """compute_auc_per_class should return all 14 class names."""
        class_names = [
            'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
            'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
            'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia',
        ]
        np.random.seed(42)
        probs = np.random.rand(100, 14)
        labels = np.random.randint(0, 2, (100, 14)).astype(float)
        auc_dict = compute_auc_per_class(probs, labels, class_names)
        self.assertEqual(set(auc_dict.keys()), set(class_names))

    def test_compute_all_metrics_keys(self):
        """compute_all_metrics should return all expected keys."""
        class_names = [
            'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
            'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
            'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia',
        ]
        np.random.seed(42)
        probs = np.random.rand(100, 14)
        labels = np.random.randint(0, 2, (100, 14)).astype(float)
        metrics = compute_all_metrics(probs, labels, class_names)
        self.assertIn('auc_per_class', metrics)
        self.assertIn('mean_auc', metrics)
        self.assertIn('ece', metrics)
        self.assertIn('nll', metrics)


if __name__ == '__main__':
    unittest.main()

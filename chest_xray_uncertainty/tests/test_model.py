"""Unit tests for model architectures and MC Dropout behavior."""

import os
import unittest

import torch

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.densenet import MCDropoutDenseNet
from models.resnet_baseline import ResNetBaseline


class TestDenseNet(unittest.TestCase):
    """Tests for MCDropoutDenseNet."""

    def setUp(self):
        """Create model with no pretrained weights for fast testing."""
        self.model = MCDropoutDenseNet(num_classes=14, dropout_rate=0.3, pretrained=False)
        self.dummy_input = torch.randn(2, 3, 224, 224)

    def test_densenet_forward_shape(self):
        """Forward pass should produce correct output shape."""
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.dummy_input)
        self.assertEqual(output.shape, (2, 14))

    def test_mc_dropout_stochastic(self):
        """MC Dropout mode should produce different outputs for same input."""
        self.model.enable_mc_dropout()
        with torch.no_grad():
            out1 = self.model(self.dummy_input)
            out2 = self.model(self.dummy_input)
        # Outputs should differ due to dropout stochasticity
        self.assertFalse(torch.allclose(out1, out2),
                         "MC Dropout outputs must differ across passes")

    def test_standard_eval_deterministic(self):
        """Standard eval mode should produce identical outputs for same input."""
        self.model.eval()
        with torch.no_grad():
            out1 = self.model(self.dummy_input)
            out2 = self.model(self.dummy_input)
        self.assertTrue(torch.allclose(out1, out2),
                        "Standard eval outputs must be identical")

    def test_freeze_backbone(self):
        """freeze_backbone should set all feature parameters to requires_grad=False."""
        self.model.freeze_backbone()
        for param in self.model.model.features.parameters():
            self.assertFalse(param.requires_grad)

    def test_unfreeze_backbone(self):
        """unfreeze_backbone should set all parameters to requires_grad=True."""
        self.model.freeze_backbone()
        self.model.unfreeze_backbone()
        for param in self.model.parameters():
            self.assertTrue(param.requires_grad)


class TestResNet(unittest.TestCase):
    """Tests for ResNetBaseline."""

    def setUp(self):
        """Create model with no pretrained weights for fast testing."""
        self.model = ResNetBaseline(num_classes=14, pretrained=False)
        self.dummy_input = torch.randn(2, 3, 224, 224)

    def test_resnet_forward_shape(self):
        """Forward pass should produce correct output shape."""
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.dummy_input)
        self.assertEqual(output.shape, (2, 14))

    def test_freeze_backbone(self):
        """freeze_backbone should freeze all non-fc parameters."""
        self.model.freeze_backbone()
        for name, param in self.model.model.named_parameters():
            if not name.startswith('fc.'):
                self.assertFalse(param.requires_grad)

    def test_unfreeze_backbone(self):
        """unfreeze_backbone should unfreeze all parameters."""
        self.model.freeze_backbone()
        self.model.unfreeze_backbone()
        for param in self.model.parameters():
            self.assertTrue(param.requires_grad)


if __name__ == '__main__':
    unittest.main()

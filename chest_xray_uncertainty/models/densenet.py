"""MC Dropout DenseNet-121 model for chest X-ray classification with uncertainty."""

import torch
import torch.nn as nn
from torchvision import models


class MCDropoutDenseNet(nn.Module):
    """DenseNet-121 with MC Dropout for uncertainty quantification.

    Replaces the standard classifier head with a custom head containing
    dropout layers that remain active during inference for Monte Carlo
    Dropout uncertainty estimation.

    Args:
        num_classes: Number of output classes (14 for ChestX-ray14).
        dropout_rate: Dropout probability for MC Dropout.
        pretrained: Whether to use ImageNet-pretrained weights.
    """

    def __init__(
        self,
        num_classes: int = 14,
        dropout_rate: float = 0.3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        # Load pretrained DenseNet-121
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.model = models.densenet121(weights=weights)

        # Replace classifier head with MC Dropout head
        # DenseNet-121 has 1024 features from the backbone
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes),
        )

    def freeze_backbone(self) -> None:
        """Freeze all backbone (feature extractor) parameters."""
        for param in self.model.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone (feature extractor) parameters."""
        for param in self.model.features.parameters():
            param.requires_grad = True

    def enable_mc_dropout(self) -> None:
        """Enable MC Dropout: set only Dropout layers to train mode.

        This is the critical method for MC Dropout inference. The model
        is set to eval mode first, then this method re-enables only the
        dropout layers so they remain stochastic during inference.
        """
        self.eval()
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, H, W).

        Returns:
            Raw logits of shape (batch_size, num_classes).
        """
        return self.model(x)

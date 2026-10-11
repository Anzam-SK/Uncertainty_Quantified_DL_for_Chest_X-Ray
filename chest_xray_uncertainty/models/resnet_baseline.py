"""ResNet-50 baseline model for chest X-ray classification (no MC Dropout)."""

import torch
import torch.nn as nn
from torchvision import models


class ResNetBaseline(nn.Module):
    """ResNet-50 baseline model without MC Dropout.

    Standard fine-tuned ResNet-50 used as a comparison baseline against
    the MC Dropout DenseNet model.

    Args:
        num_classes: Number of output classes (14 for ChestX-ray14).
        pretrained: Whether to use ImageNet-pretrained weights.
    """

    def __init__(
        self,
        num_classes: int = 14,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        # Load pretrained ResNet-50
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.model = models.resnet50(weights=weights)

        # Replace final fully connected layer
        # ResNet-50 has 2048 features from the backbone
        self.model.fc = nn.Linear(2048, num_classes)

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (everything except fc layer)."""
        for name, param in self.model.named_parameters():
            if not name.startswith('fc.'):
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters."""
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, H, W).

        Returns:
            Raw logits of shape (batch_size, num_classes).
        """
        return self.model(x)

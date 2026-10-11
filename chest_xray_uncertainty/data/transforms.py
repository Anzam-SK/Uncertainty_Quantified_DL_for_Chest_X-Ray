"""Image transform pipelines for chest X-ray preprocessing and augmentation."""

from typing import Callable

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class ClaheTransform:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to a PIL Image.

    Converts the image to grayscale, applies CLAHE, then converts back to a
    3-channel RGB PIL Image by stacking the enhanced grayscale channel.

    Args:
        clip_limit: Threshold for contrast limiting.
        tile_size: Size of grid for histogram equalization.
    """

    def __init__(self, clip_limit: float = 2.0, tile_size: int = 8) -> None:
        self.clip_limit = clip_limit
        self.tile_size = tile_size
        # cv2.CLAHE is not picklable — do NOT store it on self.
        # It is created fresh inside __call__ so worker processes can pickle this class.

    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply CLAHE to a PIL Image.

        Args:
            image: Input PIL Image (RGB or grayscale).

        Returns:
            Enhanced PIL Image in RGB mode.
        """
        # Convert PIL to numpy array
        img_array = np.array(image)

        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Create CLAHE here — avoids pickle error on Windows multiprocessing
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=(self.tile_size, self.tile_size),
        )
        enhanced = clahe.apply(gray)

        # Stack back to 3 channels (RGB)
        enhanced_rgb = np.stack([enhanced, enhanced, enhanced], axis=-1)

        return Image.fromarray(enhanced_rgb)


def get_train_transforms(image_size: int) -> transforms.Compose:
    """Build training transform pipeline with augmentation.

    Args:
        image_size: Target image size (square).

    Returns:
        Composed transform pipeline for training.
    """
    return transforms.Compose([
        ClaheTransform(clip_limit=2.0, tile_size=8),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_val_transforms(image_size: int) -> transforms.Compose:
    """Build validation/test transform pipeline without augmentation.

    Args:
        image_size: Target image size (square).

    Returns:
        Composed transform pipeline for validation/testing.
    """
    return transforms.Compose([
        ClaheTransform(clip_limit=2.0, tile_size=8),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

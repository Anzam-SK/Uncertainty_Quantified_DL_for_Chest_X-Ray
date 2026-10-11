"""Dataset classes for chest X-ray images (NIH ChestX-ray14 and Covid-X)."""

import glob
import logging
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data.transforms import get_train_transforms, get_val_transforms

logger = logging.getLogger(__name__)


def _build_image_index(image_dir: str) -> Dict[str, str]:
    """Build a mapping from filename to full path, searching recursively.

    NIH ChestX-ray14 stores images in subdirectories like
    images_001/images/, images_002/images/, etc.

    Args:
        image_dir: Root directory to search for images.

    Returns:
        Dict mapping filename (e.g. '00000001_000.png') to full path.
    """
    index = {}
    # Search recursively for all PNG/JPEG files
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG'):
        for path in glob.glob(os.path.join(image_dir, '**', ext), recursive=True):
            filename = os.path.basename(path)
            index[filename] = path
    logger.info(f"Built image index with {len(index)} images from {image_dir}")
    return index


class ChestXrayDataset(Dataset):
    """NIH ChestX-ray14 dataset for multi-label disease classification.

    Supports two modes:
    1. **NIH format** (split_file provided): Labels in Data_Entry_2017.csv with
       pipe-separated disease names in 'Finding Labels' column. Images stored
       in subdirectories (images_001/images/ through images_012/images/).
    2. **Simple CSV format** (no split_file): CSV has 'Image Index' column and
       one binary column per disease class. Images in flat directory.

    Args:
        image_dir: Path containing chest X-ray images (searched recursively).
        labels_csv: Path to CSV with image labels.
        transform: Optional transform pipeline to apply to images.
        disease_classes: Ordered list of 14 disease class names.
        split_file: Optional path to split file (train_val_list.txt or test_list.txt).
            If provided, enables NIH format parsing with pipe-separated labels.
    """

    def __init__(
        self,
        image_dir: str,
        labels_csv: str,
        transform: Optional[Callable] = None,
        disease_classes: Optional[List[str]] = None,
        split_file: Optional[str] = None,
    ) -> None:
        self.image_dir = image_dir
        self.transform = transform

        if disease_classes is None:
            raise ValueError("disease_classes must be provided")
        self.disease_classes = disease_classes

        # Build image path index (recursive search for subdirectory support)
        self.image_index = _build_image_index(image_dir)

        # Load labels
        self.df = self._load_labels(labels_csv, split_file)
        logger.info(
            f"Loaded {len(self.df)} samples from {labels_csv}"
            + (f" (split: {split_file})" if split_file else "")
        )

    def _load_labels(self, labels_csv: str, split_file: Optional[str] = None) -> pd.DataFrame:
        """Load and parse the labels file.

        Supports two formats:
        1. NIH format (split_file provided): Data_Entry_2017.csv with pipe-separated
           'Finding Labels' column, filtered by split file.
        2. Simple CSV: Has 'Image Index' and one binary column per disease class.

        Args:
            labels_csv: Path to the labels file.
            split_file: Optional path to split file.

        Returns:
            DataFrame with 'Image Index' and binary label columns.
        """
        try:
            df = pd.read_csv(labels_csv)

            if 'Image Index' not in df.columns:
                raise KeyError(
                    f"'Image Index' column not found in {labels_csv}. "
                    f"Available columns: {list(df.columns)}"
                )

            # NIH format: pipe-separated 'Finding Labels' + split file
            if split_file is not None and 'Finding Labels' in df.columns:
                # Load split file (one filename per line)
                with open(split_file, 'r') as f:
                    split_filenames = set(line.strip() for line in f if line.strip())

                # Filter to only images in this split
                df = df[df['Image Index'].isin(split_filenames)].copy()
                logger.info(f"Filtered to {len(df)} samples in split {split_file}")

                # Parse pipe-separated 'Finding Labels' into binary columns
                for cls in self.disease_classes:
                    df[cls] = df['Finding Labels'].apply(
                        lambda labels, c=cls: 1.0 if c in labels.split('|') else 0.0
                    )

                # Filter to only images that exist on disk
                exists_mask = df['Image Index'].apply(lambda x: x in self.image_index)
                missing = (~exists_mask).sum()
                if missing > 0:
                    logger.warning(
                        f"{missing} images listed in split but not found on disk. "
                        f"These will be skipped."
                    )
                df = df[exists_mask].reset_index(drop=True)

            else:
                # Simple CSV format: ensure all disease class columns exist
                for cls in self.disease_classes:
                    if cls not in df.columns:
                        matching = [c for c in df.columns if c.lower() == cls.lower()]
                        if matching:
                            df.rename(columns={matching[0]: cls}, inplace=True)
                        else:
                            logger.warning(
                                f"Class '{cls}' not found in CSV. Setting all to 0."
                            )
                            df[cls] = 0

            return df

        except Exception as e:
            raise RuntimeError(
                f"Failed to load labels from {labels_csv}: {e}"
            ) from e

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image_tensor, label_tensor, image_filename).
            - image_tensor: shape (3, H, W), float32
            - label_tensor: shape (14,), float32, binary
            - image_filename: str
        """
        row = self.df.iloc[idx]
        filename = row['Image Index']

        # Look up full path from image index
        image_path = self.image_index.get(filename)
        if image_path is None:
            logger.warning(f"Image not in index: {filename}. Returning zeros.")
            image = Image.new('RGB', (224, 224), color=0)
            labels = torch.zeros(len(self.disease_classes), dtype=torch.float32)
            if self.transform is not None:
                image = self.transform(image)
            else:
                from torchvision import transforms
                image = transforms.ToTensor()(image)
            return image, labels, filename

        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except (FileNotFoundError, OSError) as e:
            logger.warning(f"Image not found or corrupted: {image_path}. Returning zeros.")
            image = Image.new('RGB', (224, 224), color=0)
            labels = torch.zeros(len(self.disease_classes), dtype=torch.float32)
            if self.transform is not None:
                image = self.transform(image)
            else:
                from torchvision import transforms
                image = transforms.ToTensor()(image)
            return image, labels, filename

        # Extract labels
        labels = torch.tensor(
            [float(row[cls]) for cls in self.disease_classes],
            dtype=torch.float32,
        )

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        return image, labels, filename

    def get_pos_weights(self) -> torch.Tensor:
        """Compute positive class weights for weighted BCE loss.

        Returns:
            Tensor of shape (14,) where pos_weight[i] = N_neg_i / N_pos_i.
        """
        weights = []
        for cls in self.disease_classes:
            n_pos = self.df[cls].sum()
            n_neg = len(self.df) - n_pos
            if n_pos == 0:
                weights.append(1.0)
                logger.warning(f"Class '{cls}' has 0 positive samples. Using weight 1.0.")
            else:
                weights.append(n_neg / n_pos)

        return torch.tensor(weights, dtype=torch.float32)


class CovidXDataset(Dataset):
    """Covid-X dataset for out-of-distribution uncertainty evaluation.

    This dataset has no labels — it is used only to measure model uncertainty
    on data from a different distribution.

    Covid-X split files are space-separated with format:
        ID  filename  label  source

    Args:
        image_dir: Path to folder containing Covid-X images (e.g. data/Covid-X/test/).
        split_file: Path to split file (e.g. data/Covid-X/test.txt).
        transform: Optional transform pipeline.
    """

    def __init__(
        self,
        image_dir: str,
        split_file: str,
        transform: Optional[Callable] = None,
    ) -> None:
        self.image_dir = image_dir
        self.transform = transform

        try:
            # Covid-X split files are space-separated: ID filename label source
            self.filenames = []
            with open(split_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        self.filenames.append(parts[1])  # filename is second column
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Covid-X split from {split_file}: {e}"
            ) from e

        logger.info(f"Loaded {len(self.filenames)} OOD samples from {split_file}")

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """Get a single sample (no labels).

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image_tensor, image_filename).
        """
        filename = self.filenames[idx]
        image_path = os.path.join(self.image_dir, filename)

        try:
            image = Image.open(image_path).convert('RGB')
        except (FileNotFoundError, OSError):
            logger.warning(f"Image not found: {image_path}. Returning zeros.")
            image = Image.new('RGB', (224, 224), color=0)
            if self.transform is not None:
                image = self.transform(image)
            else:
                from torchvision import transforms
                image = transforms.ToTensor()(image)
            return image, filename

        if self.transform is not None:
            image = self.transform(image)

        return image, filename


def create_dataloaders(
    config: dict,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders from config.

    Args:
        config: Configuration dictionary loaded from YAML.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    data_config = config['data']
    training_config = config['training']
    disease_classes = config['disease_classes']

    image_size = data_config['image_size']
    batch_size = training_config['batch_size']
    # cv2.CLAHE is created inside __call__ (not stored on self), so it is picklable.
    # num_workers=2 is safe on Windows with this fix.
    num_workers = data_config.get('num_workers', 2)

    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)

    # Training dataset
    train_dataset = ChestXrayDataset(
        image_dir=data_config['nih_image_dir'],
        labels_csv=data_config['nih_labels_csv'],
        transform=train_transform,
        disease_classes=disease_classes,
        split_file=data_config['nih_train_csv'],
    )

    # Test dataset (using val transforms)
    test_dataset = ChestXrayDataset(
        image_dir=data_config['nih_image_dir'],
        labels_csv=data_config['nih_labels_csv'],
        transform=val_transform,
        disease_classes=disease_classes,
        split_file=data_config['nih_test_csv'],
    )

    # Split training into train/val (80/20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Full dataset — no slice. All images from train_val_list.txt are used.
    # test_list.txt images are in test_dataset only and never touch training.

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    logger.info(
        f"Created dataloaders: train={len(train_subset)}, "
        f"val={len(val_subset)}, test={len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader

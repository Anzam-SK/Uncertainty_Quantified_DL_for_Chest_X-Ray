"""Unit tests for dataset classes and transforms."""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from PIL import Image

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import ChestXrayDataset
from data.transforms import ClaheTransform, get_train_transforms, get_val_transforms


class TestClaheTransform(unittest.TestCase):
    """Tests for ClaheTransform."""

    def test_clahe_output_is_pil(self):
        """CLAHE transform should return a PIL Image."""
        transform = ClaheTransform(clip_limit=2.0, tile_size=8)
        img = Image.new('RGB', (224, 224), color=128)
        result = transform(img)
        self.assertIsInstance(result, Image.Image)

    def test_clahe_preserves_dimensions(self):
        """CLAHE transform should preserve image dimensions."""
        transform = ClaheTransform(clip_limit=2.0, tile_size=8)
        img = Image.new('RGB', (256, 256), color=128)
        result = transform(img)
        self.assertEqual(result.size, (256, 256))

    def test_clahe_output_is_rgb(self):
        """CLAHE transform should return RGB image."""
        transform = ClaheTransform(clip_limit=2.0, tile_size=8)
        img = Image.new('RGB', (224, 224), color=128)
        result = transform(img)
        self.assertEqual(result.mode, 'RGB')


class TestTransforms(unittest.TestCase):
    """Tests for transform pipelines."""

    def test_train_transforms_output_shape(self):
        """Train transforms should produce correct tensor shape."""
        transform = get_train_transforms(224)
        img = Image.new('RGB', (256, 256), color=128)
        result = transform(img)
        self.assertEqual(result.shape, (3, 224, 224))
        self.assertEqual(result.dtype, torch.float32)

    def test_val_transforms_output_shape(self):
        """Val transforms should produce correct tensor shape."""
        transform = get_val_transforms(224)
        img = Image.new('RGB', (256, 256), color=128)
        result = transform(img)
        self.assertEqual(result.shape, (3, 224, 224))
        self.assertEqual(result.dtype, torch.float32)


class TestChestXrayDataset(unittest.TestCase):
    """Tests for ChestXrayDataset."""

    def setUp(self):
        """Create temporary dataset with synthetic data."""
        self.tmpdir = tempfile.mkdtemp()
        self.image_dir = os.path.join(self.tmpdir, 'images')
        os.makedirs(self.image_dir)

        self.disease_classes = [
            'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
            'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
            'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia',
        ]

        # Create 10 synthetic images and labels
        self.n_samples = 10
        filenames = []
        for i in range(self.n_samples):
            fname = f"img_{i:03d}.png"
            img = Image.new('RGB', (224, 224), color=(i * 25, i * 25, i * 25))
            img.save(os.path.join(self.image_dir, fname))
            filenames.append(fname)

        # Create labels CSV
        data = {'Image Index': filenames}
        for cls in self.disease_classes:
            data[cls] = np.random.randint(0, 2, self.n_samples)
        df = pd.DataFrame(data)
        self.csv_path = os.path.join(self.tmpdir, 'labels.csv')
        df.to_csv(self.csv_path, index=False)

    def test_dataset_len(self):
        """Dataset length should match number of samples."""
        dataset = ChestXrayDataset(
            image_dir=self.image_dir,
            labels_csv=self.csv_path,
            transform=get_val_transforms(224),
            disease_classes=self.disease_classes,
        )
        self.assertEqual(len(dataset), self.n_samples)

    def test_getitem_shapes(self):
        """__getitem__ should return correct shapes and types."""
        dataset = ChestXrayDataset(
            image_dir=self.image_dir,
            labels_csv=self.csv_path,
            transform=get_val_transforms(224),
            disease_classes=self.disease_classes,
        )
        image, labels, filename = dataset[0]
        self.assertEqual(image.shape, (3, 224, 224))
        self.assertEqual(labels.shape, (14,))
        self.assertEqual(labels.dtype, torch.float32)
        self.assertIsInstance(filename, str)

    def test_pos_weights_shape(self):
        """get_pos_weights should return tensor of shape (14,) with positive values."""
        dataset = ChestXrayDataset(
            image_dir=self.image_dir,
            labels_csv=self.csv_path,
            transform=get_val_transforms(224),
            disease_classes=self.disease_classes,
        )
        weights = dataset.get_pos_weights()
        self.assertEqual(weights.shape, (14,))
        self.assertTrue((weights > 0).all())

    def test_missing_image_handled(self):
        """Missing images should be handled gracefully (no crash, zero labels)."""
        # Add a row with non-existent filename
        import pandas as pd
        df = pd.read_csv(self.csv_path)
        new_row = {'Image Index': 'nonexistent.png'}
        for cls in self.disease_classes:
            new_row[cls] = 0
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(self.csv_path, index=False)

        dataset = ChestXrayDataset(
            image_dir=self.image_dir,
            labels_csv=self.csv_path,
            transform=get_val_transforms(224),
            disease_classes=self.disease_classes,
        )
        # Should not crash
        image, labels, filename = dataset[len(dataset) - 1]
        self.assertEqual(filename, 'nonexistent.png')
        self.assertTrue((labels == 0).all())


if __name__ == '__main__':
    unittest.main()

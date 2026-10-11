"""Preprocessing script: apply CLAHE to all chest X-ray images."""

import argparse
import logging
import os
import sys

import yaml
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.transforms import ClaheTransform

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main() -> None:
    """Run CLAHE preprocessing on all images."""
    parser = argparse.ArgumentParser(description="Preprocess chest X-ray images with CLAHE")
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to config YAML file')
    parser.add_argument('--dry-run', action='store_true',
                        help='Process only first 100 images for verification')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_config = config['data']
    image_dir = data_config['nih_image_dir']
    output_dir = 'data/processed/images'
    clip_limit = data_config.get('clahe_clip_limit', 2.0)
    tile_size = data_config.get('clahe_tile_size', 8)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize CLAHE transform
    clahe = ClaheTransform(clip_limit=clip_limit, tile_size=tile_size)

    # Get list of images
    if not os.path.exists(image_dir):
        logger.error(f"Image directory not found: {image_dir}")
        logger.info("Please download the NIH ChestX-ray14 dataset and place images in data/raw/images/")
        sys.exit(1)

    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    if args.dry_run:
        image_files = image_files[:100]
        logger.info(f"Dry run: processing first {len(image_files)} images")

    logger.info(f"Found {len(image_files)} images to process")

    # Process images
    processed = 0
    failed = 0

    for filename in tqdm(image_files, desc="Applying CLAHE"):
        input_path = os.path.join(image_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            image = Image.open(input_path).convert('RGB')
            enhanced = clahe(image)
            enhanced.save(output_path)
            processed += 1
        except Exception as e:
            logger.warning(f"Failed to process {filename}: {e}")
            failed += 1

    logger.info(f"Processing complete: {processed} processed, {failed} failed")
    logger.info(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()

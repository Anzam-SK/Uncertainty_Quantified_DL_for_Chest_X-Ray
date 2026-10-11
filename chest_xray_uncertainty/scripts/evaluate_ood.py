"""OOD evaluation script: analyze uncertainty on Covid-X data."""

import argparse
import logging
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import ChestXrayDataset, CovidXDataset
from data.transforms import get_val_transforms
from evaluation.ood_evaluator import OODEvaluator
from models.densenet import MCDropoutDenseNet
from visualization.plots import plot_uncertainty_distribution

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main() -> None:
    """Run OOD uncertainty analysis."""
    parser = argparse.ArgumentParser(description="Evaluate OOD uncertainty on Covid-X data")
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to config YAML file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to MC Dropout model checkpoint (.pth file)')
    parser.add_argument('--mc-passes', type=int, default=None,
                        help='Number of MC Dropout passes (default from config)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save results and plots')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Device — CUDA required
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This project requires a CUDA-capable GPU.")
        sys.exit(1)
    device = torch.device("cuda")
    logger.info(f"Using device: {device}")

    # MC passes
    n_passes = args.mc_passes or config['uncertainty']['mc_passes']
    logger.info(f"MC Dropout passes: {n_passes}")

    # Output directory
    output_dir = args.output_dir or os.path.join(
        config['evaluation']['output_dir'], 'ood_analysis'
    )

    data_config = config['data']
    val_transform = get_val_transforms(data_config['image_size'])

    # Build OOD dataloader (Covid-X)
    ood_dataset = CovidXDataset(
        image_dir=data_config['covidx_image_dir'],
        split_file=data_config['covidx_csv'],
        transform=val_transform,
    )
    ood_loader = DataLoader(
        ood_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Build in-distribution dataloader (subset of NIH test set, same size as OOD)
    indist_dataset = ChestXrayDataset(
        image_dir=data_config['nih_image_dir'],
        labels_csv=data_config['nih_labels_csv'],
        transform=val_transform,
        disease_classes=config['disease_classes'],
        split_file=data_config['nih_test_csv'],
    )
    n_ood = len(ood_dataset)
    n_indist = min(n_ood, len(indist_dataset))
    indices = np.random.choice(len(indist_dataset), n_indist, replace=False)
    indist_subset = Subset(indist_dataset, indices)
    indist_loader = DataLoader(
        indist_subset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Load model
    model_config = config['model']
    model = MCDropoutDenseNet(
        num_classes=model_config['num_classes'],
        dropout_rate=model_config['dropout_rate'],
        pretrained=False,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    logger.info(f"Loaded checkpoint from {args.checkpoint}")

    # Run OOD evaluation
    ood_evaluator = OODEvaluator(model, ood_loader, indist_loader, config, device)
    results = ood_evaluator.run(n_passes=n_passes)

    # Print results
    print(f"\n{'='*60}")
    print(f"OOD UNCERTAINTY ANALYSIS")
    print(f"{'='*60}")
    print(f"OOD mean uncertainty:          {results['ood']['mean_uncertainty_scalar']:.6f}")
    print(f"In-distribution mean uncertainty: {results['in_dist']['mean_uncertainty_scalar']:.6f}")
    print(f"Separation ratio:              {results['separation_ratio']:.2f}x")
    if results['separation_ratio'] > 1.5:
        print("✓ Model produces meaningfully higher uncertainty for OOD data")
    else:
        print("⚠ Separation ratio is low — model may not distinguish OOD well")
    print(f"{'='*60}\n")

    # Generate plot
    os.makedirs(output_dir, exist_ok=True)
    plot_uncertainty_distribution(
        results['ood']['mean_uncertainty'],
        results['in_dist']['mean_uncertainty'],
        output_dir,
    )

    # Save numerical results
    np.save(os.path.join(output_dir, 'ood_uncertainty.npy'),
            results['ood']['mean_uncertainty'])
    np.save(os.path.join(output_dir, 'indist_uncertainty.npy'),
            results['in_dist']['mean_uncertainty'])

    logger.info(f"OOD analysis results saved to {output_dir}")


if __name__ == '__main__':
    main()

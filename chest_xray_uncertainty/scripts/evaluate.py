"""Evaluation script: run test-set evaluation on a trained checkpoint."""

import argparse
import logging
import os
import sys

import torch
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import ChestXrayDataset
from data.transforms import get_val_transforms
from evaluation.evaluator import Evaluator
from models.densenet import MCDropoutDenseNet
from models.resnet_baseline import ResNetBaseline
from torch.utils.data import DataLoader
from visualization.plots import (
    plot_auc_per_class,
    plot_reliability_diagram,
    plot_uncertainty_vs_accuracy,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main() -> None:
    """Run evaluation on a trained model checkpoint."""
    parser = argparse.ArgumentParser(description="Evaluate trained chest X-ray model")
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to config YAML file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--model', type=str, default='densenet',
                        choices=['densenet', 'resnet'],
                        help='Model architecture')
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
    if args.model == 'resnet':
        n_passes = 1  # Baseline uses standard inference
    logger.info(f"MC Dropout passes: {n_passes}")

    # Output directory
    output_dir = args.output_dir or os.path.join(
        config['evaluation']['output_dir'],
        os.path.splitext(os.path.basename(args.checkpoint))[0],
    )

    # Build test dataloader — labels come from Data_Entry_2017.csv,
    # filtered to test_list.txt images only via split_file.
    data_config = config['data']
    test_dataset = ChestXrayDataset(
        image_dir=data_config['nih_image_dir'],
        labels_csv=data_config['nih_labels_csv'],
        transform=get_val_transforms(data_config['image_size']),
        disease_classes=config['disease_classes'],
        split_file=data_config['nih_test_csv'],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Load model
    model_config = config['model']
    if args.model == 'densenet':
        model = MCDropoutDenseNet(
            num_classes=model_config['num_classes'],
            dropout_rate=model_config['dropout_rate'],
            pretrained=False,  # We're loading from checkpoint
        )
    else:
        model = ResNetBaseline(
            num_classes=model_config['num_classes'],
            pretrained=False,
        )

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    logger.info(f"Loaded checkpoint from {args.checkpoint}")

    # Run evaluation
    evaluator = Evaluator(model, test_loader, config, device)
    results = evaluator.run(n_passes=n_passes)

    # Print metrics table
    metrics = results['metrics']
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS — {args.model.upper()}")
    print(f"{'='*60}")
    print(f"\n{'Class':<25} {'AUC':>8}")
    print(f"{'-'*35}")
    for cls, auc in metrics['auc_per_class'].items():
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        print(f"{cls:<25} {auc_str:>8}")
    print(f"{'-'*35}")
    print(f"{'Mean AUC':<25} {metrics['mean_auc']:>8.4f}")
    print(f"{'Macro F1':<25} {metrics['macro_f1']:>8.4f}")
    print(f"{'ECE':<25} {metrics['ece']:>8.4f}")
    print(f"{'NLL':<25} {metrics['nll']:>8.4f}")
    print(f"{'='*60}\n")

    # Save results
    evaluator.save_results(results, output_dir)

    # Generate plots
    logger.info("Generating plots...")
    plot_reliability_diagram(
        results['predictions'], results['labels'],
        args.model.upper(), output_dir,
        n_bins=config['evaluation']['calibration_bins'],
    )
    plot_uncertainty_vs_accuracy(
        results['predictions'], results['labels'],
        results['uncertainty_variance'], output_dir,
    )

    logger.info(f"All results and plots saved to {output_dir}")


if __name__ == '__main__':
    main()

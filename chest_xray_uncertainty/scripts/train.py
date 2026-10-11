"""
Training entry point for chest X-ray uncertainty quantification project.
Trains either the MCDropoutDenseNet or ResNetBaseline model.

Usage:
    python scripts/train.py --model densenet --experiment-name mc_dropout
    python scripts/train.py --model resnet --experiment-name baseline
"""

import argparse
import logging
import os
import sys
import random

import numpy as np
import torch
import yaml

# Add project root to path so imports work regardless of where script is called from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import create_dataloaders
from models.densenet import MCDropoutDenseNet
from models.resnet_baseline import ResNetBaseline
from training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Set all random seeds for full reproducibility across runs.

    Args:
        seed: Integer seed value. Default 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # FIX 2: Force CUDA to use deterministic algorithms.
    # Without these two lines, PyTorch's cuDNN backend uses non-deterministic
    # convolution algorithms for speed. This means two runs of train.py with
    # the same seed will produce slightly different AUC scores — unacceptable
    # for a thesis where you need to report reproducible results.
    # Trade-off: ~5-10% slower training. Worth it for academic reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"All random seeds set to {seed}. cuDNN forced to deterministic mode.")


def compute_pos_weights_from_train_only(
    train_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> torch.Tensor:
    """
    Compute positive class weights STRICTLY from the training subset indices.

    FIX 1 — Data Leak Prevention:
    The naive approach of calling train_loader.dataset.dataset.get_pos_weights()
    grabs the ENTIRE base dataset (including validation samples) to compute class
    imbalance ratios. This means the loss function is secretly informed by the
    validation set's label distribution — a data leak that would be flagged in
    any rigorous thesis defense.

    This function fixes that by slicing the underlying DataFrame using ONLY the
    indices assigned to the training subset, then computing weights from that
    slice alone.

    Args:
        train_loader: DataLoader wrapping a torch.utils.data.Subset of ChestXrayDataset.
        device: Target device for the returned tensor.

    Returns:
        pos_weights: FloatTensor of shape (num_classes,) computed from training data only.
    """
    # train_loader.dataset is a Subset — get its internals
    train_subset = train_loader.dataset          # torch.utils.data.Subset
    full_dataset = train_subset.dataset          # the underlying ChestXrayDataset
    train_indices = train_subset.indices         # indices assigned to training split

    # Slice the label DataFrame to training rows ONLY
    train_df = full_dataset.df.iloc[train_indices].reset_index(drop=True)

    weights = []
    for cls in full_dataset.disease_classes:
        n_pos = train_df[cls].sum()
        n_neg = len(train_df) - n_pos
        # If a class has zero positives in train split, use 1.0 to avoid div/zero
        weights.append(1.0 if n_pos == 0 else n_neg / n_pos)

    pos_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    logger.info(
        f"Train-only pos_weights computed from {len(train_df)} samples | "
        f"min={pos_weights.min():.2f} max={pos_weights.max():.2f}"
    )

    return pos_weights


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train chest X-ray uncertainty model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to YAML config file."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["densenet", "resnet"],
        default="densenet",
        help="Model architecture to train."
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="experiment",
        help="Name used for checkpoint and log subdirectories."
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint .pth file to resume training from."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    return parser.parse_args()


def main() -> None:
    """Main training entry point."""
    args = parse_args()

    # --- Reproducibility (must happen before any torch operations) ---
    set_seed(args.seed)

    # --- Load config ---
    if not os.path.exists(args.config):
        raise FileNotFoundError(
            f"Config file not found: {args.config}. "
            f"Run from the project root directory."
        )
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Inject experiment name into config paths
    config["training"]["checkpoint_dir"] = os.path.join(
        config["training"]["checkpoint_dir"], args.experiment_name
    )
    config["training"]["log_dir"] = os.path.join(
        config["training"]["log_dir"], args.experiment_name
    )
    os.makedirs(config["training"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["training"]["log_dir"], exist_ok=True)

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    # --- Data ---
    logger.info("Building dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config)
    logger.info(
        f"Dataset sizes — "
        f"Train: {len(train_loader.dataset)} | "
        f"Val: {len(val_loader.dataset)} | "
        f"Test: {len(test_loader.dataset)}"
    )

    # --- Positive weights (train split ONLY — no data leak) ---
    pos_weights = compute_pos_weights_from_train_only(train_loader, device)

    # --- Model ---
    logger.info(f"Building model: {args.model}")
    if args.model == "densenet":
        model = MCDropoutDenseNet(
            num_classes=config["model"]["num_classes"],
            dropout_rate=config["model"]["dropout_rate"],
            pretrained=config["model"]["pretrained"]
        )
    elif args.model == "resnet":
        model = ResNetBaseline(
            num_classes=config["model"]["num_classes"],
            pretrained=config["model"]["pretrained"]
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters — Total: {total_params:,} | Trainable: {trainable_params:,}")

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        pos_weights=pos_weights
    )

    # --- Resume from checkpoint if requested ---
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Checkpoint not found: {args.resume}")
        trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed from checkpoint: {args.resume}")

    # --- Train ---
    logger.info(f"Starting training for experiment: '{args.experiment_name}'")
    logger.info(f"Epochs: {config['training']['epochs']} | "
                f"Batch size: {config['training']['batch_size']} | "
                f"LR: {config['training']['learning_rate']}")

    results = trainer.train(
        freeze_epochs=config["model"]["freeze_epochs"]
    )

    # --- Final summary ---
    best_auc = max(epoch["val_auc"] for epoch in results["epochs"])
    best_ece = min(epoch["val_ece"] for epoch in results["epochs"])
    logger.info("=" * 50)
    logger.info("Training complete.")
    logger.info(f"Best Validation AUC : {best_auc:.4f}")
    logger.info(f"Best Validation ECE : {best_ece:.4f}")
    logger.info(
        f"Checkpoint saved to : "
        f"{config['training']['checkpoint_dir']}/best_model.pth"
    )
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
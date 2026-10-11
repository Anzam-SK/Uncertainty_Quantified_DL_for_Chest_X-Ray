"""
Training loop for chest X-ray classification models.
Supports MCDropoutDenseNet (Bayesian) and ResNetBaseline (deterministic).
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.densenet import MCDropoutDenseNet
from training.losses import get_weighted_bce_loss, get_standard_bce_loss
from uncertainty.mc_dropout import mc_predict
from uncertainty.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


class Trainer:
    """
    Handles the full training and validation loop for both model types.

    For MCDropoutDenseNet: validation uses mc_predict() with T stochastic
    passes so that val_ece and val_auc reflect true Bayesian approximation
    outputs — not a deterministic single-pass approximation.

    For ResNetBaseline: validation uses a standard single deterministic pass.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: torch.device,
        pos_weights: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Initialize Trainer.

        Args:
            model: MCDropoutDenseNet or ResNetBaseline instance.
            train_loader: DataLoader for training split.
            val_loader: DataLoader for validation split.
            config: Full config dict loaded from default.yaml.
            device: torch.device — cuda or cpu.
            pos_weights: FloatTensor of shape (num_classes,) for BCE weighting.
                         Must be computed from training data ONLY (no data leak).
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Disease class names for metric reporting
        self.disease_classes: List[str] = config["disease_classes"]

        # --- Loss function ---
        if pos_weights is not None:
            self.loss_fn = get_weighted_bce_loss(pos_weights, device)
            logger.info("Using weighted BCEWithLogitsLoss for class imbalance.")
        else:
            self.loss_fn = get_standard_bce_loss()
            logger.info("Using standard BCEWithLogitsLoss (no pos_weight).")

        # --- Optimizer ---
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )

        # --- LR Scheduler ---
        sched_type = config["training"].get("lr_scheduler_type", "plateau")
        if sched_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config["training"]["lr_step_size"],
                gamma=config["training"]["lr_step_gamma"],
            )
            self._scheduler_type = "step"
            logger.info(
                f"Using StepLR scheduler: step={config['training']['lr_step_size']}, "
                f"gamma={config['training']['lr_step_gamma']}"
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                patience=config["training"]["lr_scheduler_patience"],
                factor=config["training"]["lr_scheduler_factor"],
            )
            self._scheduler_type = "plateau"
            logger.info("Using ReduceLROnPlateau scheduler.")

        # --- Mixed Precision ---
        self.use_amp = config["training"]["mixed_precision"] and device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        if self.use_amp:
            logger.info("Automatic Mixed Precision (AMP) enabled.")

        # --- TensorBoard ---
        log_dir = config["training"]["log_dir"]
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

        # --- Checkpoint directory ---
        self.checkpoint_dir = config["training"]["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # --- Early stopping state ---
        self.best_val_auc: float = 0.0
        self.epochs_without_improvement: int = 0

        # --- MC passes during validation (fewer than final eval for speed) ---
        # Default T=10 for validation — fast but still Bayesian.
        # Full T=50 is used only in evaluate.py for final test-set results.
        self.val_mc_passes: int = (
            config.get("evaluation", {}).get("val_mc_passes", 10)
        )

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Run one full training epoch.

        Args:
            epoch: Current epoch index (0-based).

        Returns:
            Dict with keys: train_loss, train_auc.
        """
        self.model.train()
        running_loss = 0.0
        all_probs: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1} [Train]",
            leave=False
        )

        for images, labels, _ in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with optional AMP
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.use_amp
            ):
                logits = self.model(images)
                loss = self.loss_fn(logits, labels)

            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Track metrics
            running_loss += loss.item() * images.size(0)
            with torch.no_grad():
                probs = torch.sigmoid(logits)
            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_loss = running_loss / len(self.train_loader.dataset)
        all_probs_np = np.concatenate(all_probs, axis=0)
        all_labels_np = np.concatenate(all_labels, axis=0)

        metrics = compute_all_metrics(
            all_probs_np, all_labels_np,
            self.disease_classes,
            n_bins=self.config["evaluation"]["calibration_bins"]
        )

        # Log to TensorBoard
        self.writer.add_scalar("Loss/train", epoch_loss, epoch)
        self.writer.add_scalar("AUC/train", metrics["mean_auc"], epoch)
        self.writer.add_scalar("ECE/train", metrics["ece"], epoch)
        self.writer.add_scalar(
            "LR",
            self.optimizer.param_groups[0]["lr"],
            epoch
        )

        logger.info(
            f"Epoch {epoch + 1} [Train] "
            f"Loss: {epoch_loss:.4f} | "
            f"AUC: {metrics['mean_auc']:.4f} | "
            f"ECE: {metrics['ece']:.4f}"
        )

        # Print to stdout for immediate visibility
        print(
            f"Epoch {epoch + 1} [Train] "
            f"Loss: {epoch_loss:.4f} | "
            f"AUC: {metrics['mean_auc']:.4f} | "
            f"ECE: {metrics['ece']:.4f}"
        )

        return {
            "train_loss": epoch_loss,
            "train_auc": metrics["mean_auc"],
            "train_ece": metrics["ece"],
        }

    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate the model using inference appropriate to the model type.

        CRITICAL DESIGN DECISION:
        - MCDropoutDenseNet: uses mc_predict() with T=val_mc_passes stochastic
          forward passes. This is essential — calling model.eval() alone disables
          dropout and produces deterministic outputs, which would make val_ece
          identical to a standard DenseNet. The entire thesis claim (MC Dropout
          improves calibration) would be invisible in the training curves.
        - ResNetBaseline: uses a single deterministic forward pass (model.eval()).

        BCE loss is computed from a single deterministic pass in both cases
        (for consistency and speed). Only the probability estimates used for
        AUC and ECE differ between model types.

        Args:
            epoch: Current epoch index (0-based).

        Returns:
            Dict with keys: val_loss, val_auc, val_ece.
        """
        # Always start in eval mode (disables batch norm running stats update,
        # disables training-mode dropout for loss calculation)
        self.model.eval()

        running_loss = 0.0
        all_probs: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []

        is_mc_model = isinstance(self.model, MCDropoutDenseNet)

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch + 1} [Val]",
            leave=False
        )

        with torch.no_grad():
            for images, labels, _ in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # --- Step 1: Deterministic pass for BCE loss tracking ---
                # model.eval() is already set above — no dropout active here.
                # This gives a stable, comparable loss number across all epochs.
                self.model.eval()
                logits = self.model(images)
                loss = self.loss_fn(logits, labels)
                running_loss += loss.item() * images.size(0)

                # --- Step 2: Get probabilities for AUC & ECE ---
                if is_mc_model:
                    # THE FIX: use mc_predict() so that ECE reflects the
                    # Bayesian approximation, not a plain deterministic DenseNet.
                    # T=val_mc_passes (default 10) keeps validation fast.
                    # Final evaluation uses T=50 (in evaluate.py).
                    mean_probs, _ = mc_predict(
                        model=self.model,
                        image_batch=images,
                        n_passes=self.val_mc_passes,
                        device=self.device,
                    )
                    probs = mean_probs  # shape (batch, 14) — Bayesian mean
                else:
                    # Baseline: standard single-pass sigmoid probabilities
                    probs = torch.sigmoid(logits)

                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        # --- Compute epoch-level metrics ---
        epoch_loss = running_loss / len(self.val_loader.dataset)
        all_probs_np = np.concatenate(all_probs, axis=0)
        all_labels_np = np.concatenate(all_labels, axis=0)

        metrics = compute_all_metrics(
            all_probs_np,
            all_labels_np,
            self.disease_classes,
            n_bins=self.config["evaluation"]["calibration_bins"]
        )

        # --- Log to TensorBoard ---
        self.writer.add_scalar("Loss/val", epoch_loss, epoch)
        self.writer.add_scalar("AUC/val", metrics["mean_auc"], epoch)
        self.writer.add_scalar("ECE/val", metrics["ece"], epoch)

        inference_mode = f"MC T={self.val_mc_passes}" if is_mc_model else "Deterministic"
        logger.info(
            f"Epoch {epoch + 1} [Val | {inference_mode}] "
            f"Loss: {epoch_loss:.4f} | "
            f"AUC: {metrics['mean_auc']:.4f} | "
            f"ECE: {metrics['ece']:.4f}"
        )

        # Print to stdout for immediate visibility
        print(
            f"Epoch {epoch + 1} [Val | {inference_mode}] "
            f"Loss: {epoch_loss:.4f} | "
            f"AUC: {metrics['mean_auc']:.4f} | "
            f"ECE: {metrics['ece']:.4f}"
        )

        return {
            "val_loss": epoch_loss,
            "val_auc": metrics["mean_auc"],
            "val_ece": metrics["ece"],
        }

    def train(self, freeze_epochs: int = 5) -> Dict:
        """
        Full training loop with backbone freezing, early stopping, and
        checkpoint saving.

        Args:
            freeze_epochs: Number of epochs to train with backbone frozen.
                           After this, all layers are unfrozen for fine-tuning.

        Returns:
            Dict with key 'epochs': list of per-epoch metric dicts.
        """
        total_epochs = self.config["training"]["epochs"]
        patience = self.config["training"]["early_stopping_patience"]
        all_epoch_results = []

        logger.info(
            f"Starting training: {total_epochs} epochs | "
            f"Backbone frozen for first {freeze_epochs} epochs."
        )

        for epoch in range(total_epochs):

            # --- Backbone freeze / unfreeze schedule ---
            if epoch == 0:
                self.model.freeze_backbone()
                logger.info(
                    "Backbone FROZEN — training classifier head only."
                )
            elif epoch == freeze_epochs:
                self.model.unfreeze_backbone()
                logger.info(
                    f"Epoch {epoch + 1}: Backbone UNFROZEN — full fine-tuning."
                )

            # --- Train and validate ---
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate(epoch)

            # --- LR scheduler step ---
            if self._scheduler_type == "step":
                self.scheduler.step()
            else:
                self.scheduler.step(val_metrics["val_auc"])

            # --- Early stopping and checkpoint ---
            if val_metrics["val_auc"] > self.best_val_auc:
                self.best_val_auc = val_metrics["val_auc"]
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch, val_metrics["val_auc"])
                logger.info(
                    f"  New best val_auc: {self.best_val_auc:.4f} — checkpoint saved."
                )
            else:
                self.epochs_without_improvement += 1
                logger.info(
                    f"  No improvement for {self.epochs_without_improvement} "
                    f"epoch(s). Best: {self.best_val_auc:.4f}"
                )

            if self.epochs_without_improvement >= patience:
                logger.info(
                    f"Early stopping triggered at epoch {epoch + 1}. "
                    f"Best val_auc: {self.best_val_auc:.4f}"
                )
                break

            # Combine and store epoch results
            epoch_result = {**train_metrics, **val_metrics, "epoch": epoch + 1}
            all_epoch_results.append(epoch_result)

        self.writer.close()
        return {"epochs": all_epoch_results}

    def save_checkpoint(self, epoch: int, val_auc: float) -> None:
        """
        Save model and optimizer state to disk.

        Args:
            epoch: Current epoch index (0-based).
            val_auc: Validation AUC at this checkpoint (for record-keeping).
        """
        checkpoint_path = os.path.join(
            self.checkpoint_dir, "best_model.pth"
        )
        torch.save(
            {
                "epoch": epoch,
                "val_auc": val_auc,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
            },
            checkpoint_path,
        )
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_weights_only(self, path: str) -> None:
        """Load model weights only — optimizer and epoch reset to 0 (transfer learning).

        Use this when scaling to a larger dataset so the optimizer starts fresh
        but the model begins from a well-trained feature representation.

        Args:
            path: Path to .pth checkpoint file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        # Intentionally do NOT restore optimizer state or best_val_auc
        logger.info(
            f"Loaded weights from epoch {checkpoint.get('epoch', '?')} "
            f"(val_auc={checkpoint.get('val_auc', 0.0):.4f}) — "
            f"optimizer reset, training from epoch 0."
        )

    def load_checkpoint(self, path: str) -> None:
        """
        Load model and optimizer state from a checkpoint file.

        Args:
            path: Path to .pth checkpoint file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_auc = checkpoint.get("val_auc", 0.0)

        logger.info(
            f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')} "
            f"with val_auc={self.best_val_auc:.4f}"
        )
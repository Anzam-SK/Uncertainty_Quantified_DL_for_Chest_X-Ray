"""
Single-Image MC Dropout Inference
==================================
Runs T=50 stochastic forward passes on one chest X-ray image and reports:

Terminal output:
  - Top-3 predicted diseases with mean probability and variance
  - Entropy-based uncertainty score + level (LOW / MEDIUM / HIGH)

Saved figure (<output-dir>/inference_<filename>.png):
  - Left panel  : original X-ray image
  - Right panel : all 14 disease probabilities with ±1 std error bars,
                  top-3 highlighted in red

Usage:
  python scripts/single_inference.py \
      --image "data/NIH ChestX-ray14/images/00000001_000.png" \
      --checkpoint experiments/checkpoints/thesis_112k_ultimate_run/best_model.pth
"""

import argparse
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import _build_image_index
from data.transforms import get_val_transforms
from models.densenet import MCDropoutDenseNet
from uncertainty.mc_dropout import compute_predictive_entropy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-image MC Dropout inference")
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to chest X-ray image (filename-only lookup if path does not exist)",
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="experiments/checkpoints/thesis_112k_ultimate_run/best_model.pth",
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "--config", type=str, default="config/default.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--mc-passes", type=int, default=50,
        help="Number of MC Dropout passes (default: 50)",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="experiments/results/single_inference",
        help="Directory to save the output figure",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Image path resolution
# ---------------------------------------------------------------------------
def resolve_image_path(image_arg: str, nih_image_dir: str) -> str:
    """Return the real path to the image.

    If the user-supplied path exists directly, use it.
    Otherwise search recursively inside nih_image_dir by filename.
    """
    if os.path.exists(image_arg):
        return image_arg

    filename = os.path.basename(image_arg)
    logger.info(f"Path '{image_arg}' not found directly — searching for '{filename}' ...")
    index = _build_image_index(nih_image_dir)

    if filename in index:
        resolved = index[filename]
        logger.info(f"Found at: {resolved}")
        return resolved

    logger.error(
        f"Image '{filename}' not found under '{nih_image_dir}'. "
        f"Check the filename and nih_image_dir in config."
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference(
    model: MCDropoutDenseNet,
    image_tensor: torch.Tensor,
    n_passes: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Run T MC Dropout passes on a single image tensor (1, 3, H, W).

    Returns:
        mean_probs   : (14,)  mean sigmoid probability per class
        std_probs    : (14,)  std dev across passes
        variance     : (14,)  variance across passes
        entropy      : scalar entropy value
    """
    model.eval()
    model.enable_mc_dropout()

    image_tensor = image_tensor.unsqueeze(0).to(device)   # (1, 3, H, W)
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for _ in range(n_passes):
            logits = model(image_tensor)
            probs  = torch.sigmoid(logits)                 # (1, 14)
            all_probs.append(probs.cpu().numpy()[0])       # (14,)

    stacked  = np.stack(all_probs, axis=0)                 # (T, 14)
    mean_p   = stacked.mean(axis=0)                        # (14,)
    std_p    = stacked.std(axis=0)                         # (14,)
    var_p    = stacked.var(axis=0)                         # (14,)

    entropy_val = float(
        compute_predictive_entropy(
            torch.tensor(mean_p).unsqueeze(0)
        ).item()
    )

    return mean_p, std_p, var_p, entropy_val


# ---------------------------------------------------------------------------
# Uncertainty level classifier
# ---------------------------------------------------------------------------
def classify_entropy(entropy: float) -> tuple[str, str]:
    """Return (level, colour) for terminal display."""
    if entropy < 1.5:
        return "LOW",    "\033[92m"    # green
    elif entropy < 2.5:
        return "MEDIUM", "\033[93m"    # yellow
    else:
        return "HIGH",   "\033[91m"    # red


# ---------------------------------------------------------------------------
# Terminal report
# ---------------------------------------------------------------------------
def print_terminal_report(
    filename: str,
    top3: list[tuple[str, float, float]],
    entropy: float,
    n_passes: int,
) -> None:
    level, colour = classify_entropy(entropy)
    reset = "\033[0m"
    sep   = "=" * 62

    print(f"\n{sep}")
    print(f"  SINGLE IMAGE INFERENCE  —  MCDropout-DenseNet121 (T={n_passes})")
    print(sep)
    print(f"  Image : {filename}")
    print()
    print(f"  TOP-3 PREDICTED CONDITIONS")
    print(f"  {'Rank':<6} {'Disease':<25} {'P(mean)':>8}  {'Variance':>10}")
    print(f"  {'-'*55}")
    for rank, (name, prob, var) in enumerate(top3, 1):
        print(f"  #{rank:<5} {name:<25} {prob:>8.4f}  {var:>10.6f}")
    print()
    print(f"  Uncertainty (Entropy) : {colour}{entropy:.4f}  [{level}]{reset}")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def save_figure(
    raw_image: Image.Image,
    mean_probs: np.ndarray,
    std_probs: np.ndarray,
    disease_classes: list[str],
    entropy: float,
    filename: str,
    output_dir: str,
) -> str:
    """Save a two-panel figure: X-ray on left, prediction chart on right."""

    # Sort all classes by mean probability descending for the bar chart
    sort_idx      = np.argsort(mean_probs)[::-1]
    sorted_names  = [disease_classes[i] for i in sort_idx]
    sorted_means  = mean_probs[sort_idx]
    sorted_stds   = std_probs[sort_idx]

    # Identify top-3 indices (in the sorted order)
    top3_set = set(range(3))

    bar_colors = [
        "#e74c3c" if i in top3_set else "#aab4be"
        for i in range(len(sorted_names))
    ]

    # ---- Figure layout ----
    fig = plt.figure(figsize=(16, 7), constrained_layout=True)
    fig.suptitle(
        f"MC Dropout Inference — {filename}   |   Entropy = {entropy:.4f}",
        fontsize=13, fontweight="bold",
    )

    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.6])

    # ---- Left panel: X-ray ----
    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(raw_image.convert("L"), cmap="gray", aspect="auto")
    ax_img.set_title("Input X-Ray", fontsize=11)
    ax_img.axis("off")

    # ---- Right panel: prediction bar chart ----
    ax_bar = fig.add_subplot(gs[1])

    y_pos = np.arange(len(sorted_names))
    bars  = ax_bar.barh(
        y_pos, sorted_means,
        xerr=sorted_stds,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.6,
        error_kw=dict(ecolor="#555", elinewidth=1.2, capsize=3),
        height=0.65,
        zorder=3,
    )

    # Annotate each bar with mean ± std
    for i, (mean_v, std_v) in enumerate(zip(sorted_means, sorted_stds)):
        ax_bar.text(
            mean_v + sorted_stds.max() * 0.02,
            i,
            f"{mean_v:.3f} ±{std_v:.3f}",
            va="center", fontsize=8,
            color="#222" if i not in top3_set else "#c0392b",
        )

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(sorted_names, fontsize=9)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, 1.08)
    ax_bar.set_xlabel("Mean Predicted Probability (±1 std over 50 passes)", fontsize=10)
    ax_bar.set_title(
        "Prediction Probabilities with MC Dropout Variance\n"
        "(red = Top-3, error bars = uncertainty)",
        fontsize=10,
    )
    ax_bar.axvline(0.5, color="gray", linestyle=":", alpha=0.6, linewidth=1.2)
    ax_bar.grid(axis="x", alpha=0.3, zorder=0)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # Legend
    ax_bar.legend(
        handles=[
            mpatches.Patch(color="#e74c3c", label="Top-3 predictions"),
            mpatches.Patch(color="#aab4be", label="Other classes"),
        ],
        fontsize=9, loc="lower right",
    )

    # ---- Save ----
    os.makedirs(output_dir, exist_ok=True)
    stem     = os.path.splitext(filename)[0]
    out_path = os.path.join(output_dir, f"inference_{stem}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # --- CUDA required ---
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This project requires a CUDA-capable GPU.")
        sys.exit(1)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    # --- Config ---
    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_config     = config["data"]
    model_config    = config["model"]
    disease_classes = config["disease_classes"]

    # --- Resolve image path ---
    image_path = resolve_image_path(args.image, data_config["nih_image_dir"])
    filename   = os.path.basename(image_path)

    # --- Load raw image (for display) and transformed tensor (for inference) ---
    raw_image      = Image.open(image_path).convert("RGB")
    transform      = get_val_transforms(data_config["image_size"])
    image_tensor   = transform(raw_image)                  # (3, 224, 224)

    # --- Load model ---
    model = MCDropoutDenseNet(
        num_classes=model_config["num_classes"],
        dropout_rate=model_config["dropout_rate"],
        pretrained=False,
    )
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    logger.info(
        f"Loaded checkpoint (epoch={ckpt.get('epoch','?')}, "
        f"val_auc={ckpt.get('val_auc', 0):.4f})"
    )

    # --- Inference ---
    logger.info(f"Running {args.mc_passes} MC Dropout passes on {filename} ...")
    mean_probs, std_probs, variance, entropy = run_inference(
        model, image_tensor, args.mc_passes, device
    )

    # --- Top-3 ---
    top3_idx = np.argsort(mean_probs)[::-1][:3]
    top3     = [
        (disease_classes[i], float(mean_probs[i]), float(variance[i]))
        for i in top3_idx
    ]

    # --- Terminal report ---
    print_terminal_report(filename, top3, entropy, args.mc_passes)

    # --- Figure ---
    out_path = save_figure(
        raw_image, mean_probs, std_probs,
        disease_classes, entropy, filename, args.output_dir,
    )
    print(f"  Figure saved: {out_path}\n")


if __name__ == "__main__":
    main()

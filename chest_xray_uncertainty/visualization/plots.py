"""Visualization functions for uncertainty analysis and model evaluation."""

import logging
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def plot_reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    output_dir: str,
    n_bins: int = 15,
    dpi: int = 150,
) -> Figure:
    """Plot reliability diagram (calibration curve).

    X-axis: Mean predicted confidence per bin.
    Y-axis: Actual accuracy per bin.
    Includes perfect calibration diagonal (y=x).

    Args:
        probs: Predicted probabilities, shape (N, num_classes).
        labels: Ground truth labels, shape (N, num_classes).
        model_name: Name of the model for the title.
        output_dir: Directory to save the figure.
        n_bins: Number of bins.

    Returns:
        matplotlib Figure object.
    """
    probs_flat = probs.flatten()
    labels_flat = labels.flatten()

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = []
    bin_accuracies = []
    bin_confidences = []

    for i in range(n_bins):
        in_bin = (probs_flat >= bin_edges[i]) & (probs_flat < bin_edges[i + 1])
        if in_bin.sum() == 0:
            continue
        bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
        bin_accuracies.append(labels_flat[in_bin].mean())
        bin_confidences.append(probs_flat[in_bin].mean())

    # Compute ECE for title
    from uncertainty.metrics import compute_ece
    ece = compute_ece(probs, labels, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', alpha=0.5)
    ax.plot(bin_confidences, bin_accuracies, 'bo-', label=f'{model_name}')
    ax.set_xlabel('Mean Predicted Confidence', fontsize=12)
    ax.set_ylabel('Actual Accuracy', fontsize=12)
    ax.set_title(f'Reliability Diagram — {model_name}\nECE = {ece:.4f}', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'reliability_diagram.png'), dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    return fig


def plot_uncertainty_distribution(
    ood_uncertainty: np.ndarray,
    indist_uncertainty: np.ndarray,
    output_dir: str,
) -> Figure:
    """Plot overlapping histograms of OOD vs in-distribution uncertainty.

    Args:
        ood_uncertainty: OOD uncertainty values, shape (N_ood,).
        indist_uncertainty: In-distribution uncertainty values, shape (N_indist,).
        output_dir: Directory to save the figure.

    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(indist_uncertainty, bins=50, alpha=0.6, color='blue',
            label='In-Distribution (NIH)', density=True)
    ax.hist(ood_uncertainty, bins=50, alpha=0.6, color='red',
            label='OOD (Covid-X)', density=True)

    # Vertical lines at means
    ax.axvline(np.mean(indist_uncertainty), color='blue', linestyle='--',
               linewidth=2, label=f'In-Dist Mean: {np.mean(indist_uncertainty):.4f}')
    ax.axvline(np.mean(ood_uncertainty), color='red', linestyle='--',
               linewidth=2, label=f'OOD Mean: {np.mean(ood_uncertainty):.4f}')

    # Separation ratio
    sep_ratio = np.mean(ood_uncertainty) / max(np.mean(indist_uncertainty), 1e-10)

    ax.set_xlabel('Mean Uncertainty (Variance)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Uncertainty Distribution — Separation Ratio: {sep_ratio:.2f}x',
                 fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'uncertainty_distribution.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    return fig


def plot_auc_per_class(
    auc_dict_baseline: Dict[str, Optional[float]],
    auc_dict_mc: Dict[str, Optional[float]],
    class_names: List[str],
    output_dir: str,
) -> Figure:
    """Plot horizontal bar chart comparing baseline vs MC Dropout AUC per class.

    Args:
        auc_dict_baseline: AUC per class for baseline model.
        auc_dict_mc: AUC per class for MC Dropout model.
        class_names: List of disease class names.
        output_dir: Directory to save the figure.

    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    y_pos = np.arange(len(class_names))
    bar_height = 0.35

    baseline_aucs = [auc_dict_baseline.get(c, 0) or 0 for c in class_names]
    mc_aucs = [auc_dict_mc.get(c, 0) or 0 for c in class_names]

    # Color bars: green if MC >= baseline, orange if MC < baseline
    colors = ['green' if mc >= base else 'orange'
              for mc, base in zip(mc_aucs, baseline_aucs)]

    bars1 = ax.barh(y_pos - bar_height/2, baseline_aucs, bar_height,
                    label='Baseline (ResNet-50)', color='steelblue', alpha=0.8)
    bars2 = ax.barh(y_pos + bar_height/2, mc_aucs, bar_height,
                    label='MC Dropout (DenseNet-121)', color=colors, alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_xlabel('ROC AUC', fontsize=12)
    ax.set_title('AUC per Disease Class — Baseline vs MC Dropout', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1.05)
    ax.grid(True, axis='x', alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'auc_per_class.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    return fig


def plot_uncertainty_vs_accuracy(
    mean_probs: np.ndarray,
    labels: np.ndarray,
    uncertainty: np.ndarray,
    output_dir: str,
) -> Figure:
    """Plot scatter of uncertainty vs per-sample accuracy.

    Should show negative correlation: high uncertainty → lower accuracy.

    Args:
        mean_probs: Mean predicted probabilities, shape (N, num_classes).
        labels: Ground truth labels, shape (N, num_classes).
        uncertainty: Uncertainty variance, shape (N, num_classes).
        output_dir: Directory to save the figure.

    Returns:
        matplotlib Figure object.
    """
    # Per-sample mean uncertainty
    mean_uncertainty = uncertainty.mean(axis=1)

    # Per-sample accuracy (fraction of correct binary predictions)
    predictions = (mean_probs > 0.5).astype(float)
    per_sample_accuracy = (predictions == labels).mean(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(mean_uncertainty, per_sample_accuracy, alpha=0.3, s=10)

    # Trend line
    z = np.polyfit(mean_uncertainty, per_sample_accuracy, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(mean_uncertainty.min(), mean_uncertainty.max(), 100)
    ax.plot(x_trend, p(x_trend), 'r-', linewidth=2,
            label=f'Trend (slope={z[0]:.2f})')

    ax.set_xlabel('Mean Uncertainty (Variance)', fontsize=12)
    ax.set_ylabel('Per-Sample Accuracy', fontsize=12)
    ax.set_title('Uncertainty vs Accuracy', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'uncertainty_vs_accuracy.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    return fig


def plot_training_curves(
    log_dir: str,
    output_dir: str,
) -> Optional[Figure]:
    """Plot training curves from TensorBoard event files.

    Args:
        log_dir: Directory containing TensorBoard event files.
        output_dir: Directory to save the figure.

    Returns:
        matplotlib Figure object, or None if no event files found.
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        logger.warning("TensorBoard not available. Skipping training curves plot.")
        return None

    # Find event files
    event_files = []
    for root, _, files in os.walk(log_dir):
        for f in files:
            if f.startswith('events.out.tfevents'):
                event_files.append(os.path.join(root, f))

    if not event_files:
        logger.warning(f"No TensorBoard event files found in {log_dir}")
        return None

    # Load first event file
    ea = event_accumulator.EventAccumulator(event_files[0])
    ea.Reload()

    # Extract scalar data
    tags = ea.Tags().get('scalars', [])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot AUC
    for tag in tags:
        if 'AUC' in tag:
            data = ea.Scalars(tag)
            steps = [d.step for d in data]
            values = [d.value for d in data]
            ax1.plot(steps, values, label=tag)

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('AUC', fontsize=12)
    ax1.set_title('AUC over Training', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot Loss
    for tag in tags:
        if 'Loss' in tag:
            data = ea.Scalars(tag)
            steps = [d.step for d in data]
            values = [d.value for d in data]
            ax2.plot(steps, values, label=tag)

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Loss over Training', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    return fig


def plot_uncertainty_deciles(
    mean_probs: np.ndarray,
    labels: np.ndarray,
    uncertainty: np.ndarray,
    output_dir: str,
    dpi: int = 300,
) -> Figure:
    """Plot mean accuracy grouped by uncertainty decile.

    Samples are sorted by per-sample mean MC Dropout variance and split into
    10 equal deciles. Each bar shows the mean multi-label accuracy in that
    decile. A well-calibrated Bayesian model should show declining accuracy
    as uncertainty increases (decile 1=lowest uncertainty, 10=highest).

    Args:
        mean_probs: MC Dropout mean probabilities, shape (N, num_classes).
        labels: Ground truth binary labels, shape (N, num_classes).
        uncertainty: MC Dropout variance, shape (N, num_classes).
        output_dir: Directory to save the figure.
        dpi: Resolution for saved figure.

    Returns:
        matplotlib Figure object.
    """
    # Per-sample mean uncertainty and accuracy
    mean_unc = uncertainty.mean(axis=1)                           # (N,)
    preds    = (mean_probs >= 0.5).astype(float)
    acc      = (preds == labels).mean(axis=1)                    # (N,)

    # Sort by uncertainty, split into 10 deciles
    sorted_idx  = np.argsort(mean_unc)
    n           = len(sorted_idx)
    decile_size = n // 10

    decile_labels    = []
    decile_accs      = []
    decile_mean_uncs = []

    for i in range(10):
        start = i * decile_size
        end   = (i + 1) * decile_size if i < 9 else n
        idx   = sorted_idx[start:end]
        decile_accs.append(acc[idx].mean())
        decile_mean_uncs.append(mean_unc[idx].mean())
        decile_labels.append(f"D{i+1}")

    # Color bars by decile: low uncertainty = green, high = red
    cmap   = plt.cm.RdYlGn_r
    colors = [cmap(i / 9) for i in range(10)]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(decile_labels, decile_accs, color=colors, edgecolor='white',
                  linewidth=0.8, zorder=3)

    # Annotate bars with accuracy value
    for bar, val in zip(bars, decile_accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.3f}", ha='center', va='bottom', fontsize=9)

    # Trend line
    x_vals = np.arange(10)
    z      = np.polyfit(x_vals, decile_accs, 1)
    p      = np.poly1d(z)
    ax.plot(decile_labels, p(x_vals), 'k--', linewidth=1.8,
            label=f'Trend (slope={z[0]:+.4f})', zorder=4)

    ax.set_xlabel('Uncertainty Decile  (D1 = lowest, D10 = highest)', fontsize=12)
    ax.set_ylabel('Mean Multi-Label Accuracy', fontsize=12)
    ax.set_title(
        'Uncertainty–Accuracy Trade-off by Decile\n'
        'Accuracy should decrease as uncertainty increases',
        fontsize=13,
    )
    ax.set_ylim(0, min(1.05, max(decile_accs) + 0.06))
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.35, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'uncertainty_accuracy_deciles.png'),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_uncertainty_heatmap(
    uncertainty: np.ndarray,
    disease_classes: List[str],
    output_dir: str,
    dpi: int = 300,
) -> Figure:
    """Plot per-disease mean MC Dropout variance as a sorted heatmap.

    Shows which disease classes the model is most uncertain about,
    sorted from highest to lowest mean variance.

    Args:
        uncertainty: MC Dropout variance, shape (N, num_classes).
        disease_classes: Ordered list of disease class names.
        output_dir: Directory to save the figure.
        dpi: Resolution for saved figure.

    Returns:
        matplotlib Figure object.
    """
    mean_var_per_class = uncertainty.mean(axis=0)   # (num_classes,)

    # Sort descending
    sort_idx      = np.argsort(mean_var_per_class)[::-1]
    sorted_names  = [disease_classes[i] for i in sort_idx]
    sorted_values = mean_var_per_class[sort_idx]

    # Normalise for colour mapping
    norm   = plt.Normalize(vmin=sorted_values.min(), vmax=sorted_values.max())
    cmap   = plt.cm.YlOrRd
    colors = cmap(norm(sorted_values))

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(sorted_names, sorted_values, color=colors,
                   edgecolor='white', linewidth=0.6)

    # Annotate bars
    for bar, val in zip(bars, sorted_values):
        ax.text(val + sorted_values.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.5f}", va='center', fontsize=9)

    # Colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Mean MC Dropout Variance', fontsize=10)

    ax.set_xlabel('Mean MC Dropout Variance (T=50)', fontsize=12)
    ax.set_title(
        'Per-Disease Uncertainty Heatmap\n'
        'Higher variance → model less confident for that pathology',
        fontsize=13,
    )
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'uncertainty_heatmap.png'),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_entropy_histogram(
    entropy: np.ndarray,
    output_dir: str,
    dpi: int = 300,
) -> Figure:
    """Plot histogram of predictive entropy over the NIH test set.

    High entropy = high overall prediction uncertainty across all classes.
    The distribution shape reveals how often the model is confident vs uncertain.

    Args:
        entropy: Per-sample predictive entropy, shape (N,).
        output_dir: Directory to save the figure.
        dpi: Resolution for saved figure.

    Returns:
        matplotlib Figure object.
    """
    mean_e  = float(np.mean(entropy))
    std_e   = float(np.std(entropy))
    p25, p75, p95 = np.percentile(entropy, [25, 75, 95])

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram
    n, bins, patches = ax.hist(entropy, bins=60, density=True,
                                color='steelblue', alpha=0.75,
                                edgecolor='white', linewidth=0.4,
                                label='Entropy (density)', zorder=2)

    # KDE overlay
    try:
        from scipy.stats import gaussian_kde
        kde   = gaussian_kde(entropy, bw_method='scott')
        x_kde = np.linspace(entropy.min(), entropy.max(), 300)
        ax.plot(x_kde, kde(x_kde), color='navy', linewidth=2.2,
                label='KDE', zorder=3)
    except ImportError:
        pass  # scipy optional

    # Vertical reference lines
    ax.axvline(mean_e, color='red',    linestyle='--', linewidth=1.8,
               label=f'Mean={mean_e:.3f}', zorder=4)
    ax.axvline(mean_e + std_e, color='orange', linestyle=':',  linewidth=1.5,
               label=f'Mean+1σ={mean_e+std_e:.3f}', zorder=4)
    ax.axvline(p95, color='purple',   linestyle='-.', linewidth=1.5,
               label=f'95th pct={p95:.3f}', zorder=4)

    # Stats annotation box
    stats_text = (
        f"N = {len(entropy):,}\n"
        f"Mean = {mean_e:.4f}\n"
        f"Std  = {std_e:.4f}\n"
        f"P25  = {p25:.4f}\n"
        f"P75  = {p75:.4f}\n"
        f"P95  = {p95:.4f}"
    )
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
            fontsize=9, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                      edgecolor='gray', alpha=0.9))

    ax.set_xlabel('Predictive Entropy  (−Σ p·log p)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(
        'Predictive Entropy Distribution — NIH Test Set (T=50 MC passes)\n'
        'Low entropy = confident predictions | High entropy = uncertain',
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'entropy_histogram.png'),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return fig

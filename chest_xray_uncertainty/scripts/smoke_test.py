"""
Smoke test: verifies the full pipeline on synthetic data.
No real dataset needed. Should complete in under 2 minutes on GPU.
"""
import torch
import numpy as np
from models.densenet import MCDropoutDenseNet
from models.resnet_baseline import ResNetBaseline
from uncertainty.mc_dropout import mc_predict, compute_predictive_entropy
from uncertainty.metrics import compute_ece, compute_nll, compute_auc_per_class
import yaml
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    # Load config
    with open("config/default.yaml") as f:
        config = yaml.safe_load(f)

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This project requires a CUDA-capable GPU.")
        sys.exit(1)
    device = torch.device("cuda")
    print(f"\n{'='*50}")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"{'='*50}\n")

    # Synthetic batch: 8 images, 3 channels, 224x224
    dummy_batch = torch.randn(8, 3, 224, 224).to(device)

    # Test MCDropoutDenseNet
    print("[1/5] Testing MCDropoutDenseNet forward pass...")
    model = MCDropoutDenseNet(num_classes=14, dropout_rate=0.3, pretrained=False).to(device)
    with torch.no_grad():
        out = model(dummy_batch)
    assert out.shape == (8, 14), f"Expected (8,14), got {out.shape}"
    print(f"      Output shape: {out.shape} \u2713")

    # Test MC predict
    print("[2/5] Testing MC Dropout inference (T=10 passes)...")
    mean_probs, uncertainty = mc_predict(model, dummy_batch, n_passes=10, device=device)
    assert mean_probs.shape == (8, 14)
    assert uncertainty.shape == (8, 14)
    assert (uncertainty >= 0).all()
    print(f"      Mean probs shape: {mean_probs.shape} \u2713")
    print(f"      Uncertainty range: [{uncertainty.min():.4f}, {uncertainty.max():.4f}] \u2713")

    # Test stochasticity
    print("[3/5] Verifying MC Dropout stochasticity...")
    model.eval()
    model.enable_mc_dropout()
    with torch.no_grad():
        out1 = torch.sigmoid(model(dummy_batch[:1]))
        out2 = torch.sigmoid(model(dummy_batch[:1]))
    assert not torch.allclose(out1, out2), "MC Dropout outputs must differ!"
    print(f"      Outputs differ across passes \u2713")

    # Test metrics
    print("[4/5] Testing metrics functions...")
    probs_np = mean_probs.cpu().numpy()
    labels_np = (np.random.rand(8, 14) > 0.5).astype(float)
    ece = compute_ece(probs_np, labels_np, n_bins=5)
    nll = compute_nll(probs_np, labels_np)
    assert 0 <= ece <= 1
    assert np.isfinite(nll)
    print(f"      ECE: {ece:.4f} \u2713   NLL: {nll:.4f} \u2713")

    # Test ResNet baseline
    print("[5/5] Testing ResNetBaseline forward pass...")
    baseline = ResNetBaseline(num_classes=14, pretrained=False).to(device)
    with torch.no_grad():
        out_b = baseline(dummy_batch)
    assert out_b.shape == (8, 14)
    print(f"      Output shape: {out_b.shape} \u2713")

    print(f"\n{'='*50}")
    print("ALL SMOKE TESTS PASSED \u2713")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()

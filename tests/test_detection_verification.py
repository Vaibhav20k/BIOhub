"""Phase 2 regression tests: Deterministic verification of corrected detector components."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.detection.heatmap import generate_gaussian_heatmap_3d
from src.detection.peak_detector import PeakDetector3D
from src.detection.unet3d import TemporalUNet3D
from src.training.losses import ModifiedFocalLoss


def test_modified_focal_loss_corrected_gradient():
    """Verify that normalizing by num_pos restores background gradient to > 1e-3 at pred=0.25."""
    shape = (32, 128, 128)
    scale = (1.625, 0.40625, 0.40625)
    center = (16.0, 64.0, 64.0)

    target_np = generate_gaussian_heatmap_3d(shape, [center], scale=scale, sigma_um=1.5)
    target = torch.from_numpy(target_np).unsqueeze(0).unsqueeze(0).float()

    num_pos = float(target.ge(0.5).sum().item())
    num_neg = float(target.lt(0.5).sum().item())

    assert num_pos < 150
    assert num_neg > 520000

    pred = torch.full_like(target, 0.25, requires_grad=True)
    loss_fn = ModifiedFocalLoss()
    loss = loss_fn(pred, target)
    loss.backward()

    bg_grad = pred.grad[0, 0, 0, 0, 0].item()
    # Confirms corrected background gradient is restored to > 1e-3 (amplified by ~6,600x)
    assert bg_grad > 1e-3
    assert bg_grad == pytest.approx(0.0028756, rel=1e-3)

    # Verify that loss at pred=0.25 is heavily penalized compared to clean pred=0.01
    with torch.no_grad():
        loss_clean = loss_fn(torch.full_like(target, 0.01), target).item()
        loss_noisy = loss_fn(torch.full_like(target, 0.25), target).item()
    assert loss_noisy > 20.0 * loss_clean


def test_heatmap_head_prior_bias_initialized():
    """Verify that heatmap_head is initialized with focal loss prior bias (-2.944 -> sigma=0.05)."""
    torch.manual_seed(42)
    model = TemporalUNet3D(in_channels=1, base_channels=16, feature_dim=32)

    bias_val = model.heatmap_head.bias.data.item()
    prob_val = torch.sigmoid(torch.tensor(bias_val)).item()

    # Confirms bias is initialized to approved negative prior (b ≈ -2.944, sigma ≈ 0.05)
    assert bias_val == pytest.approx(-2.944, abs=1e-2)
    assert prob_val == pytest.approx(0.05, abs=2e-3)

    # Forward pass on blank input (all zeros) outputs probability <= 0.08
    blank = torch.zeros(1, 1, 16, 32, 32)
    with torch.no_grad():
        hm, _ = model(blank)
    assert hm.mean().item() < 0.08
    assert hm.max().item() < 0.12


def test_peak_detector_diagnostic_mode_bypasses_topk():
    """Verify that diagnostic_mode=True returns all valid peaks without top-K truncation."""
    shape = (32, 64, 64)
    hm = np.full(shape, 0.05, dtype=np.float32)

    # 1 true cell with score 0.65
    true_z, true_y, true_x = 10, 20, 20
    hm[true_z, true_y, true_x] = 0.65

    # 50 noise spikes with score 0.80
    rng = np.random.RandomState(42)
    for _ in range(50):
        rz = rng.randint(2, 30)
        ry = rng.randint(4, 60)
        rx = rng.randint(4, 60)
        if abs(rz - true_z) < 3 and abs(ry - true_y) < 7 and abs(rx - true_x) < 7:
            continue
        hm[rz, ry, rx] = 0.80

    # Production detector (cap = 50) drops the true cell
    det_capped = PeakDetector3D(threshold=0.5, max_peaks_per_frame=50, subpixel=False, diagnostic_mode=False)
    peaks_capped, _ = det_capped.extract_peaks_3d(hm)
    assert len(peaks_capped) == 50
    assert not any(np.allclose(p, [true_z, true_y, true_x]) for p in peaks_capped)

    # Diagnostic detector (unconstrained) retains the true cell
    det_diag = PeakDetector3D(threshold=0.5, max_peaks_per_frame=50, subpixel=False, diagnostic_mode=True)
    peaks_diag, _ = det_diag.extract_peaks_3d(hm)
    assert len(peaks_diag) > 50
    assert any(np.allclose(p, [true_z, true_y, true_x]) for p in peaks_diag)


def test_temporal_consistency_bypass_and_execution():
    """Verify temporal attention handling across 5D (W=1) and 6D (W=2) tensors."""
    model = TemporalUNet3D(in_channels=1, base_channels=8, feature_dim=16, use_temporal_attention=True)

    # 5D single-frame input (W=1)
    x5 = torch.zeros(1, 1, 8, 16, 16)
    with torch.no_grad():
        hm5, feat5 = model(x5)
    assert hm5.shape == (1, 1, 8, 16, 16)
    assert feat5.shape == (1, 16, 8, 16, 16)

    # 6D multi-frame input (W=2)
    x6 = torch.zeros(1, 1, 2, 8, 16, 16)
    with torch.no_grad():
        hm6, feat6 = model(x6)
    assert hm6.shape == (1, 2, 1, 8, 16, 16)
    assert feat6.shape == (1, 2, 16, 8, 16, 16)

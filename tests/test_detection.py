"""Unit tests for Phase 3: 3D U-Net Detection Baseline."""

from pathlib import Path
import numpy as np
import pytest
import torch

from src.data.zarr_reader import open_dataset
from src.detection.dataset import DetectionDataset
from src.detection.heatmap import GaussianHeatmapGenerator, generate_gaussian_heatmap_3d
from src.detection.temporal_attention import TemporalCrossAttention
from src.detection.unet3d import TemporalUNet3D
from src.training.losses import DetectionLoss, ModifiedFocalLoss, SoftDiceLoss
from src.training.train_detector import run_overfit_verification


@pytest.fixture
def synthetic_dataset():
    path = Path("data/fixtures/synthetic_clip")
    assert (path.parent / f"{path.name}.zarr").exists(), "Synthetic fixture zarr missing"
    return open_dataset(path, load_tracks=True)


def test_gaussian_heatmap_3d_properties():
    shape = (16, 64, 64)
    centers = [(8.0, 32.0, 32.0)]
    scale = (1.625, 0.40625, 0.40625)

    hm = generate_gaussian_heatmap_3d(shape, centers, scale=scale, sigma_um=1.5)

    assert hm.shape == shape
    assert hm[8, 32, 32] == pytest.approx(1.0, abs=1e-3)
    assert hm.min() == 0.0

    # Test anisotropy: sigma_um = 1.5 µm
    # In voxels: sigma_z = 1.5 / 1.625 = 0.923 voxels
    # In voxels: sigma_y = 1.5 / 0.40625 = 3.692 voxels
    # Moving 2 voxels in Z should drop faster than moving 2 voxels in Y
    val_z_plus_2 = hm[10, 32, 32]
    val_y_plus_2 = hm[8, 34, 32]
    assert val_z_plus_2 < val_y_plus_2, "Axial Gaussian must decay faster in voxel space due to anisotropy"


def test_spatiotemporal_heatmap_4d():
    generator = GaussianHeatmapGenerator(sigma_um=1.5)
    shape = (2, 16, 64, 64)
    nodes_by_step = {
        0: [{"z": 8.0, "y": 30.0, "x": 30.0}],
        1: [{"z": 8.0, "y": 32.0, "x": 32.0}, {"z": 10.0, "y": 45.0, "x": 45.0}],
    }

    hm_4d = generator.generate_spatiotemporal_4d(shape, nodes_by_step)
    assert hm_4d.shape == (1, 2, 16, 64, 64)
    assert hm_4d[0, 0, 8, 30, 30].item() == pytest.approx(1.0, abs=1e-3)
    assert hm_4d[0, 1, 8, 32, 32].item() == pytest.approx(1.0, abs=1e-3)
    assert hm_4d[0, 1, 10, 45, 45].item() == pytest.approx(1.0, abs=1e-3)


def test_temporal_cross_attention():
    attn = TemporalCrossAttention(channels=32, num_heads=4)
    x = torch.randn(2, 2, 32, 8, 8, 8)
    out = attn(x, B=2, W=2)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()


def test_temporal_unet3d_forward_and_shapes():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalUNet3D(in_channels=1, base_channels=8, feature_dim=16).to(device)

    # 6D temporal patch input: (B=1, C=1, W=2, Z=16, Y=32, X=32)
    x6 = torch.randn(1, 1, 2, 16, 32, 32, device=device)
    hm6, feat6 = model(x6)
    assert hm6.shape == (1, 2, 1, 16, 32, 32)
    assert feat6.shape == (1, 2, 16, 16, 32, 32)
    assert (hm6 >= 0.0).all() and (hm6 <= 1.0).all()

    # 5D single-frame input: (B=1, C=1, Z=16, Y=32, X=32)
    x5 = torch.randn(1, 1, 16, 32, 32, device=device)
    hm5, feat5 = model(x5)
    assert hm5.shape == (1, 1, 16, 32, 32)
    assert feat5.shape == (1, 16, 16, 32, 32)


def test_detection_losses():
    fl = ModifiedFocalLoss()
    dl = SoftDiceLoss()
    criterion = DetectionLoss(focal_weight=1.0, dice_weight=1.0)

    target = torch.zeros((1, 1, 1, 16, 32, 32))
    target[0, 0, 0, 8, 16, 16] = 1.0
    perfect_pred = target.clone()
    bad_pred = torch.zeros_like(target)

    # Perfect prediction loss should be minimal
    loss_perf, metrics_perf = criterion(perfect_pred, target)
    assert loss_perf.item() < 0.05
    assert metrics_perf["loss_dice"] == pytest.approx(0.0, abs=1e-4)

    # Bad prediction loss should be strictly higher
    loss_bad, _ = criterion(bad_pred, target)
    assert loss_bad.item() > loss_perf.item()


def test_detection_dataset_loading(synthetic_dataset):
    ds = DetectionDataset(
        dataset=synthetic_dataset,
        patch_size=(16, 32, 32),
        window_size=2,
        samples_per_epoch=2,
        center_on_cell_prob=1.0,
        augment=True,
        seed=42,
    )

    sample = ds[0]
    assert sample["image"].shape == (1, 2, 16, 32, 32)
    assert sample["heatmap"].shape == (1, 2, 16, 32, 32)
    assert sample["frame_mask"].shape == (1, 2, 1, 1, 1, 1)


def test_overfit_convergence_synthetic(synthetic_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    success = run_overfit_verification(
        dataset=synthetic_dataset,
        device=device,
        patch_size=(16, 32, 32),
        window_size=2,
        base_channels=16,
        num_iterations=35,
        lr=2e-3,
    )
    assert success is True

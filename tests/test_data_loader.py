"""Unit tests for data loading, preprocessing, and patch sampling (Phase 1)."""

from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.preprocessing import (
    compute_physical_distance_matrix,
    normalize_intensity,
    physical_to_pixel,
    pixel_to_physical,
)
from src.data.zarr_reader import open_dataset
from src.data.patch_sampler import SpatioTemporalPatchSampler


FIXTURE_PATH = Path("data/fixtures/synthetic_clip")


def test_open_dataset_fixture():
    """Verify that synthetic clip can be loaded lazily with Zarr and tracksdata."""
    ds = open_dataset(FIXTURE_PATH, require_tracks=True)
    assert ds.name == "synthetic_clip"
    assert ds.shape == (5, 16, 64, 64)
    assert ds.scale == (1.625, 0.40625, 0.40625)
    assert ds.tracks is not None
    assert ds.tracks.num_nodes() == 12
    assert ds.tracks.num_edges() == 10

    # Test reading timepoints
    t_slice = ds.read_timepoints(0, 2)
    assert t_slice.shape == (2, 16, 64, 64)
    assert t_slice.dtype == np.float32

    # Test reading spatial crop
    crop = ds.read_spatial_crop(slice(1, 3), slice(4, 12), slice(10, 30), slice(20, 50))
    assert crop.shape == (2, 8, 20, 30)


def test_intensity_normalization():
    """Verify robust quantile normalization."""
    # Test array with bright outlier
    data = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 1000.0], dtype=np.float32)
    norm = normalize_intensity(data, q_low=0.1, q_high=0.9)
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0
    assert norm.shape == data.shape

    # Test with torch tensor
    tensor = torch.from_numpy(data)
    norm_t = normalize_intensity(tensor, q_low=0.1, q_high=0.9)
    assert float(norm_t.min()) >= 0.0
    assert float(norm_t.max()) <= 1.0

    # Test with metadata quantiles dict
    q_dict = {"0.01": 10.0, "0.999": 40.0}
    norm_meta = normalize_intensity(data, quantiles=q_dict, q_low=0.01, q_high=0.999)
    assert norm_meta[0] == 0.0  # < 10.0 clipped to 0
    assert norm_meta[-1] == 1.0  # > 40.0 clipped to 1


def test_coordinate_transforms():
    """Verify physical and pixel space coordinate transformations."""
    scale = (1.625, 0.40625, 0.40625)
    pixel_coords = np.array([[8.0, 20.0, 40.0]], dtype=np.float32)
    phys_coords = pixel_to_physical(pixel_coords, scale=scale)

    expected_phys = np.array([[8.0 * 1.625, 20.0 * 0.40625, 40.0 * 0.40625]])
    np.testing.assert_allclose(phys_coords, expected_phys, rtol=1e-5)

    # Round trip
    recovered_pixel = physical_to_pixel(phys_coords, scale=scale)
    np.testing.assert_allclose(recovered_pixel, pixel_coords, rtol=1e-5)

    # Test anisotropic distance matrix
    p1 = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    p2 = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)  # delta Z = 1 pixel
    p3 = np.array([[0.0, 4.0, 0.0]], dtype=np.float32)  # delta Y = 4 pixels (4 * 0.40625 = 1.625)

    dist1_2 = compute_physical_distance_matrix(p1, p2, scale=scale)[0, 0]
    dist1_3 = compute_physical_distance_matrix(p1, p3, scale=scale)[0, 0]
    assert np.isclose(dist1_2, 1.625, atol=1e-4)
    assert np.isclose(dist1_3, 1.625, atol=1e-4)


def test_spatio_temporal_patch_sampler():
    """Verify that SpatioTemporalPatchSampler extracts valid patches and subgraphs."""
    ds = open_dataset(FIXTURE_PATH, require_tracks=True)
    sampler = SpatioTemporalPatchSampler(
        dataset=ds,
        patch_size=(8, 32, 32),
        window_size=2,
        samples_per_epoch=20,
        center_on_cell_prob=1.0,
        seed=42,
    )
    assert len(sampler) == 20

    item = sampler[0]
    image = item["image"]
    nodes_by_step = item["nodes_by_step"]
    transitions = item["transitions"]

    # Image shape should be (1, W=2, Z=8, Y=32, X=32)
    assert image.shape == (1, 2, 8, 32, 32)
    assert image.dtype == torch.float32
    assert float(image.min()) >= 0.0
    assert float(image.max()) <= 1.0

    # Ensure nodes are within patch bounds
    for tau in range(2):
        for n in nodes_by_step[tau]:
            assert 0.0 <= n["z"] < 8.0, f"z {n['z']} out of bounds"
            assert 0.0 <= n["y"] < 32.0, f"y {n['y']} out of bounds"
            assert 0.0 <= n["x"] < 32.0, f"x {n['x']} out of bounds"

    # Verify transition matrix dimensions
    assert len(transitions) == 1
    t_mat = transitions[0]
    assert t_mat.shape == (len(nodes_by_step[0]), len(nodes_by_step[1]))


def test_patch_dataloader_throughput():
    """Verify PyTorch DataLoader streaming batch throughput."""
    ds = open_dataset(FIXTURE_PATH, require_tracks=True)
    sampler = SpatioTemporalPatchSampler(
        dataset=ds,
        patch_size=(8, 32, 32),
        window_size=2,
        samples_per_epoch=30,
        seed=123,
    )

    def custom_collate(batch):
        images = torch.stack([b["image"] for b in batch], dim=0)
        origins = [b["origin"] for b in batch]
        nodes = [b["nodes_by_step"] for b in batch]
        transitions = [b["transitions"] for b in batch]
        return {"images": images, "origins": origins, "nodes": nodes, "transitions": transitions}

    loader = DataLoader(sampler, batch_size=4, shuffle=False, collate_fn=custom_collate)

    batch_count = 0
    total_samples = 0
    for batch in loader:
        batch_count += 1
        total_samples += batch["images"].shape[0]
        assert batch["images"].shape[1:] == (1, 2, 8, 32, 32)

    assert batch_count == 8  # 30 samples // 4 = 7 full + 1 remainder
    assert total_samples == 30


def test_spatial_microscopy_augmentation():
    """Verify 3D spatial flipping and in-plane rotation preserve coordinate bounds."""
    from src.data.augmentation import SpatialMicroscopyAugmentation

    aug = SpatialMicroscopyAugmentation(flip_prob=1.0, rot90_prob=1.0, seed=42)
    # Shape: (C=1, W=2, Z=8, Y=32, X=32)
    img = torch.rand((1, 2, 8, 32, 32))
    nodes_by_step = {
        0: [{"node_id": 1, "z": 2.0, "y": 10.0, "x": 15.0}],
        1: [{"node_id": 2, "z": 3.0, "y": 12.0, "x": 16.0}],
    }

    aug_img, aug_nodes = aug(img, nodes_by_step)
    assert aug_img.shape == img.shape
    for tau in [0, 1]:
        for n in aug_nodes[tau]:
            assert 0.0 <= n["z"] < 8.0
            assert 0.0 <= n["y"] < 32.0
            assert 0.0 <= n["x"] < 32.0

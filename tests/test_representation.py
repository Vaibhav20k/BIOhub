"""Unit tests for cell feature extraction, Fourier positional encodings, and node embeddings."""

from pathlib import Path
import numpy as np
import polars as pl
import pytest
import torch

from src.data.zarr_reader import open_dataset
from src.detection.unet3d import TemporalUNet3D
from src.representation.feature_extractor import (
    TrilinearRoIPooler,
    extract_patch_intensity_stats,
)
from src.representation.node_embedding import (
    CellNodeEmbedding,
    extract_node_embeddings_for_dataset,
)
from src.representation.positional_encoding import (
    RelativeSpatialEncoding,
    SpatioTemporalFourierEncoding,
)


def test_spatiotemporal_fourier_encoding():
    """Verify Fourier positional encoding shapes, projections, and continuity."""
    # Test unprojected Fourier encoding
    encoder_raw = SpatioTemporalFourierEncoding(
        num_frequency_bands=8,
        include_raw=True,
        proj_dim=None,
    )
    # Expected dim: 4 coords * 8 bands * 2 (sin+cos) + 4 (raw) = 68
    assert encoder_raw.fourier_dim == 68
    assert encoder_raw.output_dim == 68

    coords = torch.tensor([[0.0, 5.0, 10.0, 15.0], [1.0, 6.2, 11.4, 16.8]], dtype=torch.float32)
    embs_raw = encoder_raw(coords)
    assert embs_raw.shape == (2, 68)

    # Identical coordinates must produce identical embeddings
    same_coords = torch.tensor([[0.0, 5.0, 10.0, 15.0], [0.0, 5.0, 10.0, 15.0]], dtype=torch.float32)
    same_embs = encoder_raw(same_coords)
    assert torch.allclose(same_embs[0], same_embs[1], atol=1e-6)

    # Test projected Fourier encoding
    encoder_proj = SpatioTemporalFourierEncoding(
        num_frequency_bands=6,
        include_raw=True,
        proj_dim=64,
    )
    assert encoder_proj.output_dim == 64
    embs_proj = encoder_proj(coords)
    assert embs_proj.shape == (2, 64)


def test_relative_spatial_encoding():
    """Verify relative displacement encoding for edge conditioning."""
    rel_encoder = RelativeSpatialEncoding(num_frequency_bands=6, output_dim=32)
    assert rel_encoder.output_dim == 32

    # Two relative displacement vectors in voxels: [dz, dy, dx]
    deltas = torch.tensor([[0.0, 0.0, 0.0], [1.5, -2.0, 3.2]], dtype=torch.float32)
    rel_embs = rel_encoder(deltas)

    assert rel_embs.shape == (2, 32)
    # Zero displacement and non-zero displacement must have distinct embeddings
    assert not torch.allclose(rel_embs[0], rel_embs[1])


def test_trilinear_roi_pooler_analytical_interpolation():
    """Verify trilinear interpolation matches exact analytical values at sub-voxel positions."""
    pooler = TrilinearRoIPooler(
        sample_neighborhood=False,
        in_channels=1,
        out_channels=None,
    )

    # Construct synthetic 3D volume where F(z, y, x) = 10.0 * z + 2.0 * y + 1.0 * x
    Z, Y, X = 8, 12, 16
    zg = np.arange(Z, dtype=np.float32)
    yg = np.arange(Y, dtype=np.float32)
    xg = np.arange(X, dtype=np.float32)
    zz, yy, xx = np.meshgrid(zg, yg, xg, indexing="ij")
    analytical_vol = 10.0 * zz + 2.0 * yy + 1.0 * xx

    feat_tensor = torch.from_numpy(analytical_vol).unsqueeze(0).unsqueeze(0)  # (1, 1, Z, Y, X)

    # Query continuous sub-voxel coordinates
    # For (z, y, x) = (2.5, 3.5, 4.5), exact value is 10*2.5 + 2*3.5 + 1*4.5 = 25 + 7 + 4.5 = 36.5
    test_coords = torch.tensor([[2.5, 3.5, 4.5], [1.0, 2.0, 3.0]], dtype=torch.float32)
    sampled = pooler(feat_tensor, test_coords)

    assert sampled.shape == (2, 1)
    expected_0 = 10.0 * 2.5 + 2.0 * 3.5 + 1.0 * 4.5
    expected_1 = 10.0 * 1.0 + 2.0 * 2.0 + 1.0 * 3.0

    assert np.isclose(sampled[0, 0].item(), expected_0, atol=1e-4)
    assert np.isclose(sampled[1, 0].item(), expected_1, atol=1e-4)


def test_trilinear_roi_pooler_neighborhood_sampling():
    """Verify 7-point morphological neighborhood pooling."""
    pooler = TrilinearRoIPooler(
        sample_neighborhood=True,
        in_channels=16,
        out_channels=64,
    )

    feat_tensor = torch.randn(1, 16, 8, 16, 16)
    coords = torch.tensor([[3.2, 7.8, 8.4], [4.0, 8.0, 8.0]], dtype=torch.float32)

    pooled = pooler(feat_tensor, coords)
    assert pooled.shape == (2, 64)

    # Empty coords check
    empty_coords = torch.empty((0, 3), dtype=torch.float32)
    empty_pooled = pooler(feat_tensor, empty_coords)
    assert empty_pooled.shape == (0, 64)


def test_extract_patch_intensity_stats():
    """Verify local microscopy image intensity summary statistics."""
    vol = np.ones((10, 20, 20), dtype=np.float32) * 0.5
    vol[4:7, 9:12, 9:12] = 1.0  # Center cube has intensity 1.0

    coords = np.array([[5.0, 10.0, 10.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    stats = extract_patch_intensity_stats(vol, coords, radius=(1, 1, 1))

    assert stats.shape == (2, 4)  # [mean, max, min, std]
    # At center cube, min and max should both be 1.0
    assert np.isclose(stats[0, 1], 1.0)
    assert np.isclose(stats[0, 2], 1.0)
    assert np.isclose(stats[0, 0], 1.0)


def test_cell_node_embedding_forward_and_gradients():
    """Verify multi-modal fusion forward pass and gradient backpropagation."""
    embedder = CellNodeEmbedding(
        in_visual_dim=32,
        num_frequency_bands=8,
        embedding_dim=128,
        sample_neighborhood=True,
        dropout=0.0,
    )

    N = 4
    # Mock inputs
    vis_features = torch.randn(N, 64, requires_grad=True)
    coords_tzyx = torch.tensor([[0.0, 4.0, 8.0, 8.0], [0.0, 5.0, 10.0, 12.0], [1.0, 4.5, 9.0, 9.0], [1.0, 5.2, 10.5, 12.5]])
    scores = torch.tensor([[0.95], [0.82], [0.90], [0.75]])
    intensity_stats = torch.randn(N, 4)

    out = embedder(vis_features, coords_tzyx, scores, intensity_stats)
    assert out.shape == (N, 128)

    # Backward gradient flow test
    loss = out.sum()
    loss.backward()
    assert vis_features.grad is not None
    assert torch.all(torch.isfinite(vis_features.grad))


def test_extract_node_embeddings_for_dataset_fixture():
    """Verify end-to-end dataset node embedding extraction on synthetic fixture."""
    fixture_path = Path("data/fixtures/synthetic_clip")
    dataset = open_dataset(fixture_path, load_tracks=True, require_tracks=True)

    # Miniature detector backbone
    model = TemporalUNet3D(in_channels=1, base_channels=8, feature_dim=32, use_temporal_attention=False)
    model.eval()

    embedder = CellNodeEmbedding(
        in_visual_dim=32,
        embedding_dim=64,
        sample_neighborhood=True,
        scale=dataset.scale,
    )
    embedder.eval()

    # Synthetic detections across t=0 and t=1
    mock_nodes_df = pl.DataFrame(
        {
            "t": [0, 0, 1, 1],
            "node_id": [0, 1, 2, 3],
            "z": [4.0, 6.0, 4.2, 6.1],
            "y": [12.0, 18.0, 12.5, 18.2],
            "x": [12.0, 18.0, 12.3, 18.4],
            "score": [0.92, 0.88, 0.94, 0.85],
        }
    )

    sorted_df, embeddings = extract_node_embeddings_for_dataset(
        model=model,
        embedder=embedder,
        dataset=dataset,
        nodes_df=mock_nodes_df,
        device=torch.device("cpu"),
    )

    assert sorted_df.height == 4
    assert embeddings.shape == (4, 64)
    assert not torch.isnan(embeddings).any()

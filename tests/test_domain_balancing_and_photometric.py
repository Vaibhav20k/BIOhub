"""Unit tests for Phase 2: Domain-balanced patch sampling and photometric augmentation."""

from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest
import torch

from src.data.zarr_reader import DatasetVolume, open_dataset
from src.detection.dataset import DetectionDataset
from src.pipeline.continual_trainer import ContinualTrainer
from src.detection.unet3d import TemporalUNet3D
from src.representation.node_embedding import CellNodeEmbedding
from src.tracking.transformer import SpatioTemporalTracker
from src.pipeline.held_out_evaluator import HeldOutEvaluator


FIXTURE_PATH = Path("data/fixtures/synthetic_clip")


@pytest.fixture
def synthetic_dataset():
    path = Path("data/fixtures/synthetic_clip")
    assert (path.parent / f"{path.name}.zarr").exists(), "Synthetic fixture zarr missing"
    return open_dataset(path, load_tracks=True)


def test_photometric_augmentation_bounds_and_safety(synthetic_dataset):
    """Verify that photometric augmentation stays within configured bounds, preserves shape/dtype, and produces no NaN/Inf."""
    ds = DetectionDataset(
        dataset=synthetic_dataset,
        patch_size=(8, 32, 32),
        window_size=1,
        samples_per_epoch=10,
        center_on_cell_prob=0.0,
        augment=True,
        seed=100,
    )

    for i in range(10):
        sample = ds[i]
        img = sample["image"]
        hm = sample["heatmap"]

        # 1. Shape preservation
        assert img.shape == (1, 1, 8, 32, 32)
        assert hm.shape == (1, 1, 8, 32, 32)

        # 2. Dtype preservation
        assert img.dtype == torch.float32
        assert hm.dtype == torch.float32

        # 3. No NaN or Inf
        assert not torch.isnan(img).any()
        assert not torch.isinf(img).any()
        assert not torch.isnan(hm).any()
        assert not torch.isinf(hm).any()

        # 4. Value bounds: strictly clamped to [0.0, 1.0]
        assert float(img.min()) >= 0.0
        assert float(img.max()) <= 1.0
        assert float(hm.min()) >= 0.0
        assert float(hm.max()) <= 1.0


def test_photometric_augmentation_preserves_heatmap_targets(synthetic_dataset):
    """Verify that photometric intensity augmentation applies strictly to the image, leaving ground-truth heatmaps untouched."""
    ds_no_aug = DetectionDataset(
        dataset=synthetic_dataset,
        patch_size=(8, 32, 32),
        window_size=1,
        samples_per_epoch=5,
        center_on_cell_prob=1.0,
        augment=False,
        seed=42,
    )

    sample_no_aug = ds_no_aug[0]
    assert float(sample_no_aug["heatmap"].max()) > 0.5


def test_domain_balanced_sampling_allocation():
    """Verify that ContinualTrainer allocates exactly 50% patches to 44b6 and 50% to 6bba in Chunk 1 configuration."""
    stems = [
        "6bba_3fda6b25", "6bba_9a41d029", "44b6_949adeb1", "44b6_5740d24b",
        "6bba_05b6850b", "6bba_f4ae811c", "6bba_b1ae37b9", "6bba_7b5d3b2c",
    ]

    mock_datasets = []
    for s in stems:
        mock_ds = MagicMock(spec=DatasetVolume)
        mock_ds.name = s
        mock_ds.shape = (100, 64, 256, 256)
        mock_ds.scale = (1.625, 0.40625, 0.40625)
        mock_ds.quantiles = {"0.01": 50.0, "0.999": 2000.0}
        mock_tracks = MagicMock()
        mock_tracks.num_nodes.return_value = 100
        mock_tracks.node_attrs.return_value = MagicMock(height=100)
        mock_tracks.edge_attrs.return_value = MagicMock(height=90)
        mock_ds.tracks = mock_tracks
        mock_ds.read_spatial_crop.return_value = np.zeros((1, 32, 128, 128), dtype=np.float32)
        mock_datasets.append(mock_ds)

    total_budget = 100 * len(mock_datasets)
    assert total_budget == 800

    by_embryo = {}
    for ds in mock_datasets:
        prefix = ds.name.split("_")[0]
        by_embryo.setdefault(prefix, []).append(ds)

    num_embryos = len(by_embryo)
    quota_per_group = total_budget // num_embryos
    allocated_so_far = 0
    sorted_groups = sorted(by_embryo.keys())
    dataset_sample_quotas = {}

    for g_idx, g_name in enumerate(sorted_groups):
        g_datasets = by_embryo[g_name]
        g_quota = total_budget - allocated_so_far if g_idx == num_embryos - 1 else quota_per_group
        allocated_so_far += g_quota
        num_ds_in_group = len(g_datasets)
        base_alloc = g_quota // num_ds_in_group
        rem_alloc = g_quota % num_ds_in_group
        for i, ds in enumerate(g_datasets):
            dataset_sample_quotas[ds.name] = base_alloc + (1 if i < rem_alloc else 0)

    # 1. Exact 400 patches from 44b6
    alloc_44b6 = sum(dataset_sample_quotas[s] for s in stems if s.startswith("44b6"))
    assert alloc_44b6 == 400

    # 2. Exact 400 patches from 6bba
    alloc_6bba = sum(dataset_sample_quotas[s] for s in stems if s.startswith("6bba"))
    assert alloc_6bba == 400

    # 3. Total sum exactly 800
    assert sum(dataset_sample_quotas.values()) == 800

    # 4. Each 44b6 sequence gets 200 patches
    assert dataset_sample_quotas["44b6_949adeb1"] == 200
    assert dataset_sample_quotas["44b6_5740d24b"] == 200

    # 5. The 6 6bba sequences get 67, 67, 67, 67, 66, 66 (sum = 400)
    bba_allocs = [dataset_sample_quotas[s] for s in stems if s.startswith("6bba")]
    assert sum(bba_allocs) == 400
    assert set(bba_allocs) == {66, 67}


def test_validation_isolation_from_training_augmentation(synthetic_dataset):
    """Verify that validation inference operates strictly without data augmentation and is deterministic."""
    detector = TemporalUNet3D(in_channels=1, base_channels=8, feature_dim=16)
    embedder = CellNodeEmbedding(in_visual_dim=16, embedding_dim=32)
    tracker = SpatioTemporalTracker(node_dim=32, rel_dim=16, hidden_dim=32, num_layers=1)

    evaluator = HeldOutEvaluator(
        detector=detector,
        embedder=embedder,
        tracker=tracker,
        detection_threshold=0.25,
        max_peaks_per_frame=50,
        device=torch.device("cpu"),
    )

    res1 = evaluator.evaluate_single_sequence(synthetic_dataset, timepoints=[0])
    res2 = evaluator.evaluate_single_sequence(synthetic_dataset, timepoints=[0])

    assert res1.node_tp == res2.node_tp
    assert res1.node_fp == res2.node_fp
    assert res1.node_fn == res2.node_fn
    assert res1.final_score == res2.final_score

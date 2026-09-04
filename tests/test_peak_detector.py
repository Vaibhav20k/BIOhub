"""Tests for 3D peak detection, sub-pixel quadratic refinement, and Hungarian bipartite matching evaluation."""

from pathlib import Path
import numpy as np
import polars as pl
import pytest
import torch

from src.data.zarr_reader import DEFAULT_SCALE, open_dataset
from src.detection.heatmap import generate_gaussian_heatmap_3d
from src.detection.peak_detector import PeakDetector3D
from src.detection.unet3d import TemporalUNet3D
from src.evaluation.node_matching import (
    BipartiteNodeEvaluator,
    evaluate_node_detections,
)


def test_peak_detector_basic_extraction():
    """Verify that PeakDetector3D correctly finds single and multiple 3D peaks."""
    shape = (16, 32, 32)
    # Generate continuous Gaussian at integer voxel center (8, 16, 16)
    hm = generate_gaussian_heatmap_3d(shape=shape, centers=[(8.0, 16.0, 16.0)], sigma_um=1.5)

    detector = PeakDetector3D(threshold=0.5, subpixel=False)
    peaks, scores = detector.extract_peaks_3d(hm)

    assert len(peaks) == 1
    assert np.allclose(peaks[0], [8.0, 16.0, 16.0], atol=1e-3)
    assert np.isclose(scores[0], 1.0, atol=1e-2)


def test_subpixel_taylor_refinement():
    """Verify that 3D Taylor expansion refines sub-voxel coordinates closer to true continuous centroid."""
    shape = (20, 40, 40)
    true_centroid = (10.35, 20.65, 22.25)
    hm = generate_gaussian_heatmap_3d(shape=shape, centers=[true_centroid], sigma_um=1.5)

    # Without subpixel
    det_discrete = PeakDetector3D(threshold=0.3, subpixel=False)
    peaks_disc, _ = det_discrete.extract_peaks_3d(hm)
    assert len(peaks_disc) == 1
    # Discrete voxel peak will be rounded to (10, 21, 22)
    assert np.allclose(peaks_disc[0], [10.0, 21.0, 22.0], atol=1e-3)
    disc_err = np.linalg.norm(peaks_disc[0] - np.array(true_centroid))

    # With subpixel
    det_sub = PeakDetector3D(threshold=0.3, subpixel=True)
    peaks_sub, _ = det_sub.extract_peaks_3d(hm)
    assert len(peaks_sub) == 1
    sub_err = np.linalg.norm(peaks_sub[0] - np.array(true_centroid))

    # Sub-pixel error must be strictly smaller than discrete rounding error
    assert sub_err < disc_err
    assert sub_err < 0.15, f"Expected subpixel error < 0.15 voxels, got {sub_err}"


def test_peak_detector_max_peaks_cap():
    """Verify that max_peaks_per_frame retains only the top-K highest scoring peaks."""
    shape = (16, 32, 32)
    hm = np.zeros(shape, dtype=np.float32)

    # Place 5 distinct peaks with different heights
    coords = [(4, 8, 8), (6, 12, 12), (8, 16, 16), (10, 20, 20), (12, 24, 24)]
    values = [0.4, 0.95, 0.7, 0.85, 0.5]
    for (z, y, x), val in zip(coords, values):
        hm[z, y, x] = val

    detector = PeakDetector3D(threshold=0.3, max_peaks_per_frame=3, subpixel=False)
    peaks, scores = detector.extract_peaks_3d(hm)

    assert len(peaks) == 3
    # Should retain 0.95, 0.85, 0.7 in sorted order
    assert np.allclose(scores, [0.95, 0.85, 0.7], atol=1e-4)
    assert np.allclose(peaks[0], [6, 12, 12])
    assert np.allclose(peaks[1], [10, 20, 20])
    assert np.allclose(peaks[2], [8, 16, 16])


def test_peak_detector_empty_heatmap():
    """Verify that an empty/zero heatmap returns empty arrays cleanly."""
    shape = (8, 16, 16)
    hm = np.zeros(shape, dtype=np.float32)

    detector = PeakDetector3D(threshold=0.3)
    peaks, scores = detector.extract_peaks_3d(hm)

    assert peaks.shape == (0, 3)
    assert scores.shape == (0,)


def test_hungarian_bipartite_node_evaluator():
    """Verify Hungarian bipartite matching with physical distance cutoff (7.0 µm)."""
    scale = (1.625, 0.40625, 0.40625)

    # Ground truth nodes in voxels at t=0
    # Physical pos:
    # g0: (2.0, 10.0, 10.0) -> (3.25, 4.0625, 4.0625)
    # g1: (4.0, 20.0, 20.0) -> (6.50, 8.1250, 8.1250)
    gt_df = pl.DataFrame(
        {
            "t": [0, 0],
            "z": [2.0, 4.0],
            "y": [10.0, 20.0],
            "x": [10.0, 20.0],
        }
    )

    # Predicted nodes:
    # p0: exactly matches g0 (dist = 0)
    # p1: matches g1 with small offset (0.5 voxels in Y -> ~0.203 µm)
    # p2: false positive far away (dist > 7.0 µm from all)
    pred_df = pl.DataFrame(
        {
            "t": [0, 0, 0],
            "z": [2.0, 4.0, 10.0],
            "y": [10.0, 20.5, 40.0],
            "x": [10.0, 20.0, 40.0],
            "score": [0.95, 0.90, 0.80],
        }
    )

    res = evaluate_node_detections(pred_df, gt_df, scale=scale, max_distance=7.0)

    assert res.tp == 2
    assert res.fp == 1  # p2 is unassigned
    assert res.fn == 0  # both g0 and g1 matched
    assert np.isclose(res.recall, 1.0)
    assert np.isclose(res.precision, 2.0 / 3.0)
    assert res.num_gt_nodes == 2
    assert res.num_pred_nodes == 3
    assert res.mean_error_um < 0.3  # very small localization error


def test_hungarian_bipartite_distance_cutoff():
    """Verify that pairs separated by more than max_distance are counted as FP and FN."""
    scale = (1.0, 1.0, 1.0)
    # Ground truth node at (0, 0, 0)
    gt_df = pl.DataFrame({"t": [0], "z": [0.0], "y": [0.0], "x": [0.0]})
    # Prediction at (0, 0, 10.0) -> distance = 10.0 > 7.0
    pred_df = pl.DataFrame({"t": [0], "z": [0.0], "y": [0.0], "x": [10.0], "score": [0.9]})

    res = evaluate_node_detections(pred_df, gt_df, scale=scale, max_distance=7.0)

    assert res.tp == 0
    assert res.fp == 1
    assert res.fn == 1
    assert res.precision == 0.0
    assert res.recall == 0.0


def test_detect_dataset_nodes_synthetic_fixture():
    """Verify full-volume detector inference on synthetic clip fixture."""
    fixture_path = Path("data/fixtures/synthetic_clip")
    dataset = open_dataset(fixture_path, load_tracks=True, require_tracks=True)

    # Initialize a miniature TemporalUNet3D
    model = TemporalUNet3D(in_channels=1, base_channels=8, feature_dim=16, use_temporal_attention=False)
    model.eval()

    detector = PeakDetector3D(threshold=0.1, scale=dataset.scale)
    pred_df = detector.detect_dataset_nodes(
        model=model,
        dataset=dataset,
        device=torch.device("cpu"),
        timepoints=[0, 1],
    )

    assert isinstance(pred_df, pl.DataFrame)
    for col in ["t", "node_id", "z", "y", "x", "score"]:
        assert col in pred_df.columns

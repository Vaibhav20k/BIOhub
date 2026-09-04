"""Unit tests for Spatio-Temporal Transformer Tracker and candidate edge graph generation."""

from pathlib import Path
import numpy as np
import polars as pl
import pytest
import torch

from src.data.zarr_reader import open_dataset
from src.representation.node_embedding import CellNodeEmbedding
from src.tracking.candidate_edges import CandidateEdgeBatch, CandidateEdgeBuilder
from src.tracking.dataset import TrackingPairDataset
from src.tracking.losses import TrackerLoss
from src.tracking.transformer import SpatioTemporalTracker


def test_candidate_edge_builder_distance_filtering():
    """Verify that CandidateEdgeBuilder strictly respects physical search radius (7.0 µm)."""
    scale = (1.625, 0.40625, 0.40625)
    builder = CandidateEdgeBuilder(max_distance_um=7.0, scale=scale)

    # Frame t nodes: node 0 at (10.0, 20.0, 20.0)
    src_df = pl.DataFrame({"node_id": [0], "z": [10.0], "y": [20.0], "x": [20.0]})

    # Frame t+1 nodes:
    # node 1: close (offset 1.0 voxel in y -> 0.406 µm < 7.0 µm) -> VALID CANDIDATE
    # node 2: far (offset 30.0 voxels in y -> 12.18 µm > 7.0 µm) -> MUST BE FILTERED OUT
    dst_df = pl.DataFrame({"node_id": [1, 2], "z": [10.0, 10.0], "y": [21.0, 50.0], "x": [20.0, 20.0]})

    cand_batch = builder.build_frame_pair_candidates(src_nodes=src_df, dst_nodes=dst_df)

    assert len(cand_batch.src_node_ids) == 1
    assert cand_batch.src_node_ids[0] == 0
    assert cand_batch.dst_node_ids[0] == 1
    assert cand_batch.distances_um[0, 0] < 1.0


def test_candidate_edge_ground_truth_labels_and_divisions():
    """Verify ground-truth edge assignment and mitosis division labeling."""
    builder = CandidateEdgeBuilder(max_distance_um=10.0, scale=(1.0, 1.0, 1.0))

    src_df = pl.DataFrame({"node_id": [10, 20], "z": [5.0, 15.0], "y": [5.0, 15.0], "x": [5.0, 15.0]})
    dst_df = pl.DataFrame({"node_id": [101, 102, 201], "z": [5.2, 5.8, 15.1], "y": [5.1, 5.2, 15.0], "x": [5.0, 5.1, 15.2]})

    # Cell 10 divides into 101 and 102; Cell 20 continues as 201
    gt_edges = {(10, 101), (10, 102), (20, 201)}
    dividing_nodes = {10}

    cand_batch = builder.build_frame_pair_candidates(
        src_nodes=src_df,
        dst_nodes=dst_df,
        gt_edges=gt_edges,
        dividing_nodes=dividing_nodes,
    )

    assert cand_batch.edge_labels is not None
    assert cand_batch.division_labels is not None

    # Check division labels: node 10 is dividing (1.0), node 20 is not (0.0)
    assert cand_batch.division_labels[0].item() == 1.0
    assert cand_batch.division_labels[1].item() == 0.0

    # Check that true edges got label 1.0
    for u, v, lbl in zip(cand_batch.src_node_ids, cand_batch.dst_node_ids, cand_batch.edge_labels.tolist()):
        if (u, v) in gt_edges:
            assert lbl == 1.0
        else:
            assert lbl == 0.0


def test_tracker_forward_pass_and_shapes():
    """Verify SpatioTemporalTracker forward tensor shapes and bounded probabilities."""
    node_dim = 64
    rel_dim = 32
    hidden_dim = 64
    tracker = SpatioTemporalTracker(
        node_dim=node_dim,
        rel_dim=rel_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
    )
    tracker.eval()

    n_src = 3
    n_dst = 4
    src_embs = torch.randn(n_src, node_dim)
    dst_embs = torch.randn(n_dst, node_dim)

    # Mock 5 candidate edges
    cand_batch = CandidateEdgeBatch(
        src_node_ids=[0, 0, 1, 2, 2],
        dst_node_ids=[0, 1, 2, 2, 3],
        src_indices=torch.tensor([0, 0, 1, 2, 2], dtype=torch.long),
        dst_indices=torch.tensor([0, 1, 2, 2, 3], dtype=torch.long),
        delta_zyx=torch.randn(5, 3),
        distances_um=torch.rand(5, 1) * 5.0,
    )

    with torch.no_grad():
        pred = tracker(src_embs, dst_embs, cand_batch)

    assert pred.edge_logits.shape == (5,)
    assert pred.edge_probs.shape == (5,)
    assert (pred.edge_probs >= 0.0).all() and (pred.edge_probs <= 1.0).all()

    assert pred.division_logits.shape == (n_src,)
    assert pred.division_probs.shape == (n_src,)
    assert (pred.division_probs >= 0.0).all() and (pred.division_probs <= 1.0).all()


def test_tracker_empty_candidates():
    """Verify tracker behavior when zero candidate edges exist."""
    tracker = SpatioTemporalTracker(node_dim=32, hidden_dim=32, num_layers=1)
    tracker.eval()

    src_embs = torch.randn(2, 32)
    dst_embs = torch.randn(2, 32)
    cand_batch = CandidateEdgeBatch(
        src_node_ids=[],
        dst_node_ids=[],
        src_indices=torch.empty((0,), dtype=torch.long),
        dst_indices=torch.empty((0,), dtype=torch.long),
        delta_zyx=torch.empty((0, 3)),
        distances_um=torch.empty((0, 1)),
    )

    pred = tracker(src_embs, dst_embs, cand_batch)
    assert pred.edge_logits.shape == (0,)
    assert pred.edge_probs.shape == (0,)
    assert pred.division_logits.shape == (2,)


def test_tracker_loss_convergence():
    """Verify that TrackerLoss gradients strictly drive loss to decrease on synthetic targets."""
    tracker = SpatioTemporalTracker(node_dim=32, rel_dim=16, hidden_dim=32, num_layers=1, dropout=0.0)
    optimizer = torch.optim.Adam(tracker.parameters(), lr=1e-2)
    criterion = TrackerLoss()

    src_embs = torch.randn(2, 32)
    dst_embs = torch.randn(2, 32)
    cand_batch = CandidateEdgeBatch(
        src_node_ids=[0, 1],
        dst_node_ids=[0, 1],
        src_indices=torch.tensor([0, 1], dtype=torch.long),
        dst_indices=torch.tensor([0, 1], dtype=torch.long),
        delta_zyx=torch.randn(2, 3),
        distances_um=torch.tensor([[1.0], [2.0]]),
        edge_labels=torch.tensor([1.0, 0.0]),
        division_labels=torch.tensor([1.0, 0.0]),
    )

    initial_loss = None
    for _ in range(15):
        optimizer.zero_grad()
        pred = tracker(src_embs, dst_embs, cand_batch)
        loss_dict = criterion(
            edge_logits=pred.edge_logits,
            edge_targets=cand_batch.edge_labels,
            division_logits=pred.division_logits,
            division_targets=cand_batch.division_labels,
        )
        loss = loss_dict["loss"]
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    assert final_loss < initial_loss * 0.5, f"Expected loss reduction by >50%, got from {initial_loss} to {final_loss}"


def test_tracking_dataset_synthetic_fixture():
    """Verify TrackingPairDataset creation and item retrieval on synthetic clip fixture."""
    fixture_path = Path("data/fixtures/synthetic_clip")
    dataset = open_dataset(fixture_path, load_tracks=True, require_tracks=True)

    embedder = CellNodeEmbedding(
        in_visual_dim=32,
        embedding_dim=64,
        scale=dataset.scale,
    )

    tracking_dataset = TrackingPairDataset(
        dataset=dataset,
        embedder=embedder,
        detector_model=None,  # Use coordinate-based embedder without full volume CNN
        max_distance_um=7.0,
        scale=dataset.scale,
    )

    assert len(tracking_dataset) > 0
    src_embs, dst_embs, cand_batch = tracking_dataset[0]

    assert src_embs.ndim == 2
    assert dst_embs.ndim == 2
    assert src_embs.shape[1] == 64
    assert dst_embs.shape[1] == 64
    assert isinstance(cand_batch, CandidateEdgeBatch)

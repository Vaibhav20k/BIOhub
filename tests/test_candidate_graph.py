"""Unit tests for Spatio-Temporal Candidate Graph construction, costs, and tracksdata export."""

import math
from pathlib import Path
import numpy as np
import polars as pl
import pytest
import rustworkx as rx
import torch
import tracksdata

from src.data.zarr_reader import open_dataset
from src.graph.candidate_graph import (
    CandidateEdge,
    CandidateGraph,
    CandidateGraphBuilder,
    CandidateNode,
    candidate_graph_to_dataframes,
    probability_to_cost,
)
from src.graph.graph_export import (
    candidate_graph_to_rustworkx,
    solution_edges_to_tracksdata,
    validate_dag_temporal_monotonicity,
)
from src.tracking.transformer import SpatioTemporalTracker


def test_probability_to_cost():
    """Verify log-odds cost transformation properties and numerical stability."""
    # At p = 0.5, log(0.5 / 0.5) = 0.0
    assert np.isclose(probability_to_cost(0.5), 0.0, atol=1e-5)

    # High probability should have negative cost (favorable reward to pick)
    cost_high = probability_to_cost(0.95)
    assert cost_high < 0.0

    # Low probability should have positive cost (penalty)
    cost_low = probability_to_cost(0.05)
    assert cost_low > 0.0

    # Symmetric around 0.5
    assert np.isclose(cost_high, -cost_low, atol=1e-4)

    # Extreme probabilities clipped by epsilon (never inf or nan)
    cost_zero = probability_to_cost(0.0)
    cost_one = probability_to_cost(1.0)
    assert math.isfinite(cost_zero) and cost_zero > 0.0
    assert math.isfinite(cost_one) and cost_one < 0.0


def test_candidate_graph_data_structures_and_adjacencies():
    """Verify node and edge insertion and bidirectional adjacency indexing."""
    graph = CandidateGraph()

    n0 = CandidateNode(node_id=1, t=0, z=2.0, y=10.0, x=10.0, score=0.9, div_prob=0.1)
    n1 = CandidateNode(node_id=2, t=1, z=2.1, y=10.2, x=10.1, score=0.85, div_prob=0.05)
    n2 = CandidateNode(node_id=3, t=1, z=5.0, y=20.0, x=20.0, score=0.7, div_prob=0.0)

    graph.add_node(n0)
    graph.add_node(n1)
    graph.add_node(n2)

    e0 = CandidateEdge(edge_id=10, source_id=1, target_id=2, dt=1, distance_um=0.5, probability=0.95, cost=-2.94)
    graph.add_edge(e0)

    assert graph.num_nodes == 3
    assert graph.num_edges == 1
    assert graph.outgoing_edges[1] == [10]
    assert graph.incoming_edges[2] == [10]
    assert len(graph.incoming_edges[3]) == 0
    assert graph.nodes_by_time[0] == [1]
    assert sorted(graph.nodes_by_time[1]) == [2, 3]

    summary = graph.get_summary()
    assert summary["num_nodes"] == 3
    assert summary["num_edges"] == 1
    assert summary["num_timepoints"] == 2


def test_temporal_monotonicity_validator():
    """Verify that validate_dag_temporal_monotonicity catches backward or same-frame edges."""
    valid_graph = CandidateGraph()
    valid_graph.add_node(CandidateNode(node_id=1, t=0, z=0.0, y=0.0, x=0.0, score=1.0))
    valid_graph.add_node(CandidateNode(node_id=2, t=1, z=0.0, y=0.0, x=0.0, score=1.0))
    valid_graph.add_edge(CandidateEdge(edge_id=0, source_id=1, target_id=2, dt=1, distance_um=0.0, probability=1.0, cost=-5.0))
    assert validate_dag_temporal_monotonicity(valid_graph) is True

    # Corrupt edge going backward in time (t=1 -> t=0)
    invalid_graph = CandidateGraph()
    invalid_graph.add_node(CandidateNode(node_id=1, t=1, z=0.0, y=0.0, x=0.0, score=1.0))
    invalid_graph.add_node(CandidateNode(node_id=2, t=0, z=0.0, y=0.0, x=0.0, score=1.0))
    invalid_graph.add_edge(CandidateEdge(edge_id=0, source_id=1, target_id=2, dt=-1, distance_um=0.0, probability=1.0, cost=0.0))
    assert validate_dag_temporal_monotonicity(invalid_graph) is False


def test_candidate_graph_builder_synthetic_nodes():
    """Verify end-to-end candidate graph assembly with mock tracker and node embeddings."""
    tracker = SpatioTemporalTracker(node_dim=32, rel_dim=16, hidden_dim=32, num_layers=1)
    tracker.eval()

    # 4 nodes across 3 timepoints
    nodes_df = pl.DataFrame(
        {
            "node_id": [10, 20, 30, 40],
            "t": [0, 1, 1, 2],
            "z": [5.0, 5.2, 15.0, 5.4],
            "y": [10.0, 10.3, 30.0, 10.5],
            "x": [10.0, 10.1, 30.0, 10.3],
            "score": [0.95, 0.90, 0.40, 0.92],
        }
    )

    node_embeddings = torch.randn(4, 32)

    builder = CandidateGraphBuilder(max_distance_um=7.0, allow_frame_skips=True)
    graph = builder.build_candidate_graph(
        nodes_df=nodes_df,
        tracker=tracker,
        node_embeddings=node_embeddings,
        device=torch.device("cpu"),
    )

    assert graph.num_nodes == 4
    # Node 10 (t=0) is close to Node 20 (t=1), but far from Node 30 (t=1, distance > 7.0 µm)
    assert len(graph.outgoing_edges[10]) >= 1
    edge_ids = graph.outgoing_edges[10]
    target_ids = [graph.edges[eid].target_id for eid in edge_ids]
    assert 20 in target_ids
    assert 30 not in target_ids  # Out-of-radius candidate must be filtered out

    # Check Polars DataFrame export
    df_nodes, df_edges = candidate_graph_to_dataframes(graph)
    assert df_nodes.height == 4
    assert df_edges.height == graph.num_edges


def test_graph_export_to_rustworkx_and_tracksdata():
    """Verify seamless conversion to rustworkx and tracksdata LineageGraph."""
    graph = CandidateGraph()
    n0 = CandidateNode(node_id=101, t=0, z=1.0, y=2.0, x=3.0, score=0.95)
    n1 = CandidateNode(node_id=102, t=1, z=1.2, y=2.1, x=3.2, score=0.92)
    e0 = CandidateEdge(edge_id=501, source_id=101, target_id=102, dt=1, distance_um=0.4, probability=0.98, cost=-3.8)

    graph.add_node(n0)
    graph.add_node(n1)
    graph.add_edge(e0)

    # 1. Rustworkx conversion
    rx_graph = candidate_graph_to_rustworkx(graph)
    assert isinstance(rx_graph, rx.PyDiGraph)
    assert rx_graph.num_nodes() == 2
    assert rx_graph.num_edges() == 1

    # 2. Tracksdata conversion of solution edge
    lineage = solution_edges_to_tracksdata(graph, active_edge_ids=[501])
    assert isinstance(lineage, tracksdata.graph.IndexedRXGraph)
    assert lineage.num_nodes() == 2
    assert lineage.num_edges() == 1
    node_attrs = lineage.node_attrs()
    assert "node_id" in node_attrs.columns
    assert "t" in node_attrs.columns

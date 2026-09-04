"""Unit tests for Integer Linear Programming (ILP) Lineage Solver and official competition metrics."""

import numpy as np
import polars as pl
import pytest

from src.evaluation.tracking_metrics import CompetitionEvaluator
from src.graph.candidate_graph import (
    CandidateEdge,
    CandidateGraph,
    CandidateNode,
)
from src.optimization.ilp_solver import LineageILPSolver


def test_ilp_linear_track_reconstruction():
    """Verify that ILP reconstructs an unbranching linear track while rejecting false positive edges."""
    graph = CandidateGraph()

    # 3 consecutive true nodes
    n0 = CandidateNode(node_id=0, t=0, z=1.0, y=10.0, x=10.0, score=0.95, cost_select=-2.0, cost_appear=2.0, cost_disappear=2.0)
    n1 = CandidateNode(node_id=1, t=1, z=1.1, y=10.1, x=10.0, score=0.95, cost_select=-2.0, cost_appear=2.0, cost_disappear=2.0)
    n2 = CandidateNode(node_id=2, t=2, z=1.2, y=10.2, x=10.1, score=0.95, cost_select=-2.0, cost_appear=2.0, cost_disappear=2.0)
    # 1 false positive candidate node at t=1
    n_fp = CandidateNode(node_id=3, t=1, z=5.0, y=30.0, x=30.0, score=0.2, cost_select=1.5, cost_appear=2.0, cost_disappear=2.0)

    graph.add_node(n0)
    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n_fp)

    # True edges: high probability, negative cost
    e0 = CandidateEdge(edge_id=10, source_id=0, target_id=1, dt=1, distance_um=0.2, probability=0.98, cost=-3.5)
    e1 = CandidateEdge(edge_id=11, source_id=1, target_id=2, dt=1, distance_um=0.2, probability=0.98, cost=-3.5)
    # False edge: low probability, positive cost
    e_fp = CandidateEdge(edge_id=12, source_id=0, target_id=3, dt=1, distance_um=5.0, probability=0.1, cost=2.2)

    graph.add_edge(e0)
    graph.add_edge(e1)
    graph.add_edge(e_fp)

    solver = LineageILPSolver(weight_edge=1.0, weight_node=1.0, weight_division=1.0)
    sol = solver.solve(graph)

    assert sol.status == "optimal"
    assert set(sol.selected_edge_ids) == {10, 11}
    assert 12 not in sol.selected_edge_ids
    assert set(sol.selected_node_ids) == {0, 1, 2}
    assert 3 not in sol.selected_node_ids
    assert sol.dividing_node_ids == []
    assert sol.appearing_node_ids == [0]
    assert sol.disappearing_node_ids == [2]


def test_ilp_mitosis_division_reconstruction():
    """Verify that ILP selects mitosis division when a cell splits into two daughters."""
    graph = CandidateGraph()

    # Mother cell at t=0
    n_mother = CandidateNode(node_id=1, t=0, z=5.0, y=10.0, x=10.0, score=0.95, div_prob=0.95, cost_select=-2.0, cost_division=-3.0, cost_appear=2.0, cost_disappear=2.0)
    # Two daughter cells at t=1
    d1 = CandidateNode(node_id=2, t=1, z=5.1, y=9.0, x=10.0, score=0.95, cost_select=-2.0, cost_division=2.0, cost_appear=2.0, cost_disappear=2.0)
    d2 = CandidateNode(node_id=3, t=1, z=5.1, y=11.0, x=10.0, score=0.95, cost_select=-2.0, cost_division=2.0, cost_appear=2.0, cost_disappear=2.0)

    graph.add_node(n_mother)
    graph.add_node(d1)
    graph.add_node(d2)

    e1 = CandidateEdge(edge_id=101, source_id=1, target_id=2, dt=1, distance_um=1.0, probability=0.95, cost=-3.0)
    e2 = CandidateEdge(edge_id=102, source_id=1, target_id=3, dt=1, distance_um=1.0, probability=0.95, cost=-3.0)

    graph.add_edge(e1)
    graph.add_edge(e2)

    solver = LineageILPSolver(weight_edge=1.0, weight_node=1.0, weight_division=1.0)
    sol = solver.solve(graph)

    assert sol.status == "optimal"
    assert set(sol.selected_edge_ids) == {101, 102}
    assert sol.dividing_node_ids == [1]
    assert set(sol.selected_node_ids) == {1, 2, 3}
    assert sol.appearing_node_ids == [1]
    assert set(sol.disappearing_node_ids) == {2, 3}


def test_ilp_empty_graph():
    """Verify solver handles empty candidate graph gracefully."""
    empty_graph = CandidateGraph()
    solver = LineageILPSolver()
    sol = solver.solve(empty_graph)
    assert sol.selected_edge_ids == []
    assert sol.selected_node_ids == []
    assert sol.status == "empty_graph"


def test_competition_evaluator_perfect_score():
    """Verify that perfect ground-truth reproduction achieves maximum competition metric."""
    gt_nodes = pl.DataFrame(
        {
            "node_id": [1, 2, 3, 4],
            "t": [0, 1, 2, 3],
            "z": [1.0, 1.1, 1.2, 1.3],
            "y": [2.0, 2.1, 2.2, 2.3],
            "x": [3.0, 3.1, 3.2, 3.3],
        }
    )
    gt_edges = pl.DataFrame(
        {
            "edge_id": [10, 11, 12],
            "source_id": [1, 2, 3],
            "target_id": [2, 3, 4],
        }
    )

    evaluator = CompetitionEvaluator(max_distance_um=7.0)
    metrics = evaluator.evaluate_lineages(
        pred_nodes=gt_nodes,
        pred_edges=gt_edges,
        gt_nodes=gt_nodes,
        gt_edges=gt_edges,
    )

    # For N_pred == N_gt: ratio = 1.0, J = 1.0 -> J_adj = 1.0 * (1 - 0.1 * 1.0) = 0.90
    assert np.isclose(metrics.node_jaccard, 1.0)
    assert np.isclose(metrics.pred_to_gt_ratio, 1.0)
    assert np.isclose(metrics.j_adj, 0.90)
    # Zero ground truth divisions, zero predicted -> J_div = 1.0
    assert np.isclose(metrics.j_div, 1.0)
    # Final score = 0.90 + 0.1 * 1.0 = 1.00
    assert np.isclose(metrics.final_score, 1.00)
    assert np.isclose(metrics.edge_f1, 1.0)

"""Export utilities for CandidateGraph to tracksdata and graph visualization frameworks."""

from typing import Dict, List, Tuple

import polars as pl
import rustworkx as rx
import tracksdata

from src.graph.candidate_graph import CandidateGraph


def validate_dag_temporal_monotonicity(graph: CandidateGraph) -> bool:
    """Verify that all edges in CandidateGraph strictly flow forward in time (t_dst > t_src)."""
    for edge in graph.edges.values():
        src_node = graph.nodes[edge.source_id]
        dst_node = graph.nodes[edge.target_id]
        if dst_node.t <= src_node.t:
            return False
    return True


def candidate_graph_to_rustworkx(graph: CandidateGraph) -> rx.PyDiGraph:
    """Convert CandidateGraph to rustworkx directed graph for fast topological analysis."""
    rx_graph = rx.PyDiGraph()

    # Map node_id to rustworkx node index
    node_id_to_rx_idx: Dict[int, int] = {}
    for node_id, node in graph.nodes.items():
        rx_idx = rx_graph.add_node(
            {
                "node_id": node.node_id,
                "t": node.t,
                "z": node.z,
                "y": node.y,
                "x": node.x,
                "score": node.score,
                "div_prob": node.div_prob,
            }
        )
        node_id_to_rx_idx[node_id] = rx_idx

    for edge_id, edge in graph.edges.items():
        rx_u = node_id_to_rx_idx[edge.source_id]
        rx_v = node_id_to_rx_idx[edge.target_id]
        rx_graph.add_edge(
            rx_u,
            rx_v,
            {
                "edge_id": edge.edge_id,
                "probability": edge.probability,
                "cost": edge.cost,
                "distance_um": edge.distance_um,
            },
        )

    return rx_graph


def solution_edges_to_tracksdata(
    graph: CandidateGraph,
    active_edge_ids: List[int],
) -> tracksdata.graph.IndexedRXGraph:
    """Convert a solved subset of active edges from CandidateGraph into a tracksdata IndexedRXGraph.

    Args:
        graph: Full CandidateGraph instance.
        active_edge_ids: List of edge_ids selected by the optimization solver (ILP).

    Returns:
        tracksdata.graph.IndexedRXGraph instance containing the reconstructed lineage tracks.
    """
    rx_tracks = tracksdata.graph.IndexedRXGraph()
    rx_tracks.add_node_attr_key("z", pl.Float64)
    rx_tracks.add_node_attr_key("y", pl.Float64)
    rx_tracks.add_node_attr_key("x", pl.Float64)

    active_node_ids = set()
    for eid in active_edge_ids:
        e = graph.edges[eid]
        active_node_ids.add(e.source_id)
        active_node_ids.add(e.target_id)

    sorted_active_ids = sorted(active_node_ids)
    if not sorted_active_ids:
        return rx_tracks

    node_dicts = [
        {
            "t": int(graph.nodes[nid].t),
            "z": float(graph.nodes[nid].z),
            "y": float(graph.nodes[nid].y),
            "x": float(graph.nodes[nid].x),
        }
        for nid in sorted_active_ids
    ]
    rx_tracks.bulk_add_nodes(node_dicts, indices=sorted_active_ids)

    edge_dicts = [
        {
            "source_id": int(graph.edges[eid].source_id),
            "target_id": int(graph.edges[eid].target_id),
        }
        for eid in active_edge_ids
    ]
    rx_tracks.bulk_add_edges(edge_dicts)

    return rx_tracks

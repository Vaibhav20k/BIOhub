"""Candidate Spatio-Temporal Tracking Graph structure and cost formulation for global lineage optimization."""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import polars as pl
import torch
import torch.nn as nn

from src.data.zarr_reader import DatasetVolume, DEFAULT_SCALE
from src.representation.node_embedding import CellNodeEmbedding, extract_node_embeddings_for_dataset
from src.tracking.candidate_edges import CandidateEdgeBatch, CandidateEdgeBuilder
from src.tracking.transformer import SpatioTemporalTracker


@dataclass
class CandidateNode:
    """Represents a candidate cell detection in the spatio-temporal graph."""

    node_id: int
    t: int
    z: float
    y: float
    x: float
    score: float  # Detection confidence tau in [0, 1]
    div_prob: float = 0.0  # Mitosis probability in [0, 1]
    cost_select: float = 0.0  # Node selection cost (negative log-odds of detection score)
    cost_appear: float = 0.0  # Cost for track initiation at this node
    cost_disappear: float = 0.0  # Cost for track termination at this node
    cost_division: float = 0.0  # Cost for dividing into 2 daughter cells


@dataclass
class CandidateEdge:
    """Represents a candidate association edge between two cells across time."""

    edge_id: int
    source_id: int
    target_id: int
    dt: int  # Frame step difference: 1 for consecutive, 2 for skip
    distance_um: float  # Physical Euclidean distance
    probability: float  # Predicted transition probability from Transformer
    cost: float  # Edge selection cost (negative log-odds of transition probability)


@dataclass
class CandidateGraph:
    """Directed Acyclic Graph (DAG) containing all candidate nodes, transition edges, and costs."""

    nodes: Dict[int, CandidateNode] = field(default_factory=dict)
    edges: Dict[int, CandidateEdge] = field(default_factory=dict)
    # Adjacency indices for fast graph traversal and constraint compilation
    outgoing_edges: Dict[int, List[int]] = field(default_factory=dict)  # node_id -> list of edge_ids
    incoming_edges: Dict[int, List[int]] = field(default_factory=dict)  # node_id -> list of edge_ids
    nodes_by_time: Dict[int, List[int]] = field(default_factory=dict)   # t -> list of node_ids

    def add_node(self, node: CandidateNode) -> None:
        self.nodes[node.node_id] = node
        if node.node_id not in self.outgoing_edges:
            self.outgoing_edges[node.node_id] = []
        if node.node_id not in self.incoming_edges:
            self.incoming_edges[node.node_id] = []
        if node.t not in self.nodes_by_time:
            self.nodes_by_time[node.t] = []
        self.nodes_by_time[node.t].append(node.node_id)

    def add_edge(self, edge: CandidateEdge) -> None:
        self.edges[edge.edge_id] = edge
        self.outgoing_edges[edge.source_id].append(edge.edge_id)
        self.incoming_edges[edge.target_id].append(edge.edge_id)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def get_summary(self) -> Dict[str, Union[int, float]]:
        """Compute structural diagnostics of the candidate graph."""
        n_nodes = self.num_nodes
        n_edges = self.num_edges
        timepoints = sorted(self.nodes_by_time.keys()) if self.nodes_by_time else []
        avg_out_degree = (n_edges / max(1, n_nodes)) if n_nodes > 0 else 0.0

        div_candidates = sum(1 for n in self.nodes.values() if n.div_prob >= 0.5)

        return {
            "num_nodes": n_nodes,
            "num_edges": n_edges,
            "num_timepoints": len(timepoints),
            "time_range": (min(timepoints), max(timepoints)) if timepoints else (0, 0),
            "avg_candidates_per_node": round(avg_out_degree, 2),
            "dividing_candidates": div_candidates,
        }


def probability_to_cost(prob: float, eps: float = 1e-4) -> float:
    """Convert probability in [0, 1] to linear programming cost via log-odds.
    
    When p > 0.5, log-odds > 0 -> cost < 0 (beneficial reward to select).
    When p < 0.5, log-odds < 0 -> cost > 0 (penalty to select).
    """
    p_clamped = max(eps, min(1.0 - eps, float(prob)))
    return -math.log(p_clamped / (1.0 - p_clamped))


class CandidateGraphBuilder:
    """Constructs optimization-ready candidate DAG from detected cells and deep learning predictions."""

    def __init__(
        self,
        max_distance_um: float = 7.0,  # Competition matching threshold
        scale: Tuple[float, float, float] = DEFAULT_SCALE,
        p_appear: float = 0.05,       # Prior probability of new track appearance
        p_disappear: float = 0.05,    # Prior probability of track termination
        allow_frame_skips: bool = False,
        skip_penalty: float = 2.0,    # Additional cost penalty for 2-frame skips
    ):
        """
        Args:
            max_distance_um: Maximum spatial search cutoff in micrometers.
            scale: Physical scale (s_z, s_y, s_x).
            p_appear: Default prior probability of a track starting at an unlinked node.
            p_disappear: Default prior probability of a track ending at an unlinked node.
            allow_frame_skips: Whether to build bridging edges across t -> t+2.
            skip_penalty: Cost penalty added to frame-skip edges.
        """
        self.max_distance_um = max_distance_um
        self.scale = scale
        self.p_appear = p_appear
        self.p_disappear = p_disappear
        self.allow_frame_skips = allow_frame_skips
        self.skip_penalty = skip_penalty

        self.cost_appear = -math.log(p_appear / (1.0 - p_appear))
        self.cost_disappear = -math.log(p_disappear / (1.0 - p_disappear))
        self.edge_builder = CandidateEdgeBuilder(max_distance_um=max_distance_um, scale=scale)

    def build_candidate_graph(
        self,
        nodes_df: pl.DataFrame,
        tracker: SpatioTemporalTracker,
        node_embeddings: torch.Tensor,
        device: torch.device = torch.device("cpu"),
    ) -> CandidateGraph:
        """Assemble full candidate graph from detections, embeddings, and tracker predictions.

        Args:
            nodes_df: Polars DataFrame with columns: ['t', 'node_id', 'z', 'y', 'x', 'score'].
                      Must be sorted or matched row-by-row with node_embeddings.
            tracker: Trained SpatioTemporalTracker model.
            node_embeddings: (N, D) tensor of unified node embeddings aligned with nodes_df.
            device: Compute device.

        Returns:
            CandidateGraph instance populated with nodes, scored edges, and ILP costs.
        """
        tracker.eval()
        tracker.to(device)

        graph = CandidateGraph()
        n_total_nodes = nodes_df.height

        if n_total_nodes == 0:
            return graph

        # Ensure node_id is int and unique
        node_records = nodes_df.to_dicts()
        timepoints = sorted(nodes_df["t"].unique().to_list())

        # Map global node_id to DataFrame row index
        id_to_row_idx = {int(r["node_id"]): i for i, r in enumerate(node_records)}

        # 1. Precompute division probabilities for all nodes using the Tracker's division head
        with torch.no_grad():
            embs_device = node_embeddings.to(device)
            div_logits = tracker.division_head(embs_device).squeeze(-1)
            div_probs = torch.sigmoid(div_logits).cpu().numpy().tolist()

        # 2. Add all candidate nodes to graph
        for i, rec in enumerate(node_records):
            u_id = int(rec["node_id"])
            score = float(rec.get("score", 0.9))
            div_p = float(div_probs[i])

            node = CandidateNode(
                node_id=u_id,
                t=int(rec["t"]),
                z=float(rec["z"]),
                y=float(rec["y"]),
                x=float(rec["x"]),
                score=score,
                div_prob=div_p,
                cost_select=probability_to_cost(score),
                cost_appear=self.cost_appear,
                cost_disappear=self.cost_disappear,
                cost_division=probability_to_cost(div_p),
            )
            graph.add_node(node)

        # 3. Construct and score candidate edges between consecutive frames
        edge_id_counter = 0

        with torch.no_grad():
            for t_idx in range(len(timepoints) - 1):
                t_src = timepoints[t_idx]
                t_dst = timepoints[t_idx + 1]

                # Consecutive transitions: dt = 1
                if t_dst == t_src + 1:
                    src_nodes_df = nodes_df.filter(pl.col("t") == t_src)
                    dst_nodes_df = nodes_df.filter(pl.col("t") == t_dst)

                    cand_batch = self.edge_builder.build_frame_pair_candidates(
                        src_nodes=src_nodes_df,
                        dst_nodes=dst_nodes_df,
                    )

                    if cand_batch.src_indices.shape[0] > 0:
                        src_sub_embs = torch.stack(
                            [node_embeddings[id_to_row_idx[nid]] for nid in src_nodes_df["node_id"].to_list()]
                        ).to(device)
                        dst_sub_embs = torch.stack(
                            [node_embeddings[id_to_row_idx[nid]] for nid in dst_nodes_df["node_id"].to_list()]
                        ).to(device)

                        pred = tracker(src_sub_embs, dst_sub_embs, cand_batch)
                        probs = pred.edge_probs.cpu().numpy()
                        dists = cand_batch.distances_um.squeeze(-1).numpy()

                        for u, v, p, d in zip(cand_batch.src_node_ids, cand_batch.dst_node_ids, probs, dists):
                            edge_cost = probability_to_cost(float(p))
                            edge = CandidateEdge(
                                edge_id=edge_id_counter,
                                source_id=int(u),
                                target_id=int(v),
                                dt=1,
                                distance_um=float(d),
                                probability=float(p),
                                cost=edge_cost,
                            )
                            graph.add_edge(edge)
                            edge_id_counter += 1

                # Frame skip transitions: dt = 2 (optional)
                if self.allow_frame_skips and t_idx + 2 < len(timepoints):
                    t_skip = timepoints[t_idx + 2]
                    if t_skip == t_src + 2:
                        src_nodes_df = nodes_df.filter(pl.col("t") == t_src)
                        dst_nodes_df = nodes_df.filter(pl.col("t") == t_skip)

                        cand_batch = self.edge_builder.build_frame_pair_candidates(
                            src_nodes=src_nodes_df,
                            dst_nodes=dst_nodes_df,
                        )

                        if cand_batch.src_indices.shape[0] > 0:
                            src_sub_embs = torch.stack(
                                [node_embeddings[id_to_row_idx[nid]] for nid in src_nodes_df["node_id"].to_list()]
                            ).to(device)
                            dst_sub_embs = torch.stack(
                                [node_embeddings[id_to_row_idx[nid]] for nid in dst_nodes_df["node_id"].to_list()]
                            ).to(device)

                            pred = tracker(src_sub_embs, dst_sub_embs, cand_batch)
                            probs = pred.edge_probs.cpu().numpy()
                            dists = cand_batch.distances_um.squeeze(-1).numpy()

                            for u, v, p, d in zip(cand_batch.src_node_ids, cand_batch.dst_node_ids, probs, dists):
                                edge_cost = probability_to_cost(float(p)) + self.skip_penalty
                                edge = CandidateEdge(
                                    edge_id=edge_id_counter,
                                    source_id=int(u),
                                    target_id=int(v),
                                    dt=2,
                                    distance_um=float(d),
                                    probability=float(p),
                                    cost=edge_cost,
                                )
                                graph.add_edge(edge)
                                edge_id_counter += 1

        return graph


def candidate_graph_to_dataframes(
    graph: CandidateGraph,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Export candidate graph to Polars DataFrames for analysis or inspection.

    Returns:
        nodes_df: Columns ['node_id', 't', 'z', 'y', 'x', 'score', 'div_prob', 'cost_select', 'cost_division']
        edges_df: Columns ['edge_id', 'source_id', 'target_id', 'dt', 'distance_um', 'probability', 'cost']
    """
    node_rows = [
        {
            "node_id": n.node_id,
            "t": n.t,
            "z": n.z,
            "y": n.y,
            "x": n.x,
            "score": n.score,
            "div_prob": n.div_prob,
            "cost_select": n.cost_select,
            "cost_division": n.cost_division,
        }
        for n in graph.nodes.values()
    ]

    edge_rows = [
        {
            "edge_id": e.edge_id,
            "source_id": e.source_id,
            "target_id": e.target_id,
            "dt": e.dt,
            "distance_um": e.distance_um,
            "probability": e.probability,
            "cost": e.cost,
        }
        for e in graph.edges.values()
    ]

    nodes_df = pl.DataFrame(node_rows) if node_rows else pl.DataFrame(
        schema={"node_id": pl.Int64, "t": pl.Int64, "z": pl.Float64, "y": pl.Float64, "x": pl.Float64, "score": pl.Float64, "div_prob": pl.Float64, "cost_select": pl.Float64, "cost_division": pl.Float64}
    )
    edges_df = pl.DataFrame(edge_rows) if edge_rows else pl.DataFrame(
        schema={"edge_id": pl.Int64, "source_id": pl.Int64, "target_id": pl.Int64, "dt": pl.Int64, "distance_um": pl.Float64, "probability": pl.Float64, "cost": pl.Float64}
    )

    return nodes_df, edges_df

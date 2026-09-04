"""Candidate spatio-temporal edge builder using KD-Tree physical distance constraints."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import polars as pl
from scipy.spatial import KDTree
import torch

from src.data.zarr_reader import DEFAULT_SCALE


@dataclass
class CandidateEdgeBatch:
    """Container for candidate edges between two timepoints."""

    src_node_ids: List[int]
    dst_node_ids: List[int]
    src_indices: torch.Tensor  # (E,) LongTensor of local source indices [0 ... N_src - 1]
    dst_indices: torch.Tensor  # (E,) LongTensor of local target indices [0 ... N_dst - 1]
    delta_zyx: torch.Tensor  # (E, 3) FloatTensor of voxel displacements
    distances_um: torch.Tensor  # (E, 1) FloatTensor of physical Euclidean distances
    edge_labels: Optional[torch.Tensor] = None  # (E,) FloatTensor: 1.0 for true transition, 0.0 otherwise
    division_labels: Optional[torch.Tensor] = None  # (N_src,) FloatTensor: 1.0 if src node divides, 0.0 otherwise


class CandidateEdgeBuilder:
    """Builds candidate association edges between consecutive frames within physical search radius."""

    def __init__(
        self,
        max_distance_um: float = 7.0,  # Official competition cutoff
        scale: Tuple[float, float, float] = DEFAULT_SCALE,
    ):
        """
        Args:
            max_distance_um: Maximum allowable Euclidean distance in micrometers for candidate edges.
            scale: Physical scale (s_z, s_y, s_x) in micrometers per voxel.
        """
        self.max_distance_um = max_distance_um
        self.scale = scale
        self.scale_arr = np.array(scale, dtype=np.float32)

    def build_frame_pair_candidates(
        self,
        src_nodes: pl.DataFrame,
        dst_nodes: pl.DataFrame,
        gt_edges: Optional[Set[Tuple[int, int]]] = None,
        dividing_nodes: Optional[Set[int]] = None,
    ) -> CandidateEdgeBatch:
        """Construct candidate edges between nodes at frame t and nodes at frame t+1.

        Args:
            src_nodes: Polars DataFrame of nodes at frame t with columns ['node_id', 'z', 'y', 'x'].
            dst_nodes: Polars DataFrame of nodes at frame t+1 with columns ['node_id', 'z', 'y', 'x'].
            gt_edges: Optional set of ground-truth (src_node_id, dst_node_id) edge tuples.
            dividing_nodes: Optional set of src_node_ids that undergo mitosis at frame t.

        Returns:
            CandidateEdgeBatch containing candidate graph connectivity and optional supervision labels.
        """
        n_src = src_nodes.height
        n_dst = dst_nodes.height

        # Handle empty cases
        if n_src == 0 or n_dst == 0:
            return CandidateEdgeBatch(
                src_node_ids=[],
                dst_node_ids=[],
                src_indices=torch.empty((0,), dtype=torch.long),
                dst_indices=torch.empty((0,), dtype=torch.long),
                delta_zyx=torch.empty((0, 3), dtype=torch.float32),
                distances_um=torch.empty((0, 1), dtype=torch.float32),
                edge_labels=torch.empty((0,), dtype=torch.float32) if gt_edges is not None else None,
                division_labels=torch.zeros((n_src,), dtype=torch.float32) if dividing_nodes is not None else None,
            )

        src_ids = src_nodes["node_id"].to_list()
        dst_ids = dst_nodes["node_id"].to_list()

        src_coords_zyx = src_nodes.select(["z", "y", "x"]).to_numpy().astype(np.float32)
        dst_coords_zyx = dst_nodes.select(["z", "y", "x"]).to_numpy().astype(np.float32)

        # Physical coordinates (N, 3) in micrometers
        src_phys = src_coords_zyx * self.scale_arr
        dst_phys = dst_coords_zyx * self.scale_arr

        # Build KD-Tree on target coordinates at frame t+1
        kdtree = KDTree(dst_phys)

        # Query all target neighbors within max_distance_um for each source node
        neighbor_indices = kdtree.query_ball_point(src_phys, r=self.max_distance_um)

        edge_src_ids: List[int] = []
        edge_dst_ids: List[int] = []
        edge_src_idx: List[int] = []
        edge_dst_idx: List[int] = []
        edge_deltas: List[List[float]] = []
        edge_dists: List[float] = []

        for u_idx, targets in enumerate(neighbor_indices):
            u_id = src_ids[u_idx]
            u_zyx = src_coords_zyx[u_idx]
            u_phys = src_phys[u_idx]

            for v_idx in targets:
                v_id = dst_ids[v_idx]
                v_zyx = dst_coords_zyx[v_idx]
                v_phys = dst_phys[v_idx]

                dist = float(np.linalg.norm(v_phys - u_phys))
                delta = [float(v_zyx[0] - u_zyx[0]), float(v_zyx[1] - u_zyx[1]), float(v_zyx[2] - u_zyx[2])]

                edge_src_ids.append(u_id)
                edge_dst_ids.append(v_id)
                edge_src_idx.append(u_idx)
                edge_dst_idx.append(v_idx)
                edge_deltas.append(delta)
                edge_dists.append(dist)

        num_edges = len(edge_src_idx)

        src_idx_t = torch.tensor(edge_src_idx, dtype=torch.long)
        dst_idx_t = torch.tensor(edge_dst_idx, dtype=torch.long)
        delta_t = torch.tensor(edge_deltas, dtype=torch.float32).reshape(num_edges, 3)
        dist_t = torch.tensor(edge_dists, dtype=torch.float32).reshape(num_edges, 1)

        # Build ground-truth edge labels if provided
        edge_labels_t: Optional[torch.Tensor] = None
        if gt_edges is not None:
            labels = [1.0 if (u, v) in gt_edges else 0.0 for u, v in zip(edge_src_ids, edge_dst_ids)]
            edge_labels_t = torch.tensor(labels, dtype=torch.float32)

        # Build division labels if provided
        div_labels_t: Optional[torch.Tensor] = None
        if dividing_nodes is not None:
            div_list = [1.0 if u in dividing_nodes else 0.0 for u in src_ids]
            div_labels_t = torch.tensor(div_list, dtype=torch.float32)

        return CandidateEdgeBatch(
            src_node_ids=edge_src_ids,
            dst_node_ids=edge_dst_ids,
            src_indices=src_idx_t,
            dst_indices=dst_idx_t,
            delta_zyx=delta_t,
            distances_um=dist_t,
            edge_labels=edge_labels_t,
            division_labels=div_labels_t,
        )

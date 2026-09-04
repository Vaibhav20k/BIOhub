"""Official competition tracking evaluation metrics: J_adj + 0.1 * J_div."""

from dataclasses import asdict, dataclass
import logging
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree
import tracksdata

from src.data.zarr_reader import DEFAULT_SCALE
from src.evaluation.node_matching import BipartiteNodeEvaluator, NodeEvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class CompetitionTrackingMetrics:
    """Quantitative performance against Kaggle Biohub competition metrics."""

    # Node detection metrics
    node_tp: int
    node_fp: int
    node_fn: int
    node_precision: float
    node_recall: float
    node_f1: float
    node_jaccard: float
    pred_to_gt_ratio: float
    j_adj: float  # max(0, J * (1 - 0.1 * N_pred / N_gt))

    # Mitosis / Division metrics
    gt_divisions: int
    pred_divisions: int
    div_tp: int
    div_fp: int
    div_fn: int
    j_div: float

    # Combined Official Competition Score
    final_score: float  # J_adj + 0.1 * J_div

    # Edge continuity metrics
    edge_tp: int
    edge_fp: int
    edge_fn: int
    edge_precision: float
    edge_recall: float
    edge_f1: float

    def to_dict(self) -> Dict[str, Union[int, float]]:
        return asdict(self)


class CompetitionEvaluator:
    """Computes official Kaggle Biohub cell tracking score: J_adj + 0.1 * J_div."""

    def __init__(
        self,
        max_distance_um: float = 7.0,  # Official competition cutoff
        scale: Tuple[float, float, float] = DEFAULT_SCALE,
    ):
        self.max_distance_um = max_distance_um
        self.scale = scale
        self.scale_arr = np.array(scale, dtype=np.float32)
        self.node_evaluator = BipartiteNodeEvaluator(max_distance=max_distance_um, scale=scale)

    def evaluate_lineages(
        self,
        pred_nodes: pl.DataFrame,
        pred_edges: pl.DataFrame,
        gt_nodes: pl.DataFrame,
        gt_edges: pl.DataFrame,
    ) -> CompetitionTrackingMetrics:
        """Evaluate full reconstructed lineage graph against ground truth annotations.

        Args:
            pred_nodes: Polars DataFrame ['node_id', 't', 'z', 'y', 'x'].
            pred_edges: Polars DataFrame ['edge_id', 'source_id', 'target_id'].
            gt_nodes: Polars DataFrame ['node_id', 't', 'z', 'y', 'x'].
            gt_edges: Polars DataFrame ['edge_id', 'source_id', 'target_id'].

        Returns:
            CompetitionTrackingMetrics with J_adj, J_div, and final combined score.
        """
        # ----------------------------------------------------
        # 1. Evaluate Nodes & J_adj
        # ----------------------------------------------------
        node_res = self.node_evaluator.evaluate(pred_nodes, gt_nodes)

        node_tp = node_res.tp
        node_fp = node_res.fp
        node_fn = node_res.fn
        node_jaccard = node_tp / max(1, node_tp + node_fp + node_fn)
        ratio = node_res.total_node_ratio

        # Official penalty: J_adj = max(0, J * (1 - 0.1 * ratio))
        penalty_factor = max(0.0, 1.0 - 0.1 * ratio)
        j_adj = float(node_jaccard * penalty_factor)

        # ----------------------------------------------------
        # 2. Build Bipartite Mapping (pred_node_id -> gt_node_id)
        # ----------------------------------------------------
        # Used to verify edge preservation across time
        pred_to_gt_map: Dict[int, int] = {}
        timepoints = sorted(gt_nodes["t"].unique().to_list())

        for t in timepoints:
            p_t = pred_nodes.filter(pl.col("t") == t)
            g_t = gt_nodes.filter(pl.col("t") == t)
            if p_t.height == 0 or g_t.height == 0:
                continue

            p_ids = p_t["node_id"].to_list()
            g_ids = g_t["node_id"].to_list()

            p_coords = p_t.select(["z", "y", "x"]).to_numpy().astype(np.float32) * self.scale_arr
            g_coords = g_t.select(["z", "y", "x"]).to_numpy().astype(np.float32) * self.scale_arr

            diff = p_coords[:, np.newaxis, :] - g_coords[np.newaxis, :, :]
            dist_mat = np.linalg.norm(diff, axis=-1)

            row_ind, col_ind = linear_sum_assignment(dist_mat)
            for r, c in zip(row_ind, col_ind):
                if dist_mat[r, c] <= self.max_distance_um:
                    pred_to_gt_map[p_ids[r]] = g_ids[c]

        # ----------------------------------------------------
        # 3. Evaluate Divisions (Mitosis Events)
        # ----------------------------------------------------
        # Ground truth divisions (out-degree >= 2)
        gt_div_nodes_set: Set[int] = set()
        if gt_edges.height > 0:
            gt_src_counts = gt_edges["source_id"].value_counts()
            gt_div_nodes_set = set(gt_src_counts.filter(pl.col("count") >= 2)["source_id"].to_list())

        # Predicted divisions (out-degree >= 2)
        pred_div_nodes_set: Set[int] = set()
        if pred_edges.height > 0:
            pred_src_counts = pred_edges["source_id"].value_counts()
            pred_div_nodes_set = set(pred_src_counts.filter(pl.col("count") >= 2)["source_id"].to_list())

        gt_div_count = len(gt_div_nodes_set)
        pred_div_count = len(pred_div_nodes_set)

        div_tp = 0
        div_fp = 0
        div_fn = 0

        if gt_div_count == 0 and pred_div_count == 0:
            j_div = 1.0
        elif gt_div_count == 0 and pred_div_count > 0:
            div_fp = pred_div_count
            j_div = 0.0
        else:
            # Match predicted dividing nodes to ground truth dividing nodes via bipartite mapping
            matched_gt_divs = set()
            for p_div_id in pred_div_nodes_set:
                mapped_gt_id = pred_to_gt_map.get(p_div_id)
                if mapped_gt_id is not None and mapped_gt_id in gt_div_nodes_set:
                    div_tp += 1
                    matched_gt_divs.add(mapped_gt_id)
                else:
                    div_fp += 1

            div_fn = gt_div_count - len(matched_gt_divs)
            j_div = div_tp / max(1, div_tp + div_fp + div_fn)

        # ----------------------------------------------------
        # 4. Final Official Competition Score
        # ----------------------------------------------------
        final_score = float(j_adj + 0.1 * j_div)

        # ----------------------------------------------------
        # 5. Evaluate Trajectory Edges
        # ----------------------------------------------------
        gt_edge_pairs = (
            set(zip(gt_edges["source_id"].to_list(), gt_edges["target_id"].to_list()))
            if gt_edges.height > 0
            else set()
        )

        edge_tp = 0
        edge_fp = 0

        if pred_edges.height > 0:
            for u_pred, v_pred in zip(pred_edges["source_id"].to_list(), pred_edges["target_id"].to_list()):
                u_gt = pred_to_gt_map.get(u_pred)
                v_gt = pred_to_gt_map.get(v_pred)
                if u_gt is not None and v_gt is not None and (u_gt, v_gt) in gt_edge_pairs:
                    edge_tp += 1
                else:
                    edge_fp += 1

        edge_fn = max(0, len(gt_edge_pairs) - edge_tp)
        edge_prec = edge_tp / max(1, edge_tp + edge_fp)
        edge_rec = edge_tp / max(1, edge_tp + edge_fn)
        edge_f1 = (2.0 * edge_prec * edge_rec) / max(1e-6, edge_prec + edge_rec)

        return CompetitionTrackingMetrics(
            node_tp=node_tp,
            node_fp=node_fp,
            node_fn=node_fn,
            node_precision=node_res.precision,
            node_recall=node_res.recall,
            node_f1=node_res.f1,
            node_jaccard=float(node_jaccard),
            pred_to_gt_ratio=float(ratio),
            j_adj=j_adj,
            gt_divisions=gt_div_count,
            pred_divisions=pred_div_count,
            div_tp=div_tp,
            div_fp=div_fp,
            div_fn=div_fn,
            j_div=float(j_div),
            final_score=final_score,
            edge_tp=edge_tp,
            edge_fp=edge_fp,
            edge_fn=edge_fn,
            edge_precision=float(edge_prec),
            edge_recall=float(edge_rec),
            edge_f1=float(edge_f1),
        )

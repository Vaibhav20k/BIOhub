"""Hungarian bipartite node matching evaluator using competition physical distance cutoff (7.0 µm)."""

from dataclasses import asdict, dataclass
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn

from src.data.zarr_reader import DatasetVolume, DEFAULT_SCALE
from src.detection.peak_detector import PeakDetector3D

logger = logging.getLogger(__name__)


@dataclass
class NodeEvaluationResult:
    """Quantitative performance metrics for cell detection matching."""

    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    total_node_ratio: float
    mean_error_um: float
    median_error_um: float
    p90_error_um: float
    max_error_um: float
    num_pred_nodes: int
    num_gt_nodes: int
    distance_errors: List[float]

    def to_dict(self) -> Dict[str, Union[int, float, List[float]]]:
        return asdict(self)


class BipartiteNodeEvaluator:
    """Evaluates spatial cell detections against ground-truth annotations via Hungarian matching."""

    def __init__(
        self,
        max_distance: float = 7.0,  # Official competition cutoff in micrometers
        scale: Tuple[float, float, float] = DEFAULT_SCALE,
    ):
        """
        Args:
            max_distance: Maximum allowable Euclidean distance in micrometers for a valid match.
            scale: Physical scale (s_z, s_y, s_x) in micrometers.
        """
        self.max_distance = max_distance
        self.scale = scale

    def evaluate(
        self,
        pred_nodes: pl.DataFrame,
        gt_nodes: pl.DataFrame,
        timepoints: Optional[List[int]] = None,
    ) -> NodeEvaluationResult:
        """Match predicted nodes to ground truth nodes independently per time frame.

        Args:
            pred_nodes: Polars DataFrame with columns ['t', 'z', 'y', 'x'] and optional ['score'].
            gt_nodes: Polars DataFrame with columns ['t', 'z', 'y', 'x'].
            timepoints: Optional list of timepoints to restrict evaluation to.

        Returns:
            NodeEvaluationResult instance.
        """
        scale_arr = np.array(self.scale, dtype=np.float32)

        if timepoints is None:
            timepoints = sorted(gt_nodes["t"].unique().to_list())

        total_tp = 0
        total_fp = 0
        total_fn = 0
        all_distance_errors: List[float] = []
        total_pred_evaluated = 0
        total_gt_evaluated = 0

        for t in timepoints:
            p_t = pred_nodes.filter(pl.col("t") == t)
            g_t = gt_nodes.filter(pl.col("t") == t)

            n_p = p_t.height
            n_g = g_t.height

            total_pred_evaluated += n_p
            total_gt_evaluated += n_g

            if n_g == 0:
                total_fp += n_p
                continue

            if n_p == 0:
                total_fn += n_g
                continue

            # Physical coordinates in micrometers: (N, 3)
            p_coords = p_t.select(["z", "y", "x"]).to_numpy().astype(np.float32) * scale_arr
            g_coords = g_t.select(["z", "y", "x"]).to_numpy().astype(np.float32) * scale_arr

            # Euclidean cost matrix (N, M)
            diff = p_coords[:, np.newaxis, :] - g_coords[np.newaxis, :, :]  # (N, M, 3)
            cost_matrix = np.linalg.norm(diff, axis=-1)  # (N, M)

            # Global optimal bipartite matching (Hungarian algorithm)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            matched_pred = set()
            matched_gt = set()

            for r, c in zip(row_ind, col_ind):
                dist = float(cost_matrix[r, c])
                if dist <= self.max_distance:
                    total_tp += 1
                    matched_pred.add(r)
                    matched_gt.add(c)
                    all_distance_errors.append(dist)
                else:
                    total_fp += 1
                    total_fn += 1

            # Unassigned predictions are false positives
            unassigned_p = n_p - len(row_ind)
            total_fp += max(0, unassigned_p)

            # Unassigned ground truths are false negatives
            unassigned_g = n_g - len(col_ind)
            total_fn += max(0, unassigned_g)

        prec = total_tp / max(1, total_tp + total_fp)
        rec = total_tp / max(1, total_tp + total_fn)
        f1 = (2.0 * prec * rec) / max(1e-6, prec + rec)
        node_ratio = total_pred_evaluated / max(1, total_gt_evaluated)

        if all_distance_errors:
            err_arr = np.array(all_distance_errors)
            mean_err = float(np.mean(err_arr))
            med_err = float(np.median(err_arr))
            p90_err = float(np.percentile(err_arr, 90))
            max_err = float(np.max(err_arr))
        else:
            mean_err = med_err = p90_err = max_err = float("nan")

        return NodeEvaluationResult(
            tp=total_tp,
            fp=total_fp,
            fn=total_fn,
            precision=float(prec),
            recall=float(rec),
            f1=float(f1),
            total_node_ratio=float(node_ratio),
            mean_error_um=mean_err,
            median_error_um=med_err,
            p90_error_um=p90_err,
            max_error_um=max_err,
            num_pred_nodes=total_pred_evaluated,
            num_gt_nodes=total_gt_evaluated,
            distance_errors=all_distance_errors,
        )

    def threshold_sweep(
        self,
        model: nn.Module,
        dataset: DatasetVolume,
        device: torch.device,
        thresholds: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        timepoints: Optional[List[int]] = None,
    ) -> pl.DataFrame:
        """Sweep detection confidence thresholds to determine the optimal operating point.

        Returns:
            Polars DataFrame summarizing Precision, Recall, F1, and Node Ratio per threshold.
        """
        assert dataset.tracks is not None, "Ground truth tracks required for threshold sweep"
        gt_nodes = dataset.tracks.node_attrs()

        if timepoints is None:
            timepoints = sorted(gt_nodes["t"].unique().to_list())

        detector = PeakDetector3D(scale=self.scale)

        # Precompute heatmaps once to avoid redundant model passes during sweep
        model.eval()
        cached_heatmaps: Dict[int, np.ndarray] = {}
        with torch.no_grad():
            for t in timepoints:
                vol_crop = dataset.read_spatial_crop(
                    slice(t, t + 1), slice(None), slice(None), slice(None)
                )
                from src.data.preprocessing import normalize_intensity
                vol_norm = normalize_intensity(vol_crop, quantiles=dataset.quantiles)
                inp = torch.from_numpy(vol_norm).unsqueeze(0).to(device)
                from torch.amp import autocast
                with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    pred_hm, _ = model(inp)
                cached_heatmaps[t] = pred_hm[0, 0].cpu().numpy().astype(np.float32)

        sweep_rows: List[Dict[str, Union[float, int]]] = []

        for thresh in thresholds:
            pred_records: List[Dict[str, Union[int, float]]] = []
            node_id_cnt = 0

            for t in timepoints:
                hm = cached_heatmaps[t]
                peaks, scores = detector.extract_peaks_3d(hm, threshold=thresh)
                for (z, y, x), score in zip(peaks, scores):
                    pred_records.append(
                        {
                            "t": int(t),
                            "node_id": int(node_id_cnt),
                            "z": float(z),
                            "y": float(y),
                            "x": float(x),
                            "score": float(score),
                        }
                    )
                    node_id_cnt += 1

            pred_df = pl.DataFrame(pred_records) if pred_records else pl.DataFrame(
                schema={"t": pl.Int64, "node_id": pl.Int64, "z": pl.Float64, "y": pl.Float64, "x": pl.Float64, "score": pl.Float64}
            )

            result = self.evaluate(pred_df, gt_nodes, timepoints=timepoints)

            sweep_rows.append(
                {
                    "threshold": float(thresh),
                    "precision": round(result.precision, 4),
                    "recall": round(result.recall, 4),
                    "f1": round(result.f1, 4),
                    "node_ratio": round(result.total_node_ratio, 4),
                    "mean_error_um": round(result.mean_error_um, 3) if not np.isnan(result.mean_error_um) else -1.0,
                    "tp": result.tp,
                    "fp": result.fp,
                    "fn": result.fn,
                }
            )

        return pl.DataFrame(sweep_rows)


def evaluate_node_detections(
    pred_nodes: pl.DataFrame,
    gt_nodes: pl.DataFrame,
    scale: Tuple[float, float, float] = DEFAULT_SCALE,
    max_distance: float = 7.0,
) -> NodeEvaluationResult:
    """Convenience function to evaluate node detections."""
    evaluator = BipartiteNodeEvaluator(max_distance=max_distance, scale=scale)
    return evaluator.evaluate(pred_nodes, gt_nodes)

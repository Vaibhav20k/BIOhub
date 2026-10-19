"""Fast per-epoch detector check on the permanent validation anchors.

Runs the detector over every annotated frame, extracts peaks at the production threshold and reports node matching
(TP/FP/FN, recall, precision, adjusted Jaccard) plus heatmap calibration: activation at ground-truth centres versus
background. This is the detector-only signal used to pick the best epoch; the full lineage score comes from
``HeldOutEvaluator``. Extracted from ``scripts/run_chunk3_training.py`` so later chunk scripts share it.
"""

from typing import Dict, List, Tuple

import numpy as np
import polars as pl
import torch

from src.data.preprocessing import normalize_intensity
from src.data.zarr_reader import DatasetVolume
from src.detection.peak_detector import PeakDetector3D
from src.evaluation.node_matching import BipartiteNodeEvaluator


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if len(values) else 0.0


@torch.no_grad()
def evaluate_anchor_detection(
    detector: torch.nn.Module,
    val_datasets: List[DatasetVolume],
    device: torch.device,
    detection_threshold: float = 0.25,
    max_peaks_per_frame: int = 150,
    max_distance_um: float = 7.0,
    bg_samples_per_frame: int = 1000,
) -> Tuple[Dict[str, Dict[str, float]], float]:
    """Returns ({sequence: metrics}, macro adjusted Jaccard over the sequences)."""
    detector.eval()
    peak_detector = PeakDetector3D(threshold=detection_threshold, max_peaks_per_frame=max_peaks_per_frame)
    results: Dict[str, Dict[str, float]] = {}

    for ds in val_datasets:
        gt_nodes = ds.tracks.node_attrs()
        rng = np.random.RandomState(42)
        detected, gt_vals, bg_vals = [], [], []

        for t in sorted(gt_nodes["t"].unique().to_list()):
            vol = ds.read_spatial_crop(slice(t, t + 1), slice(None), slice(None), slice(None))
            inp = torch.from_numpy(normalize_intensity(vol, quantiles=ds.quantiles)).unsqueeze(0).to(device)
            pred_hm, _ = detector(inp)
            hm = pred_hm[0, 0].float().cpu().numpy()

            for row in gt_nodes.filter(pl.col("t") == t).iter_rows(named=True):
                z = int(np.clip(round(row["z"]), 0, hm.shape[0] - 1))
                y = int(np.clip(round(row["y"]), 0, hm.shape[1] - 1))
                x = int(np.clip(round(row["x"]), 0, hm.shape[2] - 1))
                gt_vals.append(float(hm[z, y, x]))
            flat = hm.ravel()
            bg_vals.extend(flat[rng.choice(flat.shape[0], size=min(bg_samples_per_frame, flat.shape[0]), replace=False)])

            peaks, scores = peak_detector.extract_peaks_3d(hm, threshold=detection_threshold, max_peaks=max_peaks_per_frame)
            for (z, y, x), score in zip(peaks, scores):
                detected.append({"t": int(t), "node_id": len(detected), "z": float(z), "y": float(y),
                                 "x": float(x), "score": float(score)})

        gt_arr, bg_arr = np.asarray(gt_vals, np.float32), np.asarray(bg_vals, np.float32)
        if detected:
            res = BipartiteNodeEvaluator(max_distance=max_distance_um, scale=ds.scale).evaluate(pl.DataFrame(detected), gt_nodes)
            tp, fp, fn, prec, rec = res.tp, res.fp, res.fn, res.precision, res.recall
            jaccard = tp / max(1, tp + fp + fn)
            ratio = (tp + fp) / max(1, res.num_gt_nodes)
            j_adj = max(0.0, jaccard * (1.0 - 0.1 * ratio))
        else:
            tp, fp, fn, prec, rec, jaccard, j_adj = 0, 0, len(gt_nodes), 0.0, 0.0, 0.0, 0.0

        bg_p90 = _percentile(bg_arr, 90)
        gt_mean = float(gt_arr.mean()) if len(gt_arr) else 0.0
        results[ds.name] = {
            "gt_mean": gt_mean, "gt_median": _percentile(gt_arr, 50), "gt_p90": _percentile(gt_arr, 90),
            "gt_frac_above_threshold": float((gt_arr >= detection_threshold).mean()) if len(gt_arr) else 0.0,
            "bg_p50": _percentile(bg_arr, 50), "bg_p90": bg_p90, "bg_p99": _percentile(bg_arr, 99),
            "bg_max": float(bg_arr.max()) if len(bg_arr) else 0.0, "contrast": gt_mean / max(bg_p90, 1e-6),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "precision": float(prec), "recall": float(rec),
            "jaccard": float(jaccard), "j_adj": float(j_adj),
        }

    macro = float(np.mean([r["j_adj"] for r in results.values()])) if results else 0.0
    return results, macro

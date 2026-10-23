#!/usr/bin/env python3
"""Diagnostic Detection Threshold Sweeper across Permanent Validation Anchors.

Sweeps detection threshold tau in [0.10, 0.35] on permanent validation anchors:
- 44b6_66f9292d
- 6bba_d82a4fc6

Reports node TP, FP, FN, precision, recall, J_adj, and simulated Final Score
per threshold without modifying the locked production threshold (0.25).
"""

from pathlib import Path
import argparse
import csv
import json
import logging
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.data.zarr_reader import open_dataset
from src.data.preprocessing import normalize_intensity
from src.detection.unet3d import TemporalUNet3D
from src.detection.peak_detector import PeakDetector3D
from src.evaluation.tracking_metrics import BipartiteNodeEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def sweep_thresholds(
    detector_path: Path,
    thresholds: List[float] = [0.10, 0.15, 0.20, 0.25, 0.30],
    max_peaks_per_frame: int = 150,
    device: str = "cuda",
    output_csv: Optional[Path] = None,
) -> List[Dict]:
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    logger.info(f"Using device: {dev}")
    logger.info(f"Loading detector checkpoint: {detector_path}")

    ckpt = torch.load(detector_path, map_location=dev, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    detector = TemporalUNet3D(
        in_channels=1,
        base_channels=16,
        feature_dim=32,
        prior_prob=0.05,
        use_temporal_attention=False,
    ).to(dev)
    detector.load_state_dict(state)
    detector.eval()

    val_dir = repo_root / "data/val"
    anchor_stems = ["44b6_66f9292d", "6bba_d82a4fc6"]
    val_datasets = [open_dataset(val_dir / s, load_tracks=True) for s in anchor_stems]

    # Precompute heatmaps once for each sequence (100 frames)
    logger.info("Precomputing 3D heatmaps across validation sequences...")
    heatmaps_by_seq: Dict[str, Dict[int, np.ndarray]] = {}

    for ds in val_datasets:
        heatmaps_by_seq[ds.name] = {}
        active_t = sorted(ds.tracks.node_attrs()["t"].unique().to_list())
        t0 = time.time()
        with torch.no_grad():
            for t in active_t:
                vol_crop = ds.read_spatial_crop(
                    slice(t, t + 1),
                    slice(None),
                    slice(None),
                    slice(None),
                )
                vol_norm = normalize_intensity(vol_crop, quantiles=ds.quantiles)
                inp = torch.from_numpy(vol_norm).unsqueeze(0).to(dev)
                with torch.amp.autocast(device_type=dev.type, enabled=(dev.type == "cuda")):
                    pred_hm, _ = detector(inp)
                hm_3d = pred_hm[0, 0].cpu().numpy().astype(np.float32)
                heatmaps_by_seq[ds.name][t] = hm_3d
                del inp, pred_hm
        logger.info(f"  [{ds.name}] 100 heatmaps computed in {time.time()-t0:.1f}s.")

    bipartite_evaluator = BipartiteNodeEvaluator(max_distance=7.0, scale=(1.0, 0.26, 0.26))
    sweep_results = []

    print("\n" + "=" * 95)
    print(f"{'Threshold':<10} | {'Domain':<15} | {'TP':<6} | {'FP':<7} | {'FN':<6} | {'Recall':<8} | {'Precision':<10} | {'J_adj':<8}")
    print("=" * 95)

    for thresh in thresholds:
        macro_j_adj = 0.0
        peak_detector = PeakDetector3D(threshold=thresh, max_peaks_per_frame=max_peaks_per_frame)

        for ds in val_datasets:
            gt_nodes = ds.tracks.node_attrs()
            detected_records = []
            global_node_id = 0

            for t, hm_3d in heatmaps_by_seq[ds.name].items():
                peaks, scores = peak_detector.extract_peaks_3d(
                    hm_3d,
                    threshold=thresh,
                    max_peaks=max_peaks_per_frame,
                )
                for (z, y, x), s in zip(peaks, scores):
                    detected_records.append({
                        "t": int(t),
                        "node_id": int(global_node_id),
                        "z": float(z),
                        "y": float(y),
                        "x": float(x),
                        "score": float(s),
                    })
                    global_node_id += 1

            if len(detected_records) > 0:
                pred_df = pl.DataFrame(detected_records)
                m = bipartite_evaluator.evaluate(pred_df, gt_nodes)
                tp, fp, fn = m.tp, m.fp, m.fn
                prec, rec = m.precision, m.recall
                jaccard = tp / max(1, tp + fp + fn)
                ratio = (tp + fp) / max(1, m.num_gt_nodes)
                j_adj = max(0.0, jaccard * (1.0 - 0.1 * ratio))
            else:
                tp, fp, fn = 0, 0, len(gt_nodes)
                prec, rec, jaccard, j_adj = 0.0, 0.0, 0.0, 0.0

            macro_j_adj += 0.5 * j_adj

            row = {
                "threshold": thresh,
                "sequence": ds.name,
                "node_tp": tp,
                "node_fp": fp,
                "node_fn": fn,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "jaccard": round(jaccard, 6),
                "j_adj": round(j_adj, 6),
            }
            sweep_results.append(row)
            print(f"{thresh:<10.2f} | {ds.name:<15} | {tp:<6} | {fp:<7} | {fn:<6} | {rec:<8.2%} | {prec:<10.2%} | {j_adj:<8.4f}")

    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(sweep_results[0].keys()))
            writer.writeheader()
            writer.writerows(sweep_results)
        logger.info(f"\nSaved threshold sweep results to {output_csv}")

    return sweep_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep detection thresholds on validation anchors.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_detector.pt", help="Detector checkpoint path")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.10, 0.15, 0.20, 0.25, 0.30], help="Thresholds to sweep")
    parser.add_argument("--output", type=str, default="reports/chunk3_failure_analysis/threshold_sweep.csv", help="Output CSV path")
    args = parser.parse_args()

    sweep_thresholds(
        detector_path=Path(args.checkpoint),
        thresholds=args.thresholds,
        output_csv=Path(args.output),
    )

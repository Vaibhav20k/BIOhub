#!/usr/bin/env python3
"""Controlled inference ablation matrix and detection quality diagnostics.

Evaluates the 6 pairings:
1. Det_C2_base + Trk_C2_base
2. Det_C3_ep01 + Trk_C2_base
3. Det_C3_ep01 + Trk_C3_cont
4. Det_C3_ep10 + Trk_C2_base
5. Det_C3_ep10 + Trk_C3_cont
6. Det_C2_base + Trk_C3_cont

Under identical locked settings:
- detection_threshold: 0.25
- max_peaks_per_frame: 150
- max_distance_um: 7.0
- window_size: 1
"""

import csv
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import polars as pl
import torch

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.data.zarr_reader import DatasetVolume, open_dataset
from src.data.preprocessing import normalize_intensity
from src.detection.unet3d import TemporalUNet3D
from src.detection.peak_detector import PeakDetector3D
from src.representation.node_embedding import CellNodeEmbedding, extract_node_embeddings_for_dataset
from src.tracking.transformer import SpatioTemporalTracker
from src.pipeline.held_out_evaluator import HeldOutEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

out_dir = repo_root / "reports/chunk3_failure_analysis"
out_dir.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Validation Datasets
val_dir = repo_root / "data/val"
anchor_stems = ["44b6_66f9292d", "6bba_d82a4fc6"]
val_datasets = [open_dataset(val_dir / stem, load_tracks=True) for stem in anchor_stems]
print(f"Loaded validation datasets: {[ds.name for ds in val_datasets]}")

# 2. Checkpoints
ckpt_c2_det = repo_root / "checkpoints/best_detector.pt"
ckpt_c2_trk = repo_root / "checkpoints/best_tracker.pt"

ckpt_c3_ep1_det = repo_root / "checkpoints/chunk3_epochs/epoch_01_detector.pt"
ckpt_c3_ep10_det = repo_root / "checkpoints/chunk3_epochs/epoch_10_detector.pt"
ckpt_c3_trk = repo_root / "checkpoints/chunk3_epochs/tracker_continual_5ep.pt"

# Helper to load models
def load_detector(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model = TemporalUNet3D(
        in_channels=1,
        base_channels=16,
        feature_dim=32,
        prior_prob=0.05,
        use_temporal_attention=False,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model

def load_tracker_and_embedder(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    embedder = CellNodeEmbedding(in_visual_dim=32, embedding_dim=128).to(device)
    if "embedder_state_dict" in ckpt:
        embedder.load_state_dict(ckpt["embedder_state_dict"])
    embedder.eval()
    
    tracker = SpatioTemporalTracker(
        node_dim=128,
        rel_dim=32,
        hidden_dim=128,
        num_layers=2,
    ).to(device)
    tracker.load_state_dict(ckpt["tracker_state_dict"])
    tracker.eval()
    return embedder, tracker

print("\n--- Loading Models ---")
det_c2 = load_detector(ckpt_c2_det)
emb_c2, trk_c2 = load_tracker_and_embedder(ckpt_c2_trk)

det_c3_ep1 = load_detector(ckpt_c3_ep1_det)
det_c3_ep10 = load_detector(ckpt_c3_ep10_det)
emb_c3, trk_c3 = load_tracker_and_embedder(ckpt_c3_trk)
print("All detector and tracker models successfully loaded into VRAM.")

# 3. Heatmap Quality & Calibration Audit (Section 6.1)
print("\n--- Running Heatmap Calibration Audit ---")
heatmap_diagnostics = {}
detectors_dict = {
    "Chunk 2 Baseline Detector": det_c2,
    "Chunk 3 Epoch 1 Detector": det_c3_ep1,
    "Chunk 3 Epoch 10 Detector": det_c3_ep10,
}

rng = np.random.RandomState(42)

for det_name, det_model in detectors_dict.items():
    heatmap_diagnostics[det_name] = {}
    for ds in val_datasets:
        gt_nodes = ds.tracks.node_attrs()
        active_t = sorted(gt_nodes["t"].unique().to_list())
        
        gt_vals = []
        bg_vals = []
        
        with torch.no_grad():
            for t in active_t:
                vol_crop = ds.read_spatial_crop(
                    slice(t, t + 1),
                    slice(None),
                    slice(None),
                    slice(None),
                )
                vol_norm = normalize_intensity(vol_crop, quantiles=ds.quantiles)
                inp = torch.from_numpy(vol_norm).unsqueeze(0).to(device)
                with torch.amp.autocast(device_type="cuda"):
                    pred_hm, _ = det_model(inp)
                hm_3d = pred_hm[0, 0].cpu().numpy().astype(np.float32)
                del inp, pred_hm


                
                # GT center activations
                frame_gt = gt_nodes.filter(pl.col("t") == t)
                for row in frame_gt.iter_rows(named=True):
                    z = int(np.clip(round(row["z"]), 0, hm_3d.shape[0] - 1))
                    y = int(np.clip(round(row["y"]), 0, hm_3d.shape[1] - 1))
                    x = int(np.clip(round(row["x"]), 0, hm_3d.shape[2] - 1))
                    gt_vals.append(float(hm_3d[z, y, x]))
                
                # Sample 1000 background voxels
                flat_hm = hm_3d.ravel()
                bg_sample_idx = rng.choice(flat_hm.shape[0], size=1000, replace=False)
                bg_vals.extend(flat_hm[bg_sample_idx].tolist())
                
        gt_arr = np.array(gt_vals, dtype=np.float32)
        bg_arr = np.array(bg_vals, dtype=np.float32)
        
        # Compute exact percentiles
        gt_mean = float(np.mean(gt_arr))
        gt_median = float(np.median(gt_arr))
        gt_p90 = float(np.percentile(gt_arr, 90))
        gt_p95 = float(np.percentile(gt_arr, 95))
        gt_p99 = float(np.percentile(gt_arr, 99))
        gt_below_thresh = float(np.mean(gt_arr < 0.25))
        
        bg_mean = float(np.mean(bg_arr))
        bg_median = float(np.median(bg_arr))
        bg_p90 = float(np.percentile(bg_arr, 90))
        bg_p95 = float(np.percentile(bg_arr, 95))
        bg_p99 = float(np.percentile(bg_arr, 99))
        bg_p999 = float(np.percentile(bg_arr, 99.9))
        bg_above_thresh = float(np.mean(bg_arr >= 0.25))
        
        contrast_ratio = gt_mean / max(bg_p90, 1e-6)
        separation = gt_mean - bg_p90
        
        diag = {
            "gt_mean": gt_mean,
            "gt_median": gt_median,
            "gt_p90": gt_p90,
            "gt_p95": gt_p95,
            "gt_p99": gt_p99,
            "gt_below_thresh_frac": gt_below_thresh,
            "bg_mean": bg_mean,
            "bg_median": bg_median,
            "bg_p90": bg_p90,
            "bg_p95": bg_p95,
            "bg_p99": bg_p99,
            "bg_p999": bg_p999,
            "bg_above_thresh_frac": bg_above_thresh,
            "contrast_ratio": contrast_ratio,
            "separation": separation,
        }
        heatmap_diagnostics[det_name][ds.name] = diag
        print(f"[{det_name} | {ds.name}] GT Mean={gt_mean:.4f}, GT P90={gt_p90:.4f}, BG P90={bg_p90:.4f}, Contrast={contrast_ratio:.2f}, GT < 0.25: {gt_below_thresh:.1%}")

with open(out_dir / "domain_diagnostics.json", "w") as f:
    json.dump(heatmap_diagnostics, f, indent=2)
print("Saved domain_diagnostics.json.")

# 4. Controlled Inference Matrix (Section 5.1)
print("\n--- Running 6-Cell Controlled Inference Matrix ---")
matrix_configs = [
    {
        "id": "EXP-1",
        "name": "Reference Chunk 2 Baseline",
        "det_label": "Chunk 2 Baseline Det",
        "trk_label": "Chunk 2 Baseline Trk",
        "det": det_c2,
        "emb": emb_c2,
        "trk": trk_c2,
        "purpose": "Reproduce reference baseline",
    },
    {
        "id": "EXP-2",
        "name": "Early Detector Ablation",
        "det_label": "Chunk 3 Epoch 1 Det",
        "trk_label": "Chunk 2 Baseline Trk",
        "det": det_c3_ep1,
        "emb": emb_c2,
        "trk": trk_c2,
        "purpose": "Isolate early detector change (holding tracker constant)",
    },
    {
        "id": "EXP-3",
        "name": "Early Joint Configuration",
        "det_label": "Chunk 3 Epoch 1 Det",
        "trk_label": "Chunk 3 Continual Trk",
        "det": det_c3_ep1,
        "emb": emb_c3,
        "trk": trk_c3,
        "purpose": "Reproduce reported Chunk 3 peak configuration",
    },
    {
        "id": "EXP-4",
        "name": "Late Detector Ablation",
        "det_label": "Chunk 3 Epoch 10 Det",
        "trk_label": "Chunk 2 Baseline Trk",
        "det": det_c3_ep10,
        "emb": emb_c2,
        "trk": trk_c2,
        "purpose": "Isolate later detector degradation (holding tracker constant)",
    },
    {
        "id": "EXP-5",
        "name": "Late Joint Configuration",
        "det_label": "Chunk 3 Epoch 10 Det",
        "trk_label": "Chunk 3 Continual Trk",
        "det": det_c3_ep10,
        "emb": emb_c3,
        "trk": trk_c3,
        "purpose": "Evaluate final epoch joint configuration",
    },
    {
        "id": "EXP-6",
        "name": "Tracker-Only Ablation",
        "det_label": "Chunk 2 Baseline Det",
        "trk_label": "Chunk 3 Continual Trk",
        "det": det_c2,
        "emb": emb_c3,
        "trk": trk_c3,
        "purpose": "Isolate tracker contribution (holding detector constant)",
    },
]

ablation_rows = []

for cfg in matrix_configs:
    print(f"\nRunning {cfg['id']}: {cfg['name']} ({cfg['det_label']} + {cfg['trk_label']})...")
    evaluator = HeldOutEvaluator(
        detector=cfg["det"],
        embedder=cfg["emb"],
        tracker=cfg["trk"],
        detection_threshold=0.25,
        max_peaks_per_frame=150,
        max_distance_um=7.0,
        window_size=1,
        diagnostic_mode=False,
        device=device,
    )
    
    summary = evaluator.evaluate_cohort(val_datasets)
    
    macro_score = summary["macro_final_score"]
    macro_j_adj = summary["macro_adjusted_jaccard"]
    macro_edge_j = summary["macro_edge_jaccard"]
    macro_div_j = summary["macro_division_jaccard"]
    tp = summary["total_node_tp"]
    fp = summary["total_node_fp"]
    fn = summary["total_node_fn"]
    
    s44 = summary["sequences"]["44b6_66f9292d"]
    s6b = summary["sequences"]["6bba_d82a4fc6"]
    
    row = {
        "exp_id": cfg["id"],
        "experiment_name": cfg["name"],
        "detector": cfg["det_label"],
        "tracker": cfg["trk_label"],
        "purpose": cfg["purpose"],
        "macro_final_score": f"{macro_score:.6f}",
        "6bba_final_score": f"{s6b['final_score']:.6f}",
        "44b6_final_score": f"{s44['final_score']:.6f}",
        "macro_edge_jaccard": f"{macro_edge_j:.6f}",
        "6bba_edge_jaccard": f"{s6b['edge_jaccard']:.6f}",
        "44b6_edge_jaccard": f"{s44['edge_jaccard']:.6f}",
        "macro_adjusted_node_jaccard": f"{macro_j_adj:.6f}",
        "6bba_adjusted_node_jaccard": f"{s6b['j_adj']:.6f}",
        "44b6_adjusted_node_jaccard": f"{s44['j_adj']:.6f}",
        "macro_division_jaccard": f"{macro_div_j:.6f}",
        "6bba_division_jaccard": f"{s6b['j_div']:.6f}",
        "44b6_division_jaccard": f"{s44['j_div']:.6f}",
        "total_node_tp": tp,
        "total_node_fp": fp,
        "total_node_fn": fn,
        "6bba_node_tp": s6b["node_tp"],
        "6bba_node_fp": s6b["node_fp"],
        "6bba_node_fn": s6b["node_fn"],
        "6bba_node_recall": f"{s6b['node_recall']:.4f}",
        "6bba_node_precision": f"{s6b['node_precision']:.4f}",
        "44b6_node_tp": s44["node_tp"],
        "44b6_node_fp": s44["node_fp"],
        "44b6_node_fn": s44["node_fn"],
        "44b6_node_recall": f"{s44['node_recall']:.4f}",
        "44b6_node_precision": f"{s44['node_precision']:.4f}",
        "inference_time_sec": f"{summary['inference_time_sec']:.2f}",
        "ilp_runtime_sec": f"{summary['ilp_runtime_sec']:.2f}",
    }
    ablation_rows.append(row)
    print(f"  --> Macro: {macro_score:.4f} | 6bba: {s6b['final_score']:.4f} | 44b6: {s44['final_score']:.4f} | TP={tp}, FP={fp}, FN={fn}")
    
    # Incremental write
    csv_ablation = out_dir / "detector_tracker_ablation.csv"
    with open(csv_ablation, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ablation_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ablation_rows)
    
    del evaluator, summary
    torch.cuda.empty_cache()

print(f"\nWrote ablation matrix to {csv_ablation}")

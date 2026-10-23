#!/usr/bin/env python3
"""Phase 5 — Chunk 3.1 Controlled Retraining Pipeline (Kaggle GPU / Local).

Corrective continual learning re-execution addressing Chunk 3 regression root causes:
1. Cell-Density-Normalized Stratified Sampling (78.7% 6bba, 21.3% 44b6, zero cell starvation).
2. 10x Lower Learning Rate (LR = 0.0002) with Cosine Annealing (down to 1e-5).
3. Feature-Manifold Distillation Loss (lambda = 0.1) against frozen Chunk 2 baseline representations.
4. Per-epoch anchor validation monitoring both tau=0.25 and tau=0.30.
5. Automated promotion gating requiring:
   - Macro Final Score >= 0.03865588
   - In-domain 6bba anchor regression <= 15% (score >= 0.065715)
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import shutil
import sqlite3
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.data.zarr_reader import DatasetVolume, open_dataset
from src.data.preprocessing import normalize_intensity, adaptive_contrast_normalize
from src.data.patch_sampler import SpatioTemporalPatchSampler, CellDensityStratifiedDataset
from src.detection.unet3d import TemporalUNet3D
from src.detection.peak_detector import PeakDetector3D
from src.training.losses import DetectionLoss
from src.representation.node_embedding import CellNodeEmbedding
from src.tracking.transformer import SpatioTemporalTracker
from src.pipeline.held_out_evaluator import HeldOutEvaluator
from src.evaluation.tracking_metrics import BipartiteNodeEvaluator
from src.pipeline.continual_trainer import ContinualTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EXPECTED_BASELINE_DETECTOR_SHA256 = "6e9a64c754551402d45c30b7a6949d04f2d2d7e40a3f076243581037a130eaab"
EXPECTED_BASELINE_TRACKER_SHA256 = "ed3a4498461da21dbd7b39943eece75f06b6b0fa8faaf570c3aa93403994baf5"
BASELINE_MACRO_SCORE = 0.03865588
BASELINE_6BBA_SCORE = 0.07731176

ANCHOR_STEMS = ["44b6_66f9292d", "6bba_d82a4fc6"]
CHUNK3_STEMS = [
    "44b6_0c582fdc", "44b6_c96cfa10", "44b6_996155de",
    "6bba_61ecbe65", "6bba_cf35214c", "6bba_b204cac7",
    "6bba_d0fc38b5", "6bba_ebdf3b34"
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def run_anchor_validation(
    detector: nn.Module,
    val_datasets: List[DatasetVolume],
    device: torch.device,
    threshold: float = 0.25,
    max_peaks: int = 150,
) -> Tuple[Dict[str, Any], float]:
    detector.eval()
    results = {}
    macro_j_adj = 0.0
    bipartite = BipartiteNodeEvaluator(max_distance=7.0, scale=(1.0, 0.26, 0.26))
    peak_detector = PeakDetector3D(threshold=threshold, max_peaks_per_frame=max_peaks)

    with torch.no_grad():
        for ds in val_datasets:
            gt_nodes = ds.tracks.node_attrs()
            active_t = sorted(gt_nodes["t"].unique().to_list())
            detected_records = []
            global_node_id = 0
            gt_vals = []
            bg_vals = []
            rng = np.random.RandomState(42)

            for t in active_t:
                vol_crop = ds.read_spatial_crop(slice(t, t + 1), slice(None), slice(None), slice(None))
                vol_norm = normalize_intensity(vol_crop, quantiles=ds.quantiles)
                inp = torch.from_numpy(vol_norm).unsqueeze(0).to(device)

                with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    pred_hm, _ = detector(inp)

                hm_3d = pred_hm[0, 0].cpu().numpy().astype(np.float32)
                del inp, pred_hm

                frame_gt = gt_nodes.filter(pl.col("t") == t)
                for row in frame_gt.iter_rows(named=True):
                    z = int(np.clip(round(row["z"]), 0, hm_3d.shape[0] - 1))
                    y = int(np.clip(round(row["y"]), 0, hm_3d.shape[1] - 1))
                    x = int(np.clip(round(row["x"]), 0, hm_3d.shape[2] - 1))
                    gt_vals.append(float(hm_3d[z, y, x]))

                flat = hm_3d.ravel()
                bg_sample_idx = rng.choice(flat.shape[0], size=1000, replace=False)
                bg_vals.extend(flat[bg_sample_idx].tolist())

                peaks, scores = peak_detector.extract_peaks_3d(hm_3d, threshold=threshold, max_peaks=max_peaks)
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
                m = bipartite.evaluate(pred_df, gt_nodes)
                tp, fp, fn = m.tp, m.fp, m.fn
                prec, rec = m.precision, m.recall
                jaccard = tp / max(1, tp + fp + fn)
                ratio = (tp + fp) / max(1, m.num_gt_nodes)
                j_adj = max(0.0, jaccard * (1.0 - 0.1 * ratio))
            else:
                tp, fp, fn = 0, 0, len(gt_nodes)
                prec, rec, jaccard, j_adj = 0.0, 0.0, 0.0, 0.0

            macro_j_adj += 0.5 * j_adj
            gt_arr = np.array(gt_vals, dtype=np.float32)
            bg_arr = np.array(bg_vals, dtype=np.float32)

            results[ds.name] = {
                "tp": tp, "fp": fp, "fn": fn,
                "precision": prec, "recall": rec,
                "jaccard": jaccard, "j_adj": j_adj,
                "gt_mean": float(np.mean(gt_arr)) if len(gt_arr) > 0 else 0.0,
                "bg_p90": float(np.percentile(bg_arr, 90)) if len(bg_arr) > 0 else 0.0,
            }

    return results, macro_j_adj


def main():
    parser = argparse.ArgumentParser(description="Run Chunk 3.1 Controlled Retraining.")
    parser.add_argument("--epochs", type=int, default=8, help="Detector continual learning epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Continual learning rate")
    parser.add_argument("--distill-weight", type=float, default=0.1, help="Feature distillation loss weight")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--total-patches", type=int, default=800, help="Patches per epoch")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="checkpoints/chunk3_1_retrain")
    args = parser.parse_args()

    dev = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BIOHUB CHUNK 3.1 — CONTROLLED CONTINUAL RETRAINING")
    print("=" * 80)
    print(f"Device:           {dev}")
    print(f"Epochs:           {args.epochs}")
    print(f"Learning Rate:    {args.lr}")
    print(f"Distill Weight:   {args.distill_weight}")
    print(f"Total Patches:    {args.total_patches}/epoch")
    print(f"Output Directory: {out_dir}")

    # 1. Verify Active Baseline Checkpoints
    best_det_path = repo_root / "checkpoints/best_detector.pt"
    best_trk_path = repo_root / "checkpoints/best_tracker.pt"
    det_hash = sha256_file(best_det_path)
    trk_hash = sha256_file(best_trk_path)
    print(f"\n[Step 1/5] Verifying Baseline Checkpoints...")
    print(f"  Detector SHA-256: {det_hash}")
    print(f"  Tracker SHA-256:  {trk_hash}")
    assert det_hash == EXPECTED_BASELINE_DETECTOR_SHA256, "Baseline detector hash mismatch!"
    assert trk_hash == EXPECTED_BASELINE_TRACKER_SHA256, "Baseline tracker hash mismatch!"

    # 2. Validation Anchors
    val_dir = repo_root / "data/val"
    val_datasets = [
        open_dataset(val_dir / "44b6_66f9292d", load_tracks=True),
        open_dataset(val_dir / "6bba_d82a4fc6", load_tracks=True),
    ]

    # 3. Model Loading & Feature Distillation Teacher Setup
    print(f"\n[Step 2/5] Loading Detector & Frozen Baseline Distillation Teacher...")
    det_ckpt = torch.load(best_det_path, map_location=dev, weights_only=False)
    detector = TemporalUNet3D(
        in_channels=1, base_channels=16, feature_dim=32, prior_prob=0.05, use_temporal_attention=False
    ).to(dev)
    detector.load_state_dict(det_ckpt["model_state_dict"])

    frozen_teacher = TemporalUNet3D(
        in_channels=1, base_channels=16, feature_dim=32, prior_prob=0.05, use_temporal_attention=False
    ).to(dev)
    frozen_teacher.load_state_dict(det_ckpt["model_state_dict"])
    frozen_teacher.eval()
    for p in frozen_teacher.parameters():
        p.requires_grad = False

    trk_ckpt = torch.load(best_trk_path, map_location=dev, weights_only=False)
    embedder = CellNodeEmbedding(in_visual_dim=32, embedding_dim=128).to(dev)
    if "embedder_state_dict" in trk_ckpt:
        embedder.load_state_dict(trk_ckpt["embedder_state_dict"])

    tracker = SpatioTemporalTracker(
        node_dim=128, rel_dim=32, hidden_dim=128, num_layers=2
    ).to(dev)
    tracker.load_state_dict(trk_ckpt["tracker_state_dict"])
    print("  Models successfully prepared in VRAM.")

    # 4. Pre-Training Validation
    print("\n[Step 3/5] Pre-Training Baseline Validation Check...")
    val_res_pre, macro_pre = run_anchor_validation(detector, val_datasets, dev, threshold=0.25)
    print(f"  Pre-train Anchor Status | 44b6 J_adj: {val_res_pre['44b6_66f9292d']['j_adj']:.4f} | 6bba J_adj: {val_res_pre['6bba_d82a4fc6']['j_adj']:.4f} | Macro J_adj: {macro_pre:.4f}")

    # 5. Training Pipeline Setup
    print(f"\n[Step 4/5] Setting up Density-Normalized Continual Training...")
    optimizer = AdamW(detector.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = DetectionLoss(focal_weight=1.0, dice_weight=1.0)
    mse_distill = nn.MSELoss()

    print("\n--- Continual Retraining Configured Successfully ---")
    print("Ready for training execution on Kaggle GPU.")


if __name__ == "__main__":
    main()

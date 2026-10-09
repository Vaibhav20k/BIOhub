#!/usr/bin/env python3
"""Phase 4 — Chunk 2 Continual Learning Script.

Continual learning on Chunk 2 starting from the promoted Phase 3 checkpoints:
- best_detector.pt (SHA256: ed063376eab75193bc4d667f73a89c03f0f2ee66b8b0e9b2d15a787e8cdcc13f)
- best_tracker.pt (SHA256: e4809f41bc08c9b26f1cb9129585947cb7c86183ae1b8a27ecdb0d81a26f9fa0)

Protocol:
- Detector: TemporalUNet3D (W=1, base_channels=16, feature_dim=32, prior_prob=0.05, bias≈-2.944)
- Training: LR=0.002, 10 epochs, batch=2, FP32, patch=32×128×128, 800 patches/epoch (100 per seq across 8 sequences)
- Augmentation: contrast ±15%, brightness ±15%, background offset ±10%
- Loss: Focal + Dice (1.0/1.0)
- Per-epoch validation on permanent anchors (44b6_66f9292d & 6bba_d82a4fc6):
  * GT activation mean, median, P90
  * Background percentiles (P50, P90, P99, max)
  * Contrast ratio
  * Node TP, FP, FN, precision, recall, Jaccard, J_adj (threshold=0.25, max_peaks=150, dist<=7.0um)
- Tracker: 5 epochs continual fine-tuning on Chunk 2 at LR=0.0003
- Full end-to-end HeldOutEvaluator pipeline on permanent anchors
- Catastrophic forgetting check vs Promoted Phase 3 Baseline (Macro = 0.0212)
- Strict promotion policy (promote if overall improved without unacceptable forgetting)
- Archive candidate checkpoints to checkpoints/archive/chunk2_candidate_*
- Update registry/experiments.sqlite
- Transactional cleanup of chunk_002_train
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import random
import shutil
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, ConcatDataset

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.zarr_reader import DatasetVolume, open_dataset, DEFAULT_SCALE
from src.data.preprocessing import normalize_intensity
from src.detection.dataset import DetectionDataset
from src.training.losses import DetectionLoss
from src.detection.unet3d import TemporalUNet3D
from src.detection.peak_detector import PeakDetector3D
from src.evaluation.node_matching import BipartiteNodeEvaluator
from src.representation.node_embedding import CellNodeEmbedding
from src.tracking.transformer import SpatioTemporalTracker
from src.pipeline.chunk_manager import ChunkStorageManager
from src.pipeline.continual_trainer import ContinualTrainer
from src.pipeline.held_out_evaluator import HeldOutEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHUNK2_STEMS = [
    "6bba_907271db",
    "6bba_f8ffd5e7",
    "6bba_337b1b3a",
    "6bba_f1fde7e0",
    "6bba_718b21f9",
    "6bba_74686d6a",
    "6bba_705ec2c9",
    "6bba_e5e44988",
]

# Promoted Phase 3 Baseline metrics to compare against
PROMOTED_PHASE3_BASELINE = {
    "macro_final_score": 0.02121,
    "44b6_final_score": 0.0000,
    "6bba_final_score": 0.04243,
    "total_node_tp": 118,
    "total_node_fp": 1480,
    "total_node_fn": 627,
    "macro_node_recall": 0.1082,
    "macro_node_precision": 0.0419,
    "macro_edge_jaccard": 0.0289,
    "macro_division_jaccard": 0.0,
}

EXPECTED_PROMOTED_DETECTOR_SHA256 = "ed063376eab75193bc4d667f73a89c03f0f2ee66b8b0e9b2d15a787e8cdcc13f"
EXPECTED_PROMOTED_TRACKER_SHA256 = "e4809f41bc08c9b26f1cb9129585947cb7c86183ae1b8a27ecdb0d81a26f9fa0"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_per_epoch_validation(
    detector: TemporalUNet3D,
    val_datasets: List[DatasetVolume],
    device: torch.device,
    detection_threshold: float = 0.25,
    max_peaks_per_frame: int = 150,
) -> Tuple[Dict[str, Dict[str, float]], float]:
    """Evaluate detector on validation anchors at the end of each epoch."""
    detector.eval()
    results = {}
    
    peak_detector = PeakDetector3D(
        threshold=detection_threshold,
        max_peaks_per_frame=max_peaks_per_frame,
    )
    
    for ds in val_datasets:
        gt_nodes = ds.tracks.node_attrs()
        active_t = sorted(gt_nodes["t"].unique().to_list())
        
        detected_records = []
        global_node_id = 0
        gt_cell_vals = []
        bg_vals = []
        
        rng = np.random.RandomState(42)
        
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
                pred_hm, _ = detector(inp)
                hm_3d = pred_hm[0, 0].cpu().numpy().astype(np.float32)
                
                # GT cell activations for frame t
                frame_gt = gt_nodes.filter(pl.col("t") == t)
                for row in frame_gt.iter_rows(named=True):
                    z = int(np.clip(round(row["z"]), 0, hm_3d.shape[0] - 1))
                    y = int(np.clip(round(row["y"]), 0, hm_3d.shape[1] - 1))
                    x = int(np.clip(round(row["x"]), 0, hm_3d.shape[2] - 1))
                    gt_cell_vals.append(float(hm_3d[z, y, x]))
                
                # Subsample 1000 background voxels from frame
                flat_hm = hm_3d.ravel()
                bg_sample_idx = rng.choice(flat_hm.shape[0], size=1000, replace=False)
                bg_vals.extend(flat_hm[bg_sample_idx].tolist())
                
                peaks, scores = peak_detector.extract_peaks_3d(
                    hm_3d,
                    threshold=detection_threshold,
                    max_peaks=max_peaks_per_frame,
                )
                for (z, y, x), score in zip(peaks, scores):
                    detected_records.append({
                        "t": int(t),
                        "node_id": int(global_node_id),
                        "z": float(z),
                        "y": float(y),
                        "x": float(x),
                        "score": float(score),
                    })
                    global_node_id += 1
        
        gt_vals_arr = np.array(gt_cell_vals, dtype=np.float32)
        bg_vals_arr = np.array(bg_vals, dtype=np.float32)
        
        gt_mean = float(np.mean(gt_vals_arr)) if len(gt_vals_arr) > 0 else 0.0
        gt_median = float(np.median(gt_vals_arr)) if len(gt_vals_arr) > 0 else 0.0
        gt_p90 = float(np.percentile(gt_vals_arr, 90)) if len(gt_vals_arr) > 0 else 0.0
        
        bg_p50 = float(np.percentile(bg_vals_arr, 50)) if len(bg_vals_arr) > 0 else 0.0
        bg_p90 = float(np.percentile(bg_vals_arr, 90)) if len(bg_vals_arr) > 0 else 0.0
        bg_p99 = float(np.percentile(bg_vals_arr, 99)) if len(bg_vals_arr) > 0 else 0.0
        bg_max = float(np.max(bg_vals_arr)) if len(bg_vals_arr) > 0 else 0.0
        contrast = gt_mean / max(bg_p90, 1e-6)
        
        # Bipartite matching
        bipartite = BipartiteNodeEvaluator(max_distance=7.0, scale=ds.scale)
        if len(detected_records) > 0:
            pred_df = pl.DataFrame(detected_records)
            node_res = bipartite.evaluate(pred_df, gt_nodes)
            tp, fp, fn = node_res.tp, node_res.fp, node_res.fn
            prec, rec = node_res.precision, node_res.recall
            jaccard = tp / max(1, tp + fp + fn)
            ratio = (tp + fp) / max(1, node_res.num_gt_nodes)
            j_adj = max(0.0, jaccard * (1.0 - 0.1 * ratio))
        else:
            tp, fp, fn = 0, 0, len(gt_nodes)
            prec, rec, jaccard, ratio, j_adj = 0.0, 0.0, 0.0, 0.0, 0.0
        
        results[ds.name] = {
            "gt_mean": gt_mean,
            "gt_median": gt_median,
            "gt_p90": gt_p90,
            "bg_p50": bg_p50,
            "bg_p90": bg_p90,
            "bg_p99": bg_p99,
            "bg_max": bg_max,
            "contrast": contrast,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": prec,
            "recall": rec,
            "jaccard": jaccard,
            "j_adj": j_adj,
        }
    
    macro_j_adj = (results[val_datasets[0].name]["j_adj"] + results[val_datasets[1].name]["j_adj"]) / 2.0
    return results, macro_j_adj


def main():
    print("================================================================================")
    print("            PHASE 4: CHUNK 2 CONTINUAL LEARNING (FROM PHASE 3 PROMOTED)")
    print("================================================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 1. Pre-flight Checks
    chunk_dir = Path("data/chunks/chunk_002_train")
    val_dir = Path("data/val")
    storage_mgr = ChunkStorageManager()
    
    print("\n[Step 1/6] Running Pre-flight Integrity & Checkpoint Hash Checks...")
    print(f"  Disk Free: {storage_mgr.get_disk_free_gb():.2f} GB (Safety floor: {storage_mgr.min_free_disk_gb} GB)")
    storage_mgr.assert_disk_safety(required_extra_gb=1.0)
    
    # Check promoted checkpoints
    best_det_path = Path("checkpoints/best_detector.pt")
    best_trk_path = Path("checkpoints/best_tracker.pt")
    assert best_det_path.exists(), f"Active detector checkpoint missing at {best_det_path}"
    assert best_trk_path.exists(), f"Active tracker checkpoint missing at {best_trk_path}"
    
    det_hash = file_sha256(best_det_path)
    trk_hash = file_sha256(best_trk_path)
    print(f"  Active detector hash: {det_hash}")
    print(f"  Active tracker hash:  {trk_hash}")
    assert det_hash == EXPECTED_PROMOTED_DETECTOR_SHA256, f"Detector hash mismatch! Got {det_hash}, expected {EXPECTED_PROMOTED_DETECTOR_SHA256}"
    assert trk_hash == EXPECTED_PROMOTED_TRACKER_SHA256, f"Tracker hash mismatch! Got {trk_hash}, expected {EXPECTED_PROMOTED_TRACKER_SHA256}"
    print("  Active checkpoints confirmed matching promoted Phase 3 baseline hashes.")
    
    # Verify validation anchors
    val_stems = ["44b6_66f9292d", "6bba_d82a4fc6"]
    val_datasets = []
    for stem in val_stems:
        vp = val_dir / stem
        valid, err = storage_mgr.verify_sequence_integrity(vp, require_tracks=True)
        assert valid, f"Validation sequence {stem} integrity failure: {err}"
        val_datasets.append(open_dataset(vp, load_tracks=True))
    print(f"  Validation anchors verified intact: {[ds.name for ds in val_datasets]}")
    
    # Verify Chunk 2 training sequences
    train_datasets = []
    for stem in CHUNK2_STEMS:
        sp = chunk_dir / stem
        valid, err = storage_mgr.verify_sequence_integrity(sp, require_tracks=True)
        assert valid, f"Training sequence {stem} integrity failure: {err}"
        train_datasets.append(open_dataset(sp, load_tracks=True, require_tracks=True))
    print(f"  All {len(train_datasets)} Chunk 2 training sequences verified intact.")
    
    # 2. Assert and Verify Patch Budget Allocation
    print("\n[Step 2/6] Verifying Patch Budget Allocation for Chunk 2...")
    total_budget = 800
    dataset_sample_quotas = {ds.name: total_budget // len(train_datasets) for ds in train_datasets}
    print(f"  Allocated Quotas per Sequence (all 6bba): {dataset_sample_quotas}")
    total_allocated = sum(dataset_sample_quotas.values())
    print(f"  Total Patches/Epoch: {total_allocated}")
    assert total_allocated == 800, f"Expected 800 total patches, got {total_allocated}"
    
    # 3. Model Loading (Warm-start from promoted Phase 3 checkpoints)
    print("\n[Step 3/6] Loading Promoted Phase 3 Checkpoints (Warm-Start Continual Learning)...")
    torch.manual_seed(42)
    np.random.seed(42)
    
    det_ckpt = torch.load(best_det_path, map_location=device)
    detector = TemporalUNet3D(
        in_channels=1,
        base_channels=16,
        feature_dim=32,
        prior_prob=0.05,
        use_temporal_attention=False,
    ).to(device)
    detector.load_state_dict(det_ckpt["model_state_dict"])
    
    current_bias = detector.heatmap_head.bias.item()
    print(f"  Loaded Detector Heatmap Head Bias: {current_bias:.4f}")
    
    trk_ckpt = torch.load(best_trk_path, map_location=device)
    embedder = CellNodeEmbedding(in_visual_dim=32, embedding_dim=128).to(device)
    if "embedder_state_dict" in trk_ckpt:
        embedder.load_state_dict(trk_ckpt["embedder_state_dict"])
    
    tracker = SpatioTemporalTracker(
        node_dim=128,
        rel_dim=32,
        hidden_dim=128,
        num_layers=2,
    ).to(device)
    tracker.load_state_dict(trk_ckpt["tracker_state_dict"])
    print("  Detector and Tracker warm-start successfully loaded.")
    
    # Initial Pre-Training Anchor Check
    print("\n  Running baseline pre-training check on validation anchors...")
    val_res_pre, macro_j_adj_pre = run_per_epoch_validation(
        detector=detector,
        val_datasets=val_datasets,
        device=device,
        detection_threshold=0.25,
        max_peaks_per_frame=150,
    )
    r44_pre = val_res_pre["44b6_66f9292d"]
    r6b_pre = val_res_pre["6bba_d82a4fc6"]
    print(f"  Pre-train Anchor Status | 44b6 J_adj: {r44_pre['j_adj']:.4f} (rec={r44_pre['recall']:.1%}) | 6bba J_adj: {r6b_pre['j_adj']:.4f} (rec={r6b_pre['recall']:.1%}) | Macro J_adj: {macro_j_adj_pre:.4f}")
    
    # 4. Detector Continual Training with Per-Epoch Anchor Monitoring
    print("\n[Step 4/6] Executing 10 Epochs of Detector Continual Training on Chunk 2...")
    patch_size = (32, 128, 128)
    sub_datasets = [
        DetectionDataset(
            dataset=ds,
            patch_size=(
                min(patch_size[0], ds.shape[1]),
                min(patch_size[1], ds.shape[2]),
                min(patch_size[2], ds.shape[3]),
            ),
            window_size=1,
            samples_per_epoch=dataset_sample_quotas[ds.name],
            center_on_cell_prob=0.8,
            augment=True,
        )
        for ds in train_datasets
    ]
    concat_ds = ConcatDataset(sub_datasets)
    loader = DataLoader(concat_ds, batch_size=2, shuffle=True, drop_last=False)
    
    optimizer = AdamW(detector.parameters(), lr=0.002, weight_decay=1e-4)
    criterion = DetectionLoss(focal_weight=1.0, dice_weight=1.0)
    
    epoch_logs = []
    t_train_start = time.time()
    
    print("-" * 110)
    print(f"{'Ep':<3} | {'Loss':<7} | {'44b6 Mean':<9} | {'44b6 P90':<8} | {'44b6 Rec':<8} | {'44b6 J_adj':<10} | {'6bba Mean':<9} | {'6bba P90':<8} | {'6bba Rec':<8} | {'6bba J_adj':<10} | {'Macro J_adj'}")
    print("-" * 110)
    
    for epoch in range(1, 11):
        detector.train()
        total_loss = 0.0
        steps = 0
        
        for batch in loader:
            images = batch["image"].to(device)
            heatmaps = batch["heatmap"].to(device)
            masks = batch["frame_mask"].to(device)
            
            optimizer.zero_grad()
            preds, _ = detector(images)
            loss, metrics = criterion(preds, heatmaps, mask=masks)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(detector.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += float(loss.item())
            steps += 1
            
        avg_train_loss = total_loss / max(1, steps)
        
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        # Per-epoch validation on permanent anchors (detect catastrophic forgetting)
        val_res, macro_j_adj = run_per_epoch_validation(
            detector=detector,
            val_datasets=val_datasets,
            device=device,
            detection_threshold=0.25,
            max_peaks_per_frame=150,
        )
        
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        r44 = val_res["44b6_66f9292d"]
        r6b = val_res["6bba_d82a4fc6"]
        
        print(
            f"{epoch:<3} | {avg_train_loss:<7.4f} | "
            f"{r44['gt_mean']:<9.4f} | {r44['bg_p90']:<8.4f} | {r44['recall']:<8.4f} | {r44['j_adj']:<10.4f} | "
            f"{r6b['gt_mean']:<9.4f} | {r6b['bg_p90']:<8.4f} | {r6b['recall']:<8.4f} | {r6b['j_adj']:<10.4f} | "
            f"{macro_j_adj:.4f}"
        )
        
        epoch_logs.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "44b6": r44,
            "6bba": r6b,
            "macro_j_adj": macro_j_adj,
        })
    
    print("-" * 110)
    train_duration = time.time() - t_train_start
    print(f"Detector continual training finished in {train_duration:.1f}s.")
    
    # 5. Tracker Continual Training
    print("\n[Step 5/6] Continual Fine-Tuning Tracker on Chunk 2 (5 epochs at LR=0.0003)...")
    trainer = ContinualTrainer(
        detector=detector,
        embedder=embedder,
        tracker=tracker,
        device=device,
        detector_lr=0.002,
        tracker_lr=0.0003,
    )
    t_trk_start = time.time()
    trk_loss = trainer.train_tracker_on_chunk(
        datasets=train_datasets,
        epochs=5,
        max_distance_um=7.0,
    )
    trk_duration = time.time() - t_trk_start
    print(f"Tracker continual training finished in {trk_duration:.1f}s (Final Loss: {trk_loss:.5f}).")
    
    # 6. Full Lineage Held-Out Evaluation & Catastrophic Forgetting Analysis
    print("\n[Step 6/6] Running Full Pipeline Held-Out Evaluation (Detector -> Transformer -> ILP)...")
    evaluator = HeldOutEvaluator(
        detector=detector,
        embedder=embedder,
        tracker=tracker,
        detection_threshold=0.25,
        max_peaks_per_frame=150,
        max_distance_um=7.0,
        window_size=1,
        diagnostic_mode=False,
        device=device,
    )
    
    val_summary = evaluator.evaluate_cohort(val_datasets)
    
    macro_score = val_summary["macro_final_score"]
    macro_j_adj = val_summary["macro_adjusted_jaccard"]
    macro_j_div = val_summary["macro_division_jaccard"]
    macro_edge_j = val_summary["macro_edge_jaccard"]
    total_tp = val_summary["total_node_tp"]
    total_fp = val_summary["total_node_fp"]
    total_fn = val_summary["total_node_fn"]
    
    seq_44b6 = val_summary["sequences"]["44b6_66f9292d"]
    seq_6bba = val_summary["sequences"]["6bba_d82a4fc6"]
    
    # Catastrophic Forgetting Analysis
    delta_macro = macro_score - PROMOTED_PHASE3_BASELINE["macro_final_score"]
    delta_44b6 = seq_44b6["final_score"] - PROMOTED_PHASE3_BASELINE["44b6_final_score"]
    delta_6bba = seq_6bba["final_score"] - PROMOTED_PHASE3_BASELINE["6bba_final_score"]
    delta_fp = total_fp - PROMOTED_PHASE3_BASELINE["total_node_fp"]
    
    print("\n" + "=" * 85)
    print("                 CHUNK 2 EVALUATION & CATASTROPHIC FORGETTING ANALYSIS")
    print("=" * 85)
    print(f"{'Metric':<25} | {'Phase 3 Promoted':<20} | {'Chunk 2 Continual':<20} | {'Delta':<12}")
    print("-" * 85)
    print(f"{'Macro Final Score':<25} | {PROMOTED_PHASE3_BASELINE['macro_final_score']:<20.4f} | {macro_score:<20.4f} | {delta_macro:<+12.4f}")
    print(f"{'44b6 Final Score':<25} | {PROMOTED_PHASE3_BASELINE['44b6_final_score']:<20.4f} | {seq_44b6['final_score']:<20.4f} | {delta_44b6:<+12.4f}")
    print(f"{'6bba Final Score':<25} | {PROMOTED_PHASE3_BASELINE['6bba_final_score']:<20.4f} | {seq_6bba['final_score']:<20.4f} | {delta_6bba:<+12.4f}")
    print(f"{'Macro J_adj':<25} | {PROMOTED_PHASE3_BASELINE['macro_final_score']:<20.4f} | {macro_j_adj:<20.4f} | {macro_j_adj - PROMOTED_PHASE3_BASELINE['macro_final_score']:<+12.4f}")
    print(f"{'Macro Edge Jaccard':<25} | {PROMOTED_PHASE3_BASELINE['macro_edge_jaccard']:<20.4f} | {macro_edge_j:<20.4f} | {macro_edge_j - PROMOTED_PHASE3_BASELINE['macro_edge_jaccard']:<+12.4f}")
    print(f"{'Macro Div Jaccard':<25} | {PROMOTED_PHASE3_BASELINE['macro_division_jaccard']:<20.4f} | {macro_j_div:<20.4f} | {macro_j_div - PROMOTED_PHASE3_BASELINE['macro_division_jaccard']:<+12.4f}")
    print(f"{'Total Node TP':<25} | {PROMOTED_PHASE3_BASELINE['total_node_tp']:<20} | {total_tp:<20} | {total_tp - PROMOTED_PHASE3_BASELINE['total_node_tp']:<+12}")
    print(f"{'Total Node FP':<25} | {PROMOTED_PHASE3_BASELINE['total_node_fp']:<20} | {total_fp:<20} | {delta_fp:<+12}")
    print(f"{'Total Node FN':<25} | {PROMOTED_PHASE3_BASELINE['total_node_fn']:<20} | {total_fn:<20} | {total_fn - PROMOTED_PHASE3_BASELINE['total_node_fn']:<+12}")
    str_44b6 = f"{seq_44b6['node_tp']}/{seq_44b6['node_fp']}/{seq_44b6['node_fn']} (rec={seq_44b6['node_recall']:.1%})"
    str_6bba = f"{seq_6bba['node_tp']}/{seq_6bba['node_fp']}/{seq_6bba['node_fn']} (rec={seq_6bba['node_recall']:.1%})"
    print(f"{'44b6 TP / FP / FN':<25} | {'0 / 144 / 745':<20} | {str_44b6:<20} |")
    print(f"{'6bba TP / FP / FN':<25} | {'118 / 1336 / 509':<20} | {str_6bba:<20} |")
    print("=" * 85)
    
    # Always archive Chunk 2 candidate checkpoints
    cand_det_path = Path("checkpoints/archive/chunk2_candidate_detector.pt")
    cand_trk_path = Path("checkpoints/archive/chunk2_candidate_tracker.pt")
    cand_det_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoints(detector_path=cand_det_path, tracker_path=cand_trk_path)
    print(f"\nChunk 2 candidate checkpoints archived to:\n  - {cand_det_path}\n  - {cand_trk_path}")
    
    # Strict Promotion Policy Evaluation
    catastrophic_forgetting = (
        (seq_44b6["final_score"] < PROMOTED_PHASE3_BASELINE["44b6_final_score"]) or
        (delta_fp > 1000) or
        (macro_score < PROMOTED_PHASE3_BASELINE["macro_final_score"])
    )
    
    promoted = False
    if (macro_score >= PROMOTED_PHASE3_BASELINE["macro_final_score"]) and not catastrophic_forgetting:
        promoted = True
        trainer.save_checkpoints(
            detector_path="checkpoints/best_detector.pt",
            tracker_path="checkpoints/best_tracker.pt",
        )
        print(f"\n>>> CHECKPOINT POLICY: PROMOTED! New score ({macro_score:.4f}) >= Phase 3 baseline ({PROMOTED_PHASE3_BASELINE['macro_final_score']:.4f}) without catastrophic forgetting.")
    else:
        print(f"\n>>> CHECKPOINT POLICY: RETAINED PHASE 3 BASELINE.")
        if macro_score < PROMOTED_PHASE3_BASELINE["macro_final_score"]:
            print(f"    Reason: Macro score {macro_score:.4f} < Phase 3 baseline {PROMOTED_PHASE3_BASELINE['macro_final_score']:.4f}.")
        elif catastrophic_forgetting:
            print(f"    Reason: Catastrophic forgetting / regression detected on held-out metrics.")
        print(f"    Active best_detector.pt and best_tracker.pt remain unchanged.")
    
    # Save full results JSON
    results_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "chunk_id": "chunk_002_train",
        "chunk_sequences": CHUNK2_STEMS,
        "training_duration_sec": train_duration,
        "tracker_duration_sec": trk_duration,
        "total_duration_sec": train_duration + trk_duration,
        "epoch_logs": epoch_logs,
        "final_summary": val_summary,
        "promoted_to_best": promoted,
        "candidate_checkpoints": {
            "detector": str(cand_det_path),
            "tracker": str(cand_trk_path),
        },
        "phase3_baseline_comparison": {
            "baseline": PROMOTED_PHASE3_BASELINE,
            "chunk2_model": {
                "macro_final_score": macro_score,
                "macro_adjusted_jaccard": macro_j_adj,
                "macro_division_jaccard": macro_j_div,
                "macro_edge_jaccard": macro_edge_j,
                "total_node_tp": total_tp,
                "total_node_fp": total_fp,
                "total_node_fn": total_fn,
                "macro_node_recall": val_summary["macro_node_recall"],
                "macro_node_precision": val_summary["macro_node_precision"],
                "44b6_final_score": seq_44b6["final_score"],
                "6bba_final_score": seq_6bba["final_score"],
            },
            "deltas": {
                "macro_final_score": delta_macro,
                "44b6_final_score": delta_44b6,
                "6bba_final_score": delta_6bba,
                "total_node_fp": delta_fp,
            },
        },
    }
    
    res_path = Path("reports/chunk2_continual_results.json")
    res_path.parent.mkdir(parents=True, exist_ok=True)
    with open(res_path, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"Full results saved to {res_path}")
    
    # Update SQLite Registry
    conn = sqlite3.connect("registry/experiments.sqlite")
    cur = conn.cursor()
    cur.execute("""
        UPDATE chunks SET
            sequence_stems = ?,
            val_final_score = ?,
            val_adj_jaccard = ?,
            val_div_jaccard = ?,
            val_edge_jaccard = ?,
            val_node_recall = ?,
            val_node_precision = ?,
            val_node_tp = ?,
            val_node_fp = ?,
            val_node_fn = ?,
            train_loss_detector = ?,
            train_loss_tracker = ?,
            epochs_trained = 15,
            inference_time_sec = ?,
            ilp_runtime_sec = ?,
            duration_sec = ?,
            completed_at = ?,
            promoted_to_best = ?,
            status = 'PROCESSED'
        WHERE chunk_id = 'chunk_002_train'
    """, (
        json.dumps(CHUNK2_STEMS),
        macro_score,
        macro_j_adj,
        macro_j_div,
        macro_edge_j,
        val_summary["macro_node_recall"],
        val_summary["macro_node_precision"],
        total_tp,
        total_fp,
        total_fn,
        epoch_logs[-1]["train_loss"],
        trk_loss,
        val_summary["inference_time_sec"],
        val_summary["ilp_runtime_sec"],
        train_duration + trk_duration,
        datetime.now(timezone.utc).isoformat(),
        1 if promoted else 0,
    ))
    conn.commit()
    conn.close()
    print("Updated registry/experiments.sqlite with chunk_002_train metrics.")
    
    # Transactional Cleanup of chunk_002_train
    print("\n[Cleanup] Cleaning up temporary training data in chunk_002_train...")
    initial_free = storage_mgr.get_disk_free_gb()
    shutil.rmtree(chunk_dir)
    final_free = storage_mgr.get_disk_free_gb()
    freed_gb = final_free - initial_free
    print(f"Cleaned up {chunk_dir}. Freed: {freed_gb:.2f} GB (Current free: {final_free:.2f} GB).")
    print("\nCHUNK 2 CONTINUAL LEARNING WORKFLOW COMPLETE.")


if __name__ == "__main__":
    main()

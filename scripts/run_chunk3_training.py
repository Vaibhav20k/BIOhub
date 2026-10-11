#!/usr/bin/env python3
"""Phase 5 — Chunk 3 Continual Learning on Kaggle GPU.

Continual learning on Chunk 3 starting from the promoted Chunk 2 checkpoints:
- best_detector.pt (SHA256: 6e9a64c754551402d45c30b7a6949d04f2d2d7e40a3f076243581037a130eaab)
- best_tracker.pt (SHA256: ed3a4498461da21dbd7b39943eece75f06b6b0fa8faaf570c3aa93403994baf5)

Chunk 3 Sequences (8 sequences: 3 from 44b6, 5 from 6bba):
- 44b6: 44b6_0c582fdc, 44b6_c96cfa10, 44b6_996155de
- 6bba: 6bba_61ecbe65, 6bba_cf35214c, 6bba_b204cac7, 6bba_d0fc38b5, 6bba_ebdf3b34

Domain-Balanced Quota:
- 400 patches to 44b6 (134, 133, 133 across 3 seqs)
- 400 patches to 6bba (80 across 5 seqs)
- Total 800 patches/epoch

Protocol:
- Detector: TemporalUNet3D (W=1, spatial-only, base_channels=16, feature_dim=32, prior_prob=0.05)
- Training: LR=0.002, 10 epochs, batch=2, FP32, patch=32×128×128, 800 patches/epoch
- Augmentation: contrast ±15%, brightness ±15%, background offset ±10%
- Loss: Focal + Dice (1.0/1.0)
- Per-epoch validation on permanent anchors (44b6_66f9292d & 6bba_d82a4fc6)
- Tracker: 5 epochs continual fine-tuning on Chunk 3 at LR=0.0003
- Full end-to-end HeldOutEvaluator pipeline on permanent anchors
- Checkpoint comparison vs Promoted Chunk 2 Baseline (Macro = 0.03865588)
- Per-epoch durable checkpointing to /kaggle/working/checkpoints/
- Transactional cleanup of chunk_003_train symlinks
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
import zipfile
from typing import Any, Dict, List, Optional, Tuple

print("=" * 80)
print("BIOHUB CHUNK 3 — PHASE 5 CONTINUAL LEARNING WORKLOAD (KAGGLE GPU)")
print("=" * 80)

# Step 0: Install dependencies
print("\n[Step 0/6] Installing required dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "zarr>=3.0.0", "pyscipopt", "tracksdata"])
print("  Dependencies installed successfully.")

# Step 1: Locate and set up codebase workspace
print("\n[Step 1/6] Setting up codebase workspace...")
work_dir = Path("/kaggle/working/biohub")
work_dir.mkdir(parents=True, exist_ok=True)

# First check if zip files were mounted from the dataset
found_zip = False
for root, dirs, files in os.walk("/kaggle/input"):
    if "checkpoints.zip" in files or "src.zip" in files:
        print(f"  Found dataset zips in {root}")
        for zf_name in ["checkpoints.zip", "src.zip", "configs.zip", "registry.zip", "scripts.zip"]:
            zf_path = Path(root) / zf_name
            if zf_path.exists():
                with zipfile.ZipFile(zf_path, 'r') as zip_ref:
                    zip_ref.extractall(work_dir)
                print(f"    Extracted {zf_name}")
        for other_file in ["pyproject.toml", "dataset_catalog.json"]:
            src_other = Path(root) / other_file
            if src_other.exists():
                shutil.copy2(src_other, work_dir / other_file)
        found_zip = True
        break

if not found_zip:
    code_input_dir = None
    for root, dirs, files in os.walk("/kaggle/input"):
        if "best_detector.pt" in files:
            p = Path(root)
            code_input_dir = p.parent if p.name == "checkpoints" else p
            print(f"  Found codebase at: {code_input_dir}")
            break
    if code_input_dir is not None:
        shutil.copytree(code_input_dir, work_dir, dirs_exist_ok=True)
        print(f"  Copied code and checkpoints to: {work_dir}")

os.chdir(work_dir)
sys.path.insert(0, str(work_dir))

# Ensure working checkpoints and reports directories exist
working_ckpt_dir = Path("/kaggle/working/checkpoints")
working_ckpt_dir.mkdir(parents=True, exist_ok=True)
working_reports_dir = Path("/kaggle/working/reports")
working_reports_dir.mkdir(parents=True, exist_ok=True)

import numpy as np
import polars as pl
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, ConcatDataset

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

CHUNK3_STEMS = [
    "6bba_61ecbe65", "44b6_0c582fdc", "44b6_c96cfa10", "6bba_cf35214c",
    "6bba_b204cac7", "44b6_996155de", "6bba_d0fc38b5", "6bba_ebdf3b34"
]
ANCHOR_STEMS = ["44b6_66f9292d", "6bba_d82a4fc6"]

PROMOTED_CHUNK2_BASELINE = {
    "macro_final_score": 0.03865588,
    "44b6_final_score": 0.0000,
    "6bba_final_score": 0.07731176,
    "total_node_tp": 186,
    "total_node_fp": 3511,
    "total_node_fn": 559,
    "macro_adjusted_jaccard": 0.03865588,
    "macro_edge_jaccard": 0.05295192,
    "macro_division_jaccard": 0.0,
}

EXPECTED_PROMOTED_DETECTOR_SHA256 = "6e9a64c754551402d45c30b7a6949d04f2d2d7e40a3f076243581037a130eaab"
EXPECTED_PROMOTED_TRACKER_SHA256 = "ed3a4498461da21dbd7b39943eece75f06b6b0fa8faaf570c3aa93403994baf5"

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
    assert torch.cuda.is_available(), "CUDA GPU required for Phase 5 Chunk 3!"
    device = torch.device("cuda:0")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    
    # 1. Pre-flight Checks
    raw_train_dir = Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/train")
    if not raw_train_dir.exists():
        for root, dirs, files in os.walk("/kaggle/input"):
            if "44b6_66f9292d.zarr" in dirs:
                raw_train_dir = Path(root)
                break
    print(f"Competition train dir: {raw_train_dir}")
    assert raw_train_dir.exists(), f"Train dir not found: {raw_train_dir}"
    
    data_dir = work_dir / "data"
    val_dir = data_dir / "val"
    chunk_dir = data_dir / "chunks/chunk_003_train"
    val_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    
    storage_mgr = ChunkStorageManager(staging_dir=data_dir / "chunks", min_free_disk_gb=2.0)
    print("\n[Step 1/6] Running Pre-flight Integrity & Checkpoint Hash Checks...")
    print(f"  Disk Free: {storage_mgr.get_disk_free_gb():.2f} GB (Safety floor: {storage_mgr.min_free_disk_gb} GB)")
    storage_mgr.assert_disk_safety(required_extra_gb=1.0)
    
    # Check promoted checkpoints
    best_det_path = work_dir / "checkpoints/best_detector.pt"
    best_trk_path = work_dir / "checkpoints/best_tracker.pt"
    assert best_det_path.exists(), f"Active detector checkpoint missing at {best_det_path}"
    assert best_trk_path.exists(), f"Active tracker checkpoint missing at {best_trk_path}"
    
    det_hash = file_sha256(best_det_path)
    trk_hash = file_sha256(best_trk_path)
    print(f"  Active detector hash: {det_hash}")
    print(f"  Active tracker hash:  {trk_hash}")
    assert det_hash == EXPECTED_PROMOTED_DETECTOR_SHA256, f"Detector hash mismatch! Got {det_hash}, expected {EXPECTED_PROMOTED_DETECTOR_SHA256}"
    assert trk_hash == EXPECTED_PROMOTED_TRACKER_SHA256, f"Tracker hash mismatch! Got {trk_hash}, expected {EXPECTED_PROMOTED_TRACKER_SHA256}"
    print("  Active checkpoints confirmed matching promoted Chunk 2 baseline hashes.")
    
    # Symlink and verify validation anchors
    val_datasets = []
    print("  Verifying validation anchors:")
    for stem in ANCHOR_STEMS:
        src_z = raw_train_dir / f"{stem}.zarr"
        src_g = raw_train_dir / f"{stem}.geff"
        dst_z = val_dir / f"{stem}.zarr"
        dst_g = val_dir / f"{stem}.geff"
        if not dst_z.exists(): os.symlink(src_z, dst_z)
        if not dst_g.exists(): os.symlink(src_g, dst_g)
        
        valid, err = storage_mgr.verify_sequence_integrity(dst_z, require_tracks=True)
        assert valid, f"Validation sequence {stem} integrity failure: {err}"
        vol = open_dataset(dst_z, load_tracks=True)
        val_datasets.append(vol)
        print(f"    ✓ {stem}: zarr shape={vol.shape}, nodes={len(vol.tracks.node_attrs())}")
    
    # Symlink and verify Chunk 3 training sequences
    train_datasets = []
    print("  Verifying Chunk 3 sequences:")
    for stem in CHUNK3_STEMS:
        src_z = raw_train_dir / f"{stem}.zarr"
        src_g = raw_train_dir / f"{stem}.geff"
        dst_z = chunk_dir / f"{stem}.zarr"
        dst_g = chunk_dir / f"{stem}.geff"
        if not dst_z.exists(): os.symlink(src_z, dst_z)
        if not dst_g.exists(): os.symlink(src_g, dst_g)
        
        valid, err = storage_mgr.verify_sequence_integrity(dst_z, require_tracks=True)
        assert valid, f"Training sequence {stem} integrity failure: {err}"
        vol = open_dataset(dst_z, load_tracks=True, require_tracks=True)
        train_datasets.append(vol)
        print(f"    ✓ {stem}: zarr shape={vol.shape}, nodes={len(vol.tracks.node_attrs())}")
    
    print(f"  All {len(train_datasets)} Chunk 3 training sequences verified intact.")
    
    # 2. Patch Budget Allocation (50/50 Domain-Balanced)
    print("\n[Step 2/6] Verifying Domain-Balanced Quota Allocation (50/50 for 44b6 and 6bba)...")
    by_embryo: Dict[str, List[DatasetVolume]] = {}
    for ds in train_datasets:
        prefix = ds.name.split("_")[0]
        by_embryo.setdefault(prefix, []).append(ds)
    
    total_budget = 800
    dataset_sample_quotas: Dict[str, int] = {}
    num_embryos = len(by_embryo)
    quota_per_group = total_budget // num_embryos
    
    for g_name, g_datasets in sorted(by_embryo.items()):
        base_alloc = quota_per_group // len(g_datasets)
        rem_alloc = quota_per_group % len(g_datasets)
        for i, ds in enumerate(g_datasets):
            dataset_sample_quotas[ds.name] = base_alloc + (1 if i < rem_alloc else 0)
    
    domain_totals = {}
    for ds in train_datasets:
        p = ds.name.split("_")[0]
        domain_totals[p] = domain_totals.get(p, 0) + dataset_sample_quotas[ds.name]
    
    print(f"  Allocated Quotas per Sequence: {dataset_sample_quotas}")
    print(f"  Domain Totals: {domain_totals}")
    assert domain_totals.get("44b6") == 400, f"Expected 400 for 44b6, got {domain_totals.get('44b6')}"
    assert domain_totals.get("6bba") == 400, f"Expected 400 for 6bba, got {domain_totals.get('6bba')}"
    total_allocated = sum(dataset_sample_quotas.values())
    assert total_allocated == 800, f"Expected 800 total patches, got {total_allocated}"
    print("  Domain quota verification PASSED: exactly 400 44b6 + 400 6bba patches.")
    
    # 3. Model Loading (Warm-start from promoted Chunk 2 checkpoints)
    print("\n[Step 3/6] Loading Promoted Chunk 2 Checkpoints (Warm-Start Continual Learning)...")
    torch.manual_seed(42)
    np.random.seed(42)
    
    det_ckpt = torch.load(best_det_path, map_location=device, weights_only=False)
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
    
    trk_ckpt = torch.load(best_trk_path, map_location=device, weights_only=False)
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
    print("\n[Step 4/6] Executing 10 Epochs of Detector Continual Training on Chunk 3 (50/50 Domain-Balanced)...")
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
    loader = DataLoader(concat_ds, batch_size=2, shuffle=True, drop_last=False, num_workers=0, pin_memory=True)
    
    optimizer = AdamW(detector.parameters(), lr=0.002, weight_decay=1e-4)
    criterion = DetectionLoss(focal_weight=1.0, dice_weight=1.0)
    
    epoch_logs = []
    t_train_start = time.time()
    best_epoch_num = 1
    best_epoch_macro_j_adj = -1.0
    
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
        torch.cuda.empty_cache()
        
        # Per-epoch validation on permanent anchors
        val_res, macro_j_adj = run_per_epoch_validation(
            detector=detector,
            val_datasets=val_datasets,
            device=device,
            detection_threshold=0.25,
            max_peaks_per_frame=150,
        )
        torch.cuda.empty_cache()
        
        r44 = val_res["44b6_66f9292d"]
        r6b = val_res["6bba_d82a4fc6"]
        
        if macro_j_adj > best_epoch_macro_j_adj:
            best_epoch_macro_j_adj = macro_j_adj
            best_epoch_num = epoch
        
        print(
            f"{epoch:<3} | {avg_train_loss:<7.4f} | "
            f"{r44['gt_mean']:<9.4f} | {r44['bg_p90']:<8.4f} | {r44['recall']:<8.4f} | {r44['j_adj']:<10.4f} | "
            f"{r6b['gt_mean']:<9.4f} | {r6b['bg_p90']:<8.4f} | {r6b['recall']:<8.4f} | {r6b['j_adj']:<10.4f} | "
            f"{macro_j_adj:.4f}"
        )
        
        ep_entry = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "44b6": r44,
            "6bba": r6b,
            "macro_j_adj": macro_j_adj,
        }
        epoch_logs.append(ep_entry)
        
        # Durable per-epoch checkpointing directly to /kaggle/working/checkpoints/
        ep_ckpt_path = working_ckpt_dir / f"epoch_{epoch:02d}_detector.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": detector.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "macro_j_adj": macro_j_adj,
            "val_res": val_res,
            "config": {
                "base_channels": 16,
                "feature_dim": 32,
                "use_temporal_attention": False,
            },
        }, ep_ckpt_path)
        with open(working_ckpt_dir / f"epoch_{epoch:02d}_metrics.json", "w") as f:
            json.dump(ep_entry, f, indent=2)
        print(f"  --> [Durable Checkpoint] Saved {ep_ckpt_path.name}")
    
    print("-" * 110)
    train_duration = time.time() - t_train_start
    print(f"Detector continual training finished in {train_duration:.1f}s.")
    print(f"Peak Validation Detector: Epoch {best_epoch_num} (Macro J_adj = {best_epoch_macro_j_adj:.4f})")
    
    # 5. Tracker Continual Training
    print("\n[Step 5/6] Continual Fine-Tuning Tracker on Chunk 3 (5 epochs at LR=0.0003)...")
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
    
    # Durable tracker checkpoint
    trk_ckpt_path = working_ckpt_dir / "tracker_continual_5ep.pt"
    torch.save({
        "tracker_state_dict": tracker.state_dict(),
        "embedder_state_dict": embedder.state_dict(),
        "final_trk_loss": trk_loss,
        "config": {"hidden_dim": 128, "num_layers": 2},
    }, trk_ckpt_path)
    print(f"  --> [Durable Checkpoint] Saved {trk_ckpt_path.name}")
    
    # 6. Full Lineage Held-Out Evaluation & Catastrophic Forgetting Analysis
    print("\n[Step 6/6] Running Full Pipeline Held-Out Evaluation (Detector -> Transformer -> ILP)...")
    
    # First: evaluate final epoch (Epoch 10) detector
    evaluator_ep10 = HeldOutEvaluator(
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
    print("  Evaluating Epoch 10 detector on permanent anchors...")
    val_summary_ep10 = evaluator_ep10.evaluate_cohort(val_datasets)
    print(f"  Epoch 10 Evaluation: Macro Final Score = {val_summary_ep10['macro_final_score']:.4f}")
    
    # If peak epoch is not epoch 10, also evaluate peak epoch detector
    chosen_epoch = 10
    chosen_summary = val_summary_ep10
    chosen_detector_path = working_ckpt_dir / "epoch_10_detector.pt"
    
    if best_epoch_num != 10:
        print(f"  Peak validation detector was Epoch {best_epoch_num}. Evaluating on full lineage...")
        peak_det_path = working_ckpt_dir / f"epoch_{best_epoch_num:02d}_detector.pt"
        peak_ckpt = torch.load(peak_det_path, map_location=device, weights_only=False)
        detector_peak = TemporalUNet3D(
            in_channels=1,
            base_channels=16,
            feature_dim=32,
            prior_prob=0.05,
            use_temporal_attention=False,
        ).to(device)
        detector_peak.load_state_dict(peak_ckpt["model_state_dict"])
        
        evaluator_peak = HeldOutEvaluator(
            detector=detector_peak,
            embedder=embedder,
            tracker=tracker,
            detection_threshold=0.25,
            max_peaks_per_frame=150,
            max_distance_um=7.0,
            window_size=1,
            diagnostic_mode=False,
            device=device,
        )
        val_summary_peak = evaluator_peak.evaluate_cohort(val_datasets)
        print(f"  Epoch {best_epoch_num} Evaluation: Macro Final Score = {val_summary_peak['macro_final_score']:.4f}")
        
        if val_summary_peak["macro_final_score"] > val_summary_ep10["macro_final_score"]:
            print(f"  --> Selecting Epoch {best_epoch_num} as candidate detector (higher macro score).")
            chosen_epoch = best_epoch_num
            chosen_summary = val_summary_peak
            chosen_detector_path = peak_det_path
            # Set active detector weights to peak
            detector.load_state_dict(peak_ckpt["model_state_dict"])
        else:
            print(f"  --> Retaining Epoch 10 as candidate detector.")
    
    macro_score = chosen_summary["macro_final_score"]
    macro_j_adj = chosen_summary["macro_adjusted_jaccard"]
    macro_j_div = chosen_summary["macro_division_jaccard"]
    macro_edge_j = chosen_summary["macro_edge_jaccard"]
    total_tp = chosen_summary["total_node_tp"]
    total_fp = chosen_summary["total_node_fp"]
    total_fn = chosen_summary["total_node_fn"]
    
    seq_44b6 = chosen_summary["sequences"]["44b6_66f9292d"]
    seq_6bba = chosen_summary["sequences"]["6bba_d82a4fc6"]
    
    # Catastrophic Forgetting & Regression Analysis
    delta_macro = macro_score - PROMOTED_CHUNK2_BASELINE["macro_final_score"]
    delta_44b6 = seq_44b6["final_score"] - PROMOTED_CHUNK2_BASELINE["44b6_final_score"]
    delta_6bba = seq_6bba["final_score"] - PROMOTED_CHUNK2_BASELINE["6bba_final_score"]
    delta_fp = total_fp - PROMOTED_CHUNK2_BASELINE["total_node_fp"]
    
    print("\n" + "=" * 85)
    print(f"          CHUNK 3 EVALUATION & CATASTROPHIC FORGETTING ANALYSIS (Candidate Ep {chosen_epoch})")
    print("=" * 85)
    print(f"{'Metric':<25} | {'Chunk 2 Baseline':<20} | {'Chunk 3 Candidate':<20} | {'Delta':<12}")
    print("-" * 85)
    print(f"{'Macro Final Score':<25} | {PROMOTED_CHUNK2_BASELINE['macro_final_score']:<20.4f} | {macro_score:<20.4f} | {delta_macro:<+12.4f}")
    print(f"{'44b6 Final Score':<25} | {PROMOTED_CHUNK2_BASELINE['44b6_final_score']:<20.4f} | {seq_44b6['final_score']:<20.4f} | {delta_44b6:<+12.4f}")
    print(f"{'6bba Final Score':<25} | {PROMOTED_CHUNK2_BASELINE['6bba_final_score']:<20.4f} | {seq_6bba['final_score']:<20.4f} | {delta_6bba:<+12.4f}")
    print(f"{'Macro J_adj':<25} | {PROMOTED_CHUNK2_BASELINE['macro_adjusted_jaccard']:<20.4f} | {macro_j_adj:<20.4f} | {macro_j_adj - PROMOTED_CHUNK2_BASELINE['macro_adjusted_jaccard']:<+12.4f}")
    print(f"{'Macro Edge Jaccard':<25} | {PROMOTED_CHUNK2_BASELINE['macro_edge_jaccard']:<20.4f} | {macro_edge_j:<20.4f} | {macro_edge_j - PROMOTED_CHUNK2_BASELINE['macro_edge_jaccard']:<+12.4f}")
    print(f"{'Macro Div Jaccard':<25} | {PROMOTED_CHUNK2_BASELINE['macro_division_jaccard']:<20.4f} | {macro_j_div:<20.4f} | {macro_j_div - PROMOTED_CHUNK2_BASELINE['macro_division_jaccard']:<+12.4f}")
    print(f"{'Total Node TP':<25} | {PROMOTED_CHUNK2_BASELINE['total_node_tp']:<20} | {total_tp:<20} | {total_tp - PROMOTED_CHUNK2_BASELINE['total_node_tp']:<+12}")
    print(f"{'Total Node FP':<25} | {PROMOTED_CHUNK2_BASELINE['total_node_fp']:<20} | {total_fp:<20} | {delta_fp:<+12}")
    print(f"{'Total Node FN':<25} | {PROMOTED_CHUNK2_BASELINE['total_node_fn']:<20} | {total_fn:<20} | {total_fn - PROMOTED_CHUNK2_BASELINE['total_node_fn']:<+12}")
    str_44b6 = f"{seq_44b6['node_tp']}/{seq_44b6['node_fp']}/{seq_44b6['node_fn']} (rec={seq_44b6['node_recall']:.1%})"
    str_6bba = f"{seq_6bba['node_tp']}/{seq_6bba['node_fp']}/{seq_6bba['node_fn']} (rec={seq_6bba['node_recall']:.1%})"
    print(f"{'44b6 TP / FP / FN':<25} | {'0 / 2277 / 200':<20} | {str_44b6:<20} |")
    print(f"{'6bba TP / FP / FN':<25} | {'186 / 1234 / 359':<20} | {str_6bba:<20} |")
    print("=" * 85)
    
    # Candidate checkpoints
    cand_det_path = working_ckpt_dir / "chunk3_candidate_detector.pt"
    cand_trk_path = working_ckpt_dir / "chunk3_candidate_tracker.pt"
    trainer.save_checkpoints(detector_path=cand_det_path, tracker_path=cand_trk_path)
    print(f"\nChunk 3 candidate checkpoints archived to:\n  - {cand_det_path}\n  - {cand_trk_path}")
    
    # Strict Promotion Policy Evaluation
    # Regression threshold: permanent anchor score drops by > 15%
    rel_6bba_drop = (PROMOTED_CHUNK2_BASELINE["6bba_final_score"] - seq_6bba["final_score"]) / max(PROMOTED_CHUNK2_BASELINE["6bba_final_score"], 1e-6)
    
    material_regression = (
        (seq_44b6["final_score"] < PROMOTED_CHUNK2_BASELINE["44b6_final_score"]) or
        (rel_6bba_drop > 0.15) or
        (macro_score < PROMOTED_CHUNK2_BASELINE["macro_final_score"])
    )
    
    promoted = False
    if (macro_score >= PROMOTED_CHUNK2_BASELINE["macro_final_score"]) and not material_regression:
        promoted = True
        trainer.save_checkpoints(
            detector_path=working_ckpt_dir / "best_detector.pt",
            tracker_path=working_ckpt_dir / "best_tracker.pt",
        )
        print(f"\n>>> CHECKPOINT POLICY: PROMOTED! New score ({macro_score:.4f}) >= Chunk 2 baseline ({PROMOTED_CHUNK2_BASELINE['macro_final_score']:.4f}) without anchor regression.")
    else:
        print(f"\n>>> CHECKPOINT POLICY: RETAINED CHUNK 2 BASELINE.")
        if macro_score < PROMOTED_CHUNK2_BASELINE["macro_final_score"]:
            print(f"    Reason: Macro score {macro_score:.4f} < Chunk 2 baseline {PROMOTED_CHUNK2_BASELINE['macro_final_score']:.4f}.")
        elif material_regression:
            print(f"    Reason: Anchor regression detected (6bba relative drop: {rel_6bba_drop:.2%}).")
        print(f"    Active best_detector.pt and best_tracker.pt remain unchanged.")
    
    # Save full results JSON
    results_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "chunk_id": "chunk_003_train",
        "chunk_sequences": CHUNK3_STEMS,
        "domain_distribution": {
            "44b6": ["44b6_0c582fdc", "44b6_c96cfa10", "44b6_996155de"],
            "6bba": ["6bba_61ecbe65", "6bba_cf35214c", "6bba_b204cac7", "6bba_d0fc38b5", "6bba_ebdf3b34"],
            "sample_quotas": dataset_sample_quotas,
        },
        "chosen_detector_epoch": chosen_epoch,
        "training_duration_sec": train_duration,
        "tracker_duration_sec": trk_duration,
        "total_duration_sec": train_duration + trk_duration,
        "epoch_logs": epoch_logs,
        "final_summary": chosen_summary,
        "epoch_10_summary": val_summary_ep10,
        "promoted_to_best": promoted,
        "candidate_checkpoints": {
            "detector": str(cand_det_path),
            "tracker": str(cand_trk_path),
            "detector_sha256": file_sha256(cand_det_path),
            "tracker_sha256": file_sha256(cand_trk_path),
        },
        "chunk2_baseline_comparison": {
            "baseline": PROMOTED_CHUNK2_BASELINE,
            "chunk3_model": {
                "macro_final_score": macro_score,
                "macro_adjusted_jaccard": macro_j_adj,
                "macro_division_jaccard": macro_j_div,
                "macro_edge_jaccard": macro_edge_j,
                "total_node_tp": total_tp,
                "total_node_fp": total_fp,
                "total_node_fn": total_fn,
                "macro_node_recall": chosen_summary["macro_node_recall"],
                "macro_node_precision": chosen_summary["macro_node_precision"],
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
    
    res_path = working_reports_dir / "chunk3_continual_results.json"
    with open(res_path, "w") as f:
        json.dump(results_payload, f, indent=2)
    with open("/kaggle/working/chunk3_continual_results.json", "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"Full results saved to {res_path}")
    
    # Update SQLite Registry
    conn = sqlite3.connect(work_dir / "registry/experiments.sqlite")
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
        WHERE chunk_id = 'chunk_003_train'
    """, (
        json.dumps(CHUNK3_STEMS),
        macro_score,
        macro_j_adj,
        macro_j_div,
        macro_edge_j,
        chosen_summary["macro_node_recall"],
        chosen_summary["macro_node_precision"],
        total_tp,
        total_fp,
        total_fn,
        epoch_logs[-1]["train_loss"],
        trk_loss,
        chosen_summary["inference_time_sec"],
        chosen_summary["ilp_runtime_sec"],
        train_duration + trk_duration,
        datetime.now(timezone.utc).isoformat(),
        1 if promoted else 0,
    ))
    conn.commit()
    conn.close()
    shutil.copy(work_dir / "registry/experiments.sqlite", "/kaggle/working/experiments.sqlite")
    print("Updated registry/experiments.sqlite and copied to /kaggle/working/experiments.sqlite.")
    
    # Cleanup chunk_003_train symlinks
    print("\n[Cleanup] Cleaning up symlinks in chunk_003_train...")
    shutil.rmtree(chunk_dir)
    print("CHUNK 3 CONTINUAL LEARNING COMPLETED SUCCESSFULLY.")
    print("=" * 80)
    print(f"Final Macro Score: {macro_score:.4f} (Chunk 2 Baseline: {PROMOTED_CHUNK2_BASELINE['macro_final_score']:.4f})")
    print(f"Promoted to Best: {promoted}")
    print("=" * 80)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase 3 — Controlled Chunk 1 Retraining Script.

Re-runs Chunk 1 from scratch with:
- Domain-balanced sampling (exactly 400 patches from 44b6, 400 patches from 6bba)
- Decoupled photometric augmentation (contrast jitter ±15%, brightness ±15%, background offset ±10%)
- Locked architecture: TemporalUNet3D (W=1, base_channels=16, feature_dim=32, prior_prob=0.05, bias=-2.944)
- 10 detector epochs (batch=2, LR=0.002, AdamW, FP32, Focal+Dice 1.0/1.0)
- Per-epoch validation monitoring:
  * 44b6 vs 6bba GT activation mean, median, P90
  * Background percentiles (P50, P90, P99, max)
  * Contrast ratio
  * Node TP, FP, FN, precision, recall, Jaccard, J_adj (threshold=0.25, max_peaks=150, dist<=7.0um)
- Tracker training for 5 epochs on Chunk 1
- Full end-to-end HeldOutEvaluator pipeline on permanent anchors (44b6_66f9292d and 6bba_d82a4fc6)
- A/B comparison against baseline (0.0182)
- Strict promotion/retention checkpoint policy
- Transactional cleanup of chunk_001_train
"""

import argparse
from datetime import datetime, timezone
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

CHUNK1_STEMS = [
    "6bba_3fda6b25",
    "6bba_9a41d029",
    "44b6_949adeb1",
    "44b6_5740d24b",
    "6bba_05b6850b",
    "6bba_f4ae811c",
    "6bba_b1ae37b9",
    "6bba_7b5d3b2c",
]

BASELINE_METRICS = {
    "macro_final_score": 0.01817,
    "44b6_final_score": 0.0000,
    "6bba_final_score": 0.03634,
    "total_node_tp": 106,
    "total_node_fp": 2571,
    "total_node_fn": 639,
    "macro_node_recall": 0.0972,
    "macro_node_precision": 0.0328,
    "macro_edge_jaccard": 0.0262,
    "macro_division_jaccard": 0.0,
}


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
        
        # Deterministic validation sampling
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
                with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    pred_hm, _ = detector(inp)
                hm_3d = pred_hm[0, 0].cpu().numpy().astype(np.float32)
                
                # GT cell activations for frame t
                frame_gt = gt_nodes.filter(pl.col("t") == t)
                for row in frame_gt.iter_rows(named=True):
                    z = int(np.clip(round(row["z"]), 0, hm_3d.shape[0] - 1))
                    y = int(np.clip(round(row["y"]), 0, hm_3d.shape[1] - 1))
                    x = int(np.clip(round(row["x"]), 0, hm_3d.shape[2] - 1))
                    gt_cell_vals.append(float(hm_3d[z, y, x]))
                
                # Sample background voxels (subsample 1000 voxels from frame)
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
    print("            PHASE 3: CONTROLLED CHUNK 1 RETRAINING (DOMAIN-BALANCED)")
    print("================================================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 1. Pre-flight Checks
    chunk_dir = Path("data/chunks/chunk_001_train")
    val_dir = Path("data/val")
    storage_mgr = ChunkStorageManager()
    
    print("\n[Step 1/6] Running Pre-flight Integrity & Backup Checks...")
    print(f"  Disk Free: {storage_mgr.get_disk_free_gb():.2f} GB (Threshold: {storage_mgr.min_free_disk_gb} GB)")
    storage_mgr.assert_disk_safety(required_extra_gb=1.0)
    
    # Check golden checkpoints
    best_det_backup = Path("checkpoints/archive/chunk1_baseline_pre_domain_balance_detector.pt")
    best_trk_backup = Path("checkpoints/archive/chunk1_baseline_pre_domain_balance_tracker.pt")
    assert best_det_backup.exists(), f"Golden detector backup missing at {best_det_backup}"
    assert best_trk_backup.exists(), f"Golden tracker backup missing at {best_trk_backup}"
    print("  Golden baseline checkpoints successfully verified in checkpoints/archive/.")
    
    # Verify validation anchors
    val_stems = ["44b6_66f9292d", "6bba_d82a4fc6"]
    val_datasets = []
    for stem in val_stems:
        vp = val_dir / stem
        valid, err = storage_mgr.verify_sequence_integrity(vp, require_tracks=True)
        assert valid, f"Validation sequence {stem} integrity failure: {err}"
        val_datasets.append(open_dataset(vp, load_tracks=True))
    print(f"  Validation anchors verified: {[ds.name for ds in val_datasets]}")
    
    # Verify Chunk 1 training sequences
    train_datasets = []
    for stem in CHUNK1_STEMS:
        sp = chunk_dir / stem
        valid, err = storage_mgr.verify_sequence_integrity(sp, require_tracks=True)
        assert valid, f"Training sequence {stem} integrity failure: {err}"
        train_datasets.append(open_dataset(sp, load_tracks=True, require_tracks=True))
    print(f"  All {len(train_datasets)} Chunk 1 training sequences verified intact.")
    
    # 2. Assert and Verify 400/400 Domain Allocation
    print("\n[Step 2/6] Verifying Domain-Balanced Quota Allocation...")
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
    print("  Domain quota verification PASSED: exactly 400 44b6 + 400 6bba patches.")
    
    # 3. Model Initialization (Fresh locked architecture)
    print("\n[Step 3/6] Initializing Fresh Architecture (Candidate A: W=1, prior_prob=0.05)...")
    torch.manual_seed(42)
    np.random.seed(42)
    
    detector = TemporalUNet3D(
        in_channels=1,
        base_channels=16,
        feature_dim=32,
        prior_prob=0.05,
        use_temporal_attention=False,
    ).to(device)
    
    init_bias = detector.heatmap_head.bias.item()
    print(f"  Detector Heatmap Head Bias: {init_bias:.4f} (expected ~ -2.944)")
    assert abs(init_bias - (-2.944)) < 0.05, f"Unexpected init bias: {init_bias}"
    
    embedder = CellNodeEmbedding(in_visual_dim=32, embedding_dim=128).to(device)
    tracker = SpatioTemporalTracker(node_dim=128, rel_dim=32, hidden_dim=128, num_layers=2).to(device)
    
    # 4. Detector Training with Per-Epoch Validation
    print("\n[Step 4/6] Executing 10 Epochs of Detector Training...")
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
        
        # Free training cache before validation inference
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        # Per-epoch validation
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
    print(f"Detector training finished in {train_duration:.1f}s.")
    
    # 5. Tracker Training
    print("\n[Step 5/6] Training Spatio-Temporal Transformer Tracker on Chunk 1 (5 epochs)...")
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
    print(f"Tracker training finished in {trk_duration:.1f}s (Final Loss: {trk_loss:.5f}).")
    
    # 6. Full Lineage Evaluation & A/B Comparison
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
    
    print("\n" + "=" * 80)
    print("                    FINAL EVALUATION RESULTS & A/B COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<25} | {'Baseline (Pre-Balance)':<25} | {'Retrained (Domain-Balanced)':<25}")
    print("-" * 80)
    print(f"{'Macro Final Score':<25} | {BASELINE_METRICS['macro_final_score']:<25.4f} | {macro_score:<25.4f}")
    print(f"{'44b6 Final Score':<25} | {BASELINE_METRICS['44b6_final_score']:<25.4f} | {seq_44b6['final_score']:<25.4f}")
    print(f"{'6bba Final Score':<25} | {BASELINE_METRICS['6bba_final_score']:<25.4f} | {seq_6bba['final_score']:<25.4f}")
    print(f"{'Macro J_adj':<25} | {BASELINE_METRICS['macro_final_score']:<25.4f} | {macro_j_adj:<25.4f}")
    print(f"{'Macro Edge Jaccard':<25} | {BASELINE_METRICS['macro_edge_jaccard']:<25.4f} | {macro_edge_j:<25.4f}")
    print(f"{'Macro Div Jaccard':<25} | {BASELINE_METRICS['macro_division_jaccard']:<25.4f} | {macro_j_div:<25.4f}")
    print(f"{'Total Node TP':<25} | {BASELINE_METRICS['total_node_tp']:<25} | {total_tp:<25}")
    print(f"{'Total Node FP':<25} | {BASELINE_METRICS['total_node_fp']:<25} | {total_fp:<25}")
    print(f"{'Total Node FN':<25} | {BASELINE_METRICS['total_node_fn']:<25} | {total_fn:<25}")
    str_44b6 = f"{seq_44b6['node_tp']} / {seq_44b6['node_fp']} / {seq_44b6['node_fn']} (rec={seq_44b6['node_recall']:.1%})"
    str_6bba = f"{seq_6bba['node_tp']} / {seq_6bba['node_fp']} / {seq_6bba['node_fn']} (rec={seq_6bba['node_recall']:.1%})"
    print(f"{'44b6 TP / FP / FN':<25} | {'0 / 304 / 745 (rec=0.0%)':<25} | {str_44b6:<25}")
    print(f"{'6bba TP / FP / FN':<25} | {'106 / 2267 / - (rec=19.4%)':<25} | {str_6bba:<25}")
    print("=" * 80)
    
    # Archive experimental checkpoints
    exp_det_path = Path("checkpoints/archive/chunk1_domain_balanced_detector.pt")
    exp_trk_path = Path("checkpoints/archive/chunk1_domain_balanced_tracker.pt")
    trainer.save_checkpoints(detector_path=exp_det_path, tracker_path=exp_trk_path)
    print(f"Experimental checkpoints archived to {exp_det_path} and {exp_trk_path}.")
    
    # Strict Checkpoint Promotion Policy
    promoted = False
    if macro_score >= BASELINE_METRICS["macro_final_score"]:
        promoted = True
        trainer.save_checkpoints(
            detector_path="checkpoints/best_detector.pt",
            tracker_path="checkpoints/best_tracker.pt",
        )
        print(f"\n>>> CHECKPOINT POLICY: PROMOTED! New score ({macro_score:.4f}) >= baseline ({BASELINE_METRICS['macro_final_score']:.4f}).")
    else:
        print(f"\n>>> CHECKPOINT POLICY: RETAINED GOLDEN BASELINE. ({BASELINE_METRICS['macro_final_score']:.4f} > {macro_score:.4f}).")
    
    # Save full results JSON
    results_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "training_duration_sec": train_duration,
        "tracker_duration_sec": trk_duration,
        "total_duration_sec": train_duration + trk_duration,
        "epoch_logs": epoch_logs,
        "final_summary": val_summary,
        "promoted_to_best": promoted,
        "baseline_comparison": {
            "baseline": BASELINE_METRICS,
            "new_model": {
                "macro_final_score": macro_score,
                "macro_adjusted_jaccard": macro_j_adj,
                "macro_division_jaccard": macro_j_div,
                "macro_edge_jaccard": macro_edge_j,
                "total_node_tp": total_tp,
                "total_node_fp": total_fp,
                "total_node_fn": total_fn,
            },
        },
    }
    
    res_path = Path("reports/chunk1_domain_balanced_results.json")
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
        WHERE chunk_id = 'chunk_001_train'
    """, (
        json.dumps(CHUNK1_STEMS),
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
    print("Updated registry/experiments.sqlite with latest run metrics.")
    
    # Transactional Cleanup of chunk_001_train
    print("\n[Cleanup] Cleaning up temporary training data in chunk_001_train...")
    initial_free = storage_mgr.get_disk_free_gb()
    shutil.rmtree(chunk_dir)
    final_free = storage_mgr.get_disk_free_gb()
    freed_gb = final_free - initial_free
    print(f"Cleaned up {chunk_dir}. Freed: {freed_gb:.2f} GB (Current free: {final_free:.2f} GB).")


if __name__ == "__main__":
    main()

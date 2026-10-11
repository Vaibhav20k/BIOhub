#!/usr/bin/env python3
"""Evaluate Peak Checkpoint (Epoch 3) with full HeldOutEvaluator pipeline."""

import json
import logging
import time
import sys
from pathlib import Path
import torch

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.zarr_reader import open_dataset
from src.detection.unet3d import TemporalUNet3D
from src.representation.node_embedding import CellNodeEmbedding
from src.tracking.transformer import SpatioTemporalTracker
from src.pipeline.held_out_evaluator import HeldOutEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

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

def evaluate_detector_with_tracker(det_path: Path, trk_path: Path, label: str, device: torch.device):
    logger.info(f"\n{'='*70}\nEVALUATING: {label}\nDetector: {det_path}\nTracker: {trk_path}\n{'='*70}")
    
    # 1. Load detector
    det_ckpt = torch.load(det_path, map_location=device, weights_only=False)
    detector = TemporalUNet3D(
        in_channels=1,
        base_channels=16,
        feature_dim=32,
        prior_prob=0.05,
        use_temporal_attention=False,
    ).to(device)
    detector.load_state_dict(det_ckpt["model_state_dict"])
    detector.eval()
    
    # 2. Load tracker & embedder
    trk_ckpt = torch.load(trk_path, map_location=device, weights_only=False)
    embedder = CellNodeEmbedding(in_visual_dim=32, embedding_dim=128).to(device)
    if "embedder_state_dict" in trk_ckpt:
        embedder.load_state_dict(trk_ckpt["embedder_state_dict"])
    embedder.eval()
    
    tracker = SpatioTemporalTracker(
        node_dim=128,
        rel_dim=32,
        hidden_dim=trk_ckpt.get("config", {}).get("hidden_dim", 128),
        num_layers=trk_ckpt.get("config", {}).get("num_layers", 2),
    ).to(device)
    tracker.load_state_dict(trk_ckpt["tracker_state_dict"])
    tracker.eval()
    
    # 3. Open validation sequences
    val_datasets = [
        open_dataset(Path("data/val/44b6_66f9292d.zarr"), load_tracks=True),
        open_dataset(Path("data/val/6bba_d82a4fc6.zarr"), load_tracks=True),
    ]
    
    # 4. Run HeldOutEvaluator
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
    
    t0 = time.time()
    summary = evaluator.evaluate_cohort(val_datasets)
    elapsed = time.time() - t0
    
    macro_score = summary["macro_final_score"]
    seq_44b6 = summary["sequences"]["44b6_66f9292d"]
    seq_6bba = summary["sequences"]["6bba_d82a4fc6"]
    
    delta_macro = macro_score - PROMOTED_PHASE3_BASELINE["macro_final_score"]
    delta_44b6 = seq_44b6["final_score"] - PROMOTED_PHASE3_BASELINE["44b6_final_score"]
    delta_6bba = seq_6bba["final_score"] - PROMOTED_PHASE3_BASELINE["6bba_final_score"]
    delta_fp = summary["total_node_fp"] - PROMOTED_PHASE3_BASELINE["total_node_fp"]
    
    print("\n" + "=" * 85)
    print(f"                 RESULTS FOR: {label}")
    print("=" * 85)
    print(f"{'Metric':<25} | {'Phase 3 Promoted':<20} | {label:<20} | {'Delta':<12}")
    print("-" * 85)
    print(f"{'Macro Final Score':<25} | {PROMOTED_PHASE3_BASELINE['macro_final_score']:<20.4f} | {macro_score:<20.4f} | {delta_macro:<+12.4f}")
    print(f"{'44b6 Final Score':<25} | {PROMOTED_PHASE3_BASELINE['44b6_final_score']:<20.4f} | {seq_44b6['final_score']:<20.4f} | {delta_44b6:<+12.4f}")
    print(f"{'6bba Final Score':<25} | {PROMOTED_PHASE3_BASELINE['6bba_final_score']:<20.4f} | {seq_6bba['final_score']:<20.4f} | {delta_6bba:<+12.4f}")
    print(f"{'Macro J_adj':<25} | {PROMOTED_PHASE3_BASELINE['macro_final_score']:<20.4f} | {summary['macro_adjusted_jaccard']:<20.4f} | {summary['macro_adjusted_jaccard'] - PROMOTED_PHASE3_BASELINE['macro_final_score']:<+12.4f}")
    print(f"{'Macro Edge Jaccard':<25} | {PROMOTED_PHASE3_BASELINE['macro_edge_jaccard']:<20.4f} | {summary['macro_edge_jaccard']:<20.4f} | {summary['macro_edge_jaccard'] - PROMOTED_PHASE3_BASELINE['macro_edge_jaccard']:<+12.4f}")
    print(f"{'Total Node TP':<25} | {PROMOTED_PHASE3_BASELINE['total_node_tp']:<20} | {summary['total_node_tp']:<20} | {summary['total_node_tp'] - PROMOTED_PHASE3_BASELINE['total_node_tp']:<+12}")
    print(f"{'Total Node FP':<25} | {PROMOTED_PHASE3_BASELINE['total_node_fp']:<20} | {summary['total_node_fp']:<20} | {delta_fp:<+12}")
    print(f"{'Total Node FN':<25} | {PROMOTED_PHASE3_BASELINE['total_node_fn']:<20} | {summary['total_node_fn']:<20} | {summary['total_node_fn'] - PROMOTED_PHASE3_BASELINE['total_node_fn']:<+12}")
    str_44b6 = f"{seq_44b6['node_tp']}/{seq_44b6['node_fp']}/{seq_44b6['node_fn']} (rec={seq_44b6['node_recall']:.1%})"
    str_6bba = f"{seq_6bba['node_tp']}/{seq_6bba['node_fp']}/{seq_6bba['node_fn']} (rec={seq_6bba['node_recall']:.1%})"
    print(f"{'44b6 TP / FP / FN':<25} | {'0 / 144 / 745':<20} | {str_44b6:<20} |")
    print(f"{'6bba TP / FP / FN':<25} | {'118 / 1336 / 509':<20} | {str_6bba:<20} |")
    print(f"Total Evaluation Elapsed: {elapsed:.1f}s")
    print("=" * 85)
    
    return summary

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    det_e3 = Path("checkpoints/chunk2_epochs/epoch_03_detector.pt")
    trk_base = Path("checkpoints/best_tracker.pt")
    trk_c5 = Path("checkpoints/chunk2_epochs/tracker_continual_5ep.pt")
    
    res1 = evaluate_detector_with_tracker(det_e3, trk_base, "Epoch 3 Det + Baseline Tracker", device)
    
    # Save results
    out_path = Path("reports/epoch_03_evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump({"epoch_03_baseline_tracker": res1}, f, indent=2)
    print(f"\nSaved evaluation results to {out_path}")

if __name__ == "__main__":
    main()

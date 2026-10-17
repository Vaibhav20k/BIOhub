#!/usr/bin/env python3
"""Controlled inference ablation matrix and detection quality diagnostics.

Memory-safe, high-speed cached evaluation across 6 combinations:
1. EXP-1: Det_C2_base + Trk_C2_base (Reference reproduction)
2. EXP-2: Det_C3_ep01 + Trk_C2_base (Isolate early detector change)
3. EXP-3: Det_C3_ep01 + Trk_C3_cont (Reproduce reported Chunk 3 peak)
4. EXP-4: Det_C3_ep10 + Trk_C2_base (Isolate later detector degradation)
5. EXP-5: Det_C3_ep10 + Trk_C3_cont (Evaluate final epoch joint config)
6. EXP-6: Det_C2_base + Trk_C3_cont (Isolate tracker contribution!)
"""

import csv
import gc
import json
import logging
import os
from pathlib import Path
import sys
import time

import numpy as np
import polars as pl
import torch

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.data.zarr_reader import DatasetVolume, open_dataset
from src.data.preprocessing import normalize_intensity
from src.detection.unet3d import TemporalUNet3D
from src.detection.peak_detector import PeakDetector3D
from src.evaluation.tracking_metrics import CompetitionEvaluator
from src.graph.candidate_graph import CandidateGraphBuilder
from src.optimization.ilp_solver import LineageILPSolver
from src.representation.node_embedding import CellNodeEmbedding, extract_node_embeddings_for_dataset
from src.tracking.transformer import SpatioTemporalTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

out_dir = repo_root / "reports/chunk3_failure_analysis"
out_dir.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Validation Anchors
val_dir = repo_root / "data/val"
anchor_stems = ["44b6_66f9292d", "6bba_d82a4fc6"]
val_datasets = [open_dataset(val_dir / stem, load_tracks=True) for stem in anchor_stems]
print(f"Validation anchors: {[ds.name for ds in val_datasets]}")

# Detector definitions
detector_definitions = [
    ("Det_C2_Base", repo_root / "checkpoints/best_detector.pt", "Chunk 2 Baseline Detector"),
    ("Det_C3_Ep01", repo_root / "checkpoints/chunk3_epochs/epoch_01_detector.pt", "Chunk 3 Epoch 1 Detector"),
    ("Det_C3_Ep10", repo_root / "checkpoints/chunk3_epochs/epoch_10_detector.pt", "Chunk 3 Epoch 10 Detector"),
]

# Tracker definitions
tracker_definitions = [
    ("Trk_C2_Base", repo_root / "checkpoints/best_tracker.pt", "Chunk 2 Baseline Tracker"),
    ("Trk_C3_Cont", repo_root / "checkpoints/chunk3_epochs/tracker_continual_5ep.pt", "Chunk 3 Continual Tracker"),
]

# Step 1: Detect and cache nodes & embeddings for each detector
# We also compute the Section 6.1 heatmap statistics here
print("\n" + "=" * 80)
print("PHASE 1: DETECTOR INFERENCE & HEATMAP CALIBRATION AUDIT")
print("=" * 80)

heatmap_diagnostics = {}
# detector_cache[det_key][ds.name] = (nodes_df, node_embs_c2, node_embs_c3)
# Note: embeddings depend on both detector and embedder.
# We will compute visual features (feature_map) from detector, then embedder projects to 128-d.
# Let's extract detector feature maps and peak locations.

detection_cache = {}

for det_key, det_path, det_label in detector_definitions:
    print(f"\n--- Running Detector: {det_label} ---")
    ckpt = torch.load(det_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    detector = TemporalUNet3D(
        in_channels=1,
        base_channels=16,
        feature_dim=32,
        prior_prob=0.05,
        use_temporal_attention=False,
    ).to(device)
    detector.load_state_dict(state)
    detector.eval()
    
    detection_cache[det_key] = {}
    heatmap_diagnostics[det_label] = {}
    
    peak_detector = PeakDetector3D(threshold=0.25, max_peaks_per_frame=150)
    rng = np.random.RandomState(42)
    
    for ds in val_datasets:
        gt_nodes = ds.tracks.node_attrs()
        active_t = sorted(gt_nodes["t"].unique().to_list())
        
        detected_records = []
        global_node_id = 0
        gt_vals = []
        bg_vals = []
        
        # Store frame feature volumes on CPU to save GPU memory
        frame_features_cpu = {}
        
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
                    pred_hm, feat_map = detector(inp)
                
                hm_3d = pred_hm[0, 0].cpu().numpy().astype(np.float32)
                # feat_map is (1, 32, Z, Y, X)
                frame_features_cpu[t] = feat_map.cpu()
                del inp, pred_hm, feat_map
                
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
                
                peaks, scores = peak_detector.extract_peaks_3d(
                    hm_3d,
                    threshold=0.25,
                    max_peaks=150,
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
        
        nodes_df = pl.DataFrame(detected_records) if detected_records else pl.DataFrame()
        detection_cache[det_key][ds.name] = {
            "nodes_df": nodes_df,
            "frame_features": frame_features_cpu,
        }
        
        # Heatmap diagnostics
        gt_arr = np.array(gt_vals, dtype=np.float32)
        bg_arr = np.array(bg_vals, dtype=np.float32)
        
        diag = {
            "num_gt_centers": len(gt_arr),
            "num_bg_samples": len(bg_arr),
            "num_detected_peaks": len(detected_records),
            "gt_mean": float(np.mean(gt_arr)),
            "gt_median": float(np.median(gt_arr)),
            "gt_p90": float(np.percentile(gt_arr, 90)),
            "gt_p95": float(np.percentile(gt_arr, 95)),
            "gt_p99": float(np.percentile(gt_arr, 99)),
            "gt_fraction_below_025": float(np.mean(gt_arr < 0.25)),
            "bg_mean": float(np.mean(bg_arr)),
            "bg_median": float(np.median(bg_arr)),
            "bg_p90": float(np.percentile(bg_arr, 90)),
            "bg_p95": float(np.percentile(bg_arr, 95)),
            "bg_p99": float(np.percentile(bg_arr, 99)),
            "bg_p999": float(np.percentile(bg_arr, 99.9)),
            "bg_fraction_above_025": float(np.mean(bg_arr >= 0.25)),
            "contrast_ratio": float(np.mean(gt_arr) / max(np.percentile(bg_arr, 90), 1e-6)),
            "separation": float(np.mean(gt_arr) - np.percentile(bg_arr, 90)),
        }
        heatmap_diagnostics[det_label][ds.name] = diag
        print(f"  [{ds.name}] Detected: {len(detected_records)} peaks | GT Mean: {diag['gt_mean']:.4f} | BG P90: {diag['bg_p90']:.4f} | GT < 0.25: {diag['gt_fraction_below_025']:.1%}")
    
    del detector
    torch.cuda.empty_cache()
    gc.collect()

with open(out_dir / "domain_diagnostics.json", "w") as f:
    json.dump(heatmap_diagnostics, f, indent=2)
print("\nSaved heatmap quality audit to domain_diagnostics.json.")

# Step 2: Evaluate Controlled Inference Matrix (Section 5.1)
print("\n" + "=" * 80)
print("PHASE 2: CONTROLLED INFERENCE MATRIX (6 CONFIGURATIONS)")
print("=" * 80)

matrix_experiments = [
    {
        "exp_id": "EXP-1",
        "name": "Reference Chunk 2 Baseline",
        "det_key": "Det_C2_Base",
        "det_label": "Chunk 2 Baseline Det",
        "trk_key": "Trk_C2_Base",
        "trk_path": ckpt_c2_trk,
        "trk_label": "Chunk 2 Baseline Trk",
        "purpose": "Reproduce reference baseline score (0.0387)",
    },
    {
        "exp_id": "EXP-2",
        "name": "Early Detector Ablation",
        "det_key": "Det_C3_Ep01",
        "det_label": "Chunk 3 Epoch 1 Det",
        "trk_key": "Trk_C2_Base",
        "trk_path": ckpt_c2_trk,
        "trk_label": "Chunk 2 Baseline Trk",
        "purpose": "Isolate early detector change holding tracker constant",
    },
    {
        "exp_id": "EXP-3",
        "name": "Early Joint Configuration",
        "det_key": "Det_C3_Ep01",
        "det_label": "Chunk 3 Epoch 1 Det",
        "trk_key": "Trk_C3_Cont",
        "trk_path": ckpt_c3_trk,
        "trk_label": "Chunk 3 Continual Trk",
        "purpose": "Reproduce reported Chunk 3 peak configuration (0.0083)",
    },
    {
        "exp_id": "EXP-4",
        "name": "Late Detector Ablation",
        "det_key": "Det_C3_Ep10",
        "det_label": "Chunk 3 Epoch 10 Det",
        "trk_key": "Trk_C2_Base",
        "trk_path": ckpt_c2_trk,
        "trk_label": "Chunk 2 Baseline Trk",
        "purpose": "Isolate later detector degradation holding tracker constant",
    },
    {
        "exp_id": "EXP-5",
        "name": "Late Joint Configuration",
        "det_key": "Det_C3_Ep10",
        "det_label": "Chunk 3 Epoch 10 Det",
        "trk_key": "Trk_C3_Cont",
        "trk_path": ckpt_c3_trk,
        "trk_label": "Chunk 3 Continual Trk",
        "purpose": "Evaluate final epoch joint configuration (0.0006)",
    },
    {
        "exp_id": "EXP-6",
        "name": "Tracker-Only Ablation",
        "det_key": "Det_C2_Base",
        "det_label": "Chunk 2 Baseline Det",
        "trk_key": "Trk_C3_Cont",
        "trk_path": ckpt_c3_trk,
        "trk_label": "Chunk 3 Continual Trk",
        "purpose": "Isolate tracker contribution holding detector constant",
    },
]

comp_evaluator = CompetitionEvaluator(max_distance_um=7.0)
ilp_solver = LineageILPSolver()
builder = CandidateGraphBuilder(max_distance_um=7.0, allow_frame_skips=False)

ablation_results = []

for exp in matrix_experiments:
    print(f"\n[{exp['exp_id']}] Running: {exp['name']} ({exp['det_label']} + {exp['trk_label']})...")
    
    # Load tracker and embedder
    trk_ckpt = torch.load(exp["trk_path"], map_location=device, weights_only=False)
    embedder = CellNodeEmbedding(in_visual_dim=32, embedding_dim=128).to(device)
    if "embedder_state_dict" in trk_ckpt:
        embedder.load_state_dict(trk_ckpt["embedder_state_dict"])
    embedder.eval()
    
    tracker = SpatioTemporalTracker(
        node_dim=128,
        rel_dim=32,
        hidden_dim=128,
        num_layers=2,
    ).to(device)
    tracker.load_state_dict(trk_ckpt["tracker_state_dict"])
    tracker.eval()
    
    seq_metrics = {}
    
    for ds in val_datasets:
        cached_data = detection_cache[exp["det_key"]][ds.name]
        raw_nodes_df = cached_data["nodes_df"]
        frame_feats = cached_data["frame_features"]
        
        if raw_nodes_df.height == 0:
            m = comp_evaluator.evaluate(
                pred_nodes_df=pl.DataFrame(),
                pred_edges=[],
                gt_tracks=ds.tracks,
                scale=ds.scale,
            )
            seq_metrics[ds.name] = m
            continue
        
        # Extract embeddings for nodes using this embedder
        sorted_nodes = raw_nodes_df.sort(["t", "node_id"])
        node_embs_list = []
        
        with torch.no_grad():
            for row in sorted_nodes.iter_rows(named=True):
                t = row["t"]
                z = int(np.clip(round(row["z"]), 0, ds.shape[1] - 1))
                y = int(np.clip(round(row["y"]), 0, ds.shape[2] - 1))
                x = int(np.clip(round(row["x"]), 0, ds.shape[3] - 1))
                
                feat_t = frame_feats[t] # (1, 32, Z, Y, X)
                vis_feat = feat_t[0, :, z, y, x].unsqueeze(0).to(device) # (1, 32)
                
                norm_z = (z / max(ds.shape[1] - 1, 1)) * 2.0 - 1.0
                norm_y = (y / max(ds.shape[2] - 1, 1)) * 2.0 - 1.0
                norm_x = (x / max(ds.shape[3] - 1, 1)) * 2.0 - 1.0
                norm_t = (t / max(ds.shape[0] - 1, 1)) * 2.0 - 1.0
                norm_score = float(row.get("score", 0.5)) * 2.0 - 1.0
                
                pos_spatial = torch.tensor([[norm_z, norm_y, norm_x]], device=device, dtype=torch.float32)
                pos_temporal = torch.tensor([[norm_t]], device=device, dtype=torch.float32)
                score_t = torch.tensor([[norm_score]], device=device, dtype=torch.float32)
                
                emb = embedder(vis_feat, pos_spatial, pos_temporal, score_t)
                node_embs_list.append(emb[0].cpu())
                
        node_embs = torch.stack(node_embs_list) # (N, 128)
        
        # Build Candidate Graph
        builder_ds = CandidateGraphBuilder(max_distance_um=7.0, scale=ds.scale, allow_frame_skips=False)
        graph = builder_ds.build_candidate_graph(
            nodes_df=sorted_nodes,
            tracker=tracker,
            node_embeddings=node_embs,
            device=device,
        )
        
        # Solve ILP
        sol = ilp_solver.solve(graph)
        
        # Extract solved nodes and edges
        solved_nodes_df = sorted_nodes.filter(pl.col("node_id").is_in(sol.selected_node_ids))
        sel_edges_set = set(sol.selected_edge_ids)
        solved_edges = [
            {"source_id": edge.source_id, "target_id": edge.target_id}
            for eid, edge in graph.edges.items()
            if eid in sel_edges_set
        ]
        
        metrics = comp_evaluator.evaluate(
            pred_nodes_df=solved_nodes_df,
            pred_edges=solved_edges,
            gt_tracks=ds.tracks,
            scale=ds.scale,
        )
        seq_metrics[ds.name] = metrics
        print(f"    ✓ {ds.name}: Final={metrics.final_score:.4f}, J_adj={metrics.j_adj:.4f}, Edge_J={metrics.edge_jaccard:.4f}, TP={metrics.node_tp}, FP={metrics.node_fp}, FN={metrics.node_fn}")
    
    m44 = seq_metrics["44b6_66f9292d"]
    m6b = seq_metrics["6bba_d82a4fc6"]
    
    macro_score = (m44.final_score + m6b.final_score) / 2.0
    macro_j_adj = (m44.j_adj + m6b.j_adj) / 2.0
    macro_edge_j = (m44.edge_jaccard + m6b.edge_jaccard) / 2.0
    macro_div_j = (m44.j_div + m6b.j_div) / 2.0
    total_tp = m44.node_tp + m6b.node_tp
    total_fp = m44.node_fp + m6b.node_fp
    total_fn = m44.node_fn + m6b.node_fn
    macro_prec = (m44.node_precision + m6b.node_precision) / 2.0
    macro_rec = (m44.node_recall + m6b.node_recall) / 2.0
    
    row = {
        "exp_id": exp["exp_id"],
        "name": exp["name"],
        "detector": exp["det_label"],
        "tracker": exp["trk_label"],
        "purpose": exp["purpose"],
        "macro_final_score": f"{macro_score:.6f}",
        "6bba_final_score": f"{m6b.final_score:.6f}",
        "44b6_final_score": f"{m44.final_score:.6f}",
        "macro_edge_jaccard": f"{macro_edge_j:.6f}",
        "6bba_edge_jaccard": f"{m6b.edge_jaccard:.6f}",
        "44b6_edge_jaccard": f"{m44.edge_jaccard:.6f}",
        "macro_adjusted_node_jaccard": f"{macro_j_adj:.6f}",
        "total_node_tp": total_tp,
        "total_node_fp": total_fp,
        "total_node_fn": total_fn,
        "6bba_node_recall": f"{m6b.node_recall:.4f}",
        "6bba_node_precision": f"{m6b.node_precision:.4f}",
        "44b6_node_recall": f"{m44.node_recall:.4f}",
        "44b6_node_precision": f"{m44.node_precision:.4f}",
    }
    ablation_results.append(row)
    print(f"  ==> [{exp['exp_id']}] Macro Final Score: {macro_score:.4f} (6bba={m6b.final_score:.4f}, 44b6={m44.final_score:.4f})")
    
    del tracker, embedder
    torch.cuda.empty_cache()
    gc.collect()

csv_ablation = out_dir / "detector_tracker_ablation.csv"
with open(csv_ablation, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(ablation_results[0].keys()))
    writer.writeheader()
    writer.writerows(ablation_results)

print(f"\nWrote full ablation matrix to {csv_ablation}")

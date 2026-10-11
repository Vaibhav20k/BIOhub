#!/usr/bin/env python3
"""Biohub Promotion Verification Gate: Rigorous side-by-side verification of Phase 3 baseline vs Config B candidate."""

import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import polars as pl
import torch

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.zarr_reader import DatasetVolume, open_dataset
from src.data.preprocessing import normalize_intensity
from src.detection.unet3d import TemporalUNet3D
from src.detection.peak_detector import PeakDetector3D
from src.evaluation.node_matching import BipartiteNodeEvaluator
from src.evaluation.tracking_metrics import CompetitionEvaluator
from src.representation.node_embedding import CellNodeEmbedding, extract_node_embeddings_for_dataset
from src.tracking.transformer import SpatioTemporalTracker
from src.graph.candidate_graph import CandidateGraphBuilder
from src.optimization.ilp_solver import LineageILPSolver
from src.pipeline.held_out_evaluator import HeldOutEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ANCHOR_STEMS = ["44b6_66f9292d", "6bba_d82a4fc6"]

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def compute_heatmap_and_detection_metrics(
    detector: TemporalUNet3D,
    dataset: DatasetVolume,
    device: torch.device,
    detection_threshold: float = 0.25,
    max_peaks_per_frame: int = 150,
) -> Tuple[Dict[str, float], pl.DataFrame, Dict[str, float]]:
    """Compute detailed heatmap statistics and raw peak bipartite metrics across all frames."""
    detector.eval()
    peak_detector = PeakDetector3D(threshold=detection_threshold, max_peaks_per_frame=max_peaks_per_frame)
    gt_nodes = dataset.tracks.node_attrs()
    active_t = sorted(gt_nodes["t"].unique().to_list())
    
    gt_cell_vals = []
    bg_vals = []
    detected_records = []
    global_node_id = 0
    rng = np.random.RandomState(42)
    
    with torch.no_grad():
        for t in active_t:
            vol_crop = dataset.read_spatial_crop(
                slice(t, t + 1),
                slice(None),
                slice(None),
                slice(None),
            )
            vol_norm = normalize_intensity(vol_crop, quantiles=dataset.quantiles)
            inp = torch.from_numpy(vol_norm).unsqueeze(0).to(device)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                pred_hm, _ = detector(inp)
            hm_3d = pred_hm[0, 0].cpu().numpy().astype(np.float32)
            del inp, pred_hm
            
            # Ground truth cell center activations
            frame_gt = gt_nodes.filter(pl.col("t") == t)
            for row in frame_gt.iter_rows(named=True):
                z = int(np.clip(round(row["z"]), 0, hm_3d.shape[0] - 1))
                y = int(np.clip(round(row["y"]), 0, hm_3d.shape[1] - 1))
                x = int(np.clip(round(row["x"]), 0, hm_3d.shape[2] - 1))
                gt_cell_vals.append(float(hm_3d[z, y, x]))
                
            # Random background samples (1000 per frame)
            flat_hm = hm_3d.ravel()
            bg_idx = rng.choice(flat_hm.shape[0], size=1000, replace=False)
            bg_vals.extend(flat_hm[bg_idx].tolist())
            
            peaks, scores = peak_detector.extract_peaks_3d(hm_3d, threshold=detection_threshold, max_peaks=max_peaks_per_frame)
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
                
    gt_arr = np.array(gt_cell_vals, dtype=np.float32)
    bg_arr = np.array(bg_vals, dtype=np.float32)
    
    hm_stats = {
        "gt_count": len(gt_arr),
        "gt_mean": float(np.mean(gt_arr)) if len(gt_arr) else 0.0,
        "gt_median": float(np.median(gt_arr)) if len(gt_arr) else 0.0,
        "gt_p90": float(np.percentile(gt_arr, 90)) if len(gt_arr) else 0.0,
        "gt_min": float(np.min(gt_arr)) if len(gt_arr) else 0.0,
        "gt_max": float(np.max(gt_arr)) if len(gt_arr) else 0.0,
        "bg_sample_count": len(bg_arr),
        "bg_p50": float(np.percentile(bg_arr, 50)) if len(bg_arr) else 0.0,
        "bg_p90": float(np.percentile(bg_arr, 90)) if len(bg_arr) else 0.0,
        "bg_p99": float(np.percentile(bg_arr, 99)) if len(bg_arr) else 0.0,
        "bg_max": float(np.max(bg_arr)) if len(bg_arr) else 0.0,
        "contrast_ratio": float(np.mean(gt_arr) / max(float(np.percentile(bg_arr, 90)), 1e-6)) if len(gt_arr) else 0.0,
    }
    
    # Bipartite matching on raw peaks
    bipartite = BipartiteNodeEvaluator(max_distance=7.0, scale=dataset.scale)
    if len(detected_records) > 0:
        pred_df = pl.DataFrame(detected_records)
        b_res = bipartite.evaluate(pred_df, gt_nodes)
        raw_det = {
            "num_preds": len(pred_df),
            "tp": int(b_res.tp),
            "fp": int(b_res.fp),
            "fn": int(b_res.fn),
            "precision": float(b_res.precision),
            "recall": float(b_res.recall),
            "jaccard": float(b_res.tp / max(1, b_res.tp + b_res.fp + b_res.fn)),
            "pred_to_gt_ratio": float((b_res.tp + b_res.fp) / max(1, b_res.num_gt_nodes)),
            "j_adj": float(max(0.0, (b_res.tp / max(1, b_res.tp + b_res.fp + b_res.fn)) * (1.0 - 0.1 * ((b_res.tp + b_res.fp) / max(1, b_res.num_gt_nodes))))),
        }
    else:
        pred_df = pl.DataFrame(schema={"t": pl.Int64, "node_id": pl.Int64, "z": pl.Float32, "y": pl.Float32, "x": pl.Float32, "score": pl.Float32})
        raw_det = {
            "num_preds": 0, "tp": 0, "fp": 0, "fn": len(gt_nodes),
            "precision": 0.0, "recall": 0.0, "jaccard": 0.0, "pred_to_gt_ratio": 0.0, "j_adj": 0.0,
        }
        
    return hm_stats, pred_df, raw_det

def run_full_lineage_pipeline(
    detector: TemporalUNet3D,
    embedder: CellNodeEmbedding,
    tracker: SpatioTemporalTracker,
    dataset: DatasetVolume,
    detected_nodes_df: pl.DataFrame,
    device: torch.device,
    max_distance_um: float = 7.0,
) -> Dict[str, Any]:
    """Execute tracker, graph assembly, ILP solving, and official CompetitionEvaluator."""
    t0 = time.time()
    gt_nodes = dataset.tracks.node_attrs()
    gt_edges = dataset.tracks.edge_attrs()
    
    if len(detected_nodes_df) == 0:
        evaluator = CompetitionEvaluator(max_distance_um=max_distance_um)
        empty_metrics = evaluator.evaluate_lineages(
            pred_nodes=pl.DataFrame(schema={"node_id": pl.Int64, "t": pl.Int64, "z": pl.Float64, "y": pl.Float64, "x": pl.Float64}),
            pred_edges=pl.DataFrame(schema={"source_id": pl.Int64, "target_id": pl.Int64}),
            gt_nodes=gt_nodes,
            gt_edges=gt_edges,
        )
        return empty_metrics.to_dict()
        
    # 1. Feature embedding
    nodes_with_emb = extract_node_embeddings_for_dataset(
        dataset=dataset,
        nodes_df=detected_nodes_df,
        detector_model=detector,
        embedder=embedder,
        device=device,
        window_size=1,
    )
    
    # 2. Candidate graph assembly
    graph_builder = CandidateGraphBuilder(max_distance_um=max_distance_um, allow_skips=False, scale=dataset.scale)
    cand_graph = graph_builder.build_candidate_graph(nodes_with_emb, detector_model=detector)
    
    # 3. Transformer tracking predictions
    tracker.eval()
    cand_graph_scored = tracker.score_candidate_graph(cand_graph, device=device)
    
    # 4. Global SCIP ILP Solving
    ilp_solver = LineageILPSolver()
    t_ilp_start = time.time()
    ilp_solution = ilp_solver.solve(cand_graph_scored, dataset_name=dataset.name)
    ilp_time = time.time() - t_ilp_start
    
    # Extract solved nodes and edges
    selected_node_ids = set(ilp_solution.selected_nodes)
    solved_nodes_df = detected_nodes_df.filter(pl.col("node_id").is_in(list(selected_node_ids)))
    
    selected_edges = set(ilp_solution.selected_edges)
    rx_graph = cand_graph_scored.graph
    solved_edges_list = []
    for edge_tuple in selected_edges:
        if isinstance(edge_tuple, tuple) and len(edge_tuple) == 2:
            u, v = edge_tuple
            src_nid = int(rx_graph.get_node_data(u)["node_id"])
            tgt_nid = int(rx_graph.get_node_data(v)["node_id"])
            solved_edges_list.append({"source_id": src_nid, "target_id": tgt_nid})
            
    solved_edges_df = pl.DataFrame(solved_edges_list) if solved_edges_list else pl.DataFrame(schema={"source_id": pl.Int64, "target_id": pl.Int64})
    
    # 5. Score against official competition metrics
    evaluator = CompetitionEvaluator(max_distance_um=max_distance_um)
    metrics = evaluator.evaluate_lineages(
        pred_nodes=solved_nodes_df,
        pred_edges=solved_edges_df,
        gt_nodes=gt_nodes,
        gt_edges=gt_edges,
    )
    
    m_dict = metrics.to_dict()
    m_dict["total_runtime_sec"] = time.time() - t0
    m_dict["ilp_runtime_sec"] = ilp_time
    m_dict["solved_nodes_count"] = len(solved_nodes_df)
    m_dict["solved_edges_count"] = len(solved_edges_df)
    return m_dict

def evaluate_full_model(
    det_path: Path,
    trk_path: Path,
    model_name: str,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate a complete detector + tracker model checkpoint on both permanent validation anchors."""
    logger.info(f"\n=======================================================")
    logger.info(f"STARTING FULL EVALUATION FOR: {model_name}")
    logger.info(f"Detector: {det_path} (SHA: {file_sha256(det_path)[:16]}...)")
    logger.info(f"Tracker:  {trk_path} (SHA: {file_sha256(trk_path)[:16]}...)")
    logger.info(f"=======================================================")
    
    det_ckpt = torch.load(det_path, map_location=device, weights_only=False)
    detector = TemporalUNet3D(in_channels=1, base_channels=16, feature_dim=32, prior_prob=0.05, use_temporal_attention=False).to(device)
    detector.load_state_dict(det_ckpt["model_state_dict"])
    detector.eval()
    
    trk_ckpt = torch.load(trk_path, map_location=device, weights_only=False)
    embedder = CellNodeEmbedding(in_visual_dim=32, embedding_dim=128).to(device)
    if "embedder_state_dict" in trk_ckpt:
        embedder.load_state_dict(trk_ckpt["embedder_state_dict"])
    embedder.eval()
    
    tracker = SpatioTemporalTracker(
        node_dim=128, rel_dim=32,
        hidden_dim=trk_ckpt.get("config", {}).get("hidden_dim", 128),
        num_layers=trk_ckpt.get("config", {}).get("num_layers", 2),
    ).to(device)
    tracker.load_state_dict(trk_ckpt["tracker_state_dict"])
    tracker.eval()
    
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
    
    sequences_result = {}
    
    for stem in ANCHOR_STEMS:
        logger.info(f"\n--- Evaluating Sequence: {stem} ---")
        ds = open_dataset(Path(f"data/val/{stem}.zarr"), load_tracks=True)
        
        # 1. Heatmap stats and raw peak extraction
        logger.info(f"[{stem}] Computing full-volume heatmap stats and peak bipartite metrics...")
        hm_stats, raw_peaks_df, raw_det_metrics = compute_heatmap_and_detection_metrics(
            detector=detector,
            dataset=ds,
            device=device,
            detection_threshold=0.25,
            max_peaks_per_frame=150,
        )
        torch.cuda.empty_cache()
        logger.info(f"[{stem}] Raw Peaks: {raw_det_metrics['num_preds']} | TP: {raw_det_metrics['tp']} | FP: {raw_det_metrics['fp']} | Recall: {raw_det_metrics['recall']:.1%} | J_adj: {raw_det_metrics['j_adj']:.4f}")
        logger.info(f"[{stem}] Heatmap: GT mean={hm_stats['gt_mean']:.4f}, BG p90={hm_stats['bg_p90']:.4f}, Contrast={hm_stats['contrast_ratio']:.2f}")
        
        # 2. Official competition lineage pipeline (HeldOutEvaluator -> SCIP ILP -> CompetitionEvaluator)
        logger.info(f"[{stem}] Running official competition lineage pipeline...")
        m, t_infer, t_ilp = evaluator.evaluate_single_sequence(ds, return_timings=True)
        lineage_metrics = m.to_dict()
        lineage_metrics["inference_time_sec"] = t_infer
        lineage_metrics["ilp_runtime_sec"] = t_ilp
        torch.cuda.empty_cache()
        
        logger.info(f"[{stem}] Lineage Final Score: {lineage_metrics['final_score']:.4f} (J_adj={lineage_metrics['j_adj']:.4f}, Edge_J={lineage_metrics['edge_jaccard']:.4f}, Div_J={lineage_metrics['j_div']:.4f})")
        logger.info(f"[{stem}] Lineage Nodes: TP={lineage_metrics['node_tp']}, FP={lineage_metrics['node_fp']}, FN={lineage_metrics['node_fn']}, Recall={lineage_metrics['node_recall']:.1%}, Prec={lineage_metrics['node_precision']:.1%}")
        
        sequences_result[stem] = {
            "heatmap_stats": hm_stats,
            "raw_detection_metrics": raw_det_metrics,
            "lineage_metrics": lineage_metrics,
        }
        
    s44 = sequences_result["44b6_66f9292d"]["lineage_metrics"]
    s6b = sequences_result["6bba_d82a4fc6"]["lineage_metrics"]
    macro_score = (s44["final_score"] + s6b["final_score"]) / 2.0
    macro_j_adj = (s44["j_adj"] + s6b["j_adj"]) / 2.0
    macro_edge_j = (s44["edge_jaccard"] + s6b["edge_jaccard"]) / 2.0
    macro_div_j = (s44["j_div"] + s6b["j_div"]) / 2.0
    
    total_tp = s44["node_tp"] + s6b["node_tp"]
    total_fp = s44["node_fp"] + s6b["node_fp"]
    total_fn = s44["node_fn"] + s6b["node_fn"]
    
    summary = {
        "model_name": model_name,
        "detector_checkpoint": str(det_path),
        "detector_sha256": file_sha256(det_path),
        "tracker_checkpoint": str(trk_path),
        "tracker_sha256": file_sha256(trk_path),
        "macro_final_score": macro_score,
        "macro_adjusted_jaccard": macro_j_adj,
        "macro_edge_jaccard": macro_edge_j,
        "macro_division_jaccard": macro_div_j,
        "total_node_tp": total_tp,
        "total_node_fp": total_fp,
        "total_node_fn": total_fn,
        "sequences": sequences_result,
    }
    return summary

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    # 1. Checkpoint Provenance
    p3_det = Path("checkpoints/best_detector.pt")
    p3_trk = Path("checkpoints/best_tracker.pt")
    cand_det = Path("checkpoints/chunk2_epochs/epoch_03_detector.pt")
    cand_trk = Path("checkpoints/chunk2_epochs/tracker_continual_5ep.pt")
    
    assert p3_det.exists(), f"Phase 3 detector missing: {p3_det}"
    assert p3_trk.exists(), f"Phase 3 tracker missing: {p3_trk}"
    assert cand_det.exists(), f"Candidate detector missing: {cand_det}"
    assert cand_trk.exists(), f"Candidate tracker missing: {cand_trk}"
    
    # 2. Evaluate Preserved Phase 3 Baseline
    baseline_summary = evaluate_full_model(
        det_path=p3_det,
        trk_path=p3_trk,
        model_name="Preserved Phase 3 Baseline",
        device=device,
    )
    
    # 3. Evaluate Config B Candidate
    candidate_summary = evaluate_full_model(
        det_path=cand_det,
        trk_path=cand_trk,
        model_name="Candidate Config B (Epoch 3 Det + Continual Tracker)",
        device=device,
    )
    
    # 4. Compare and generate verification artifact
    verification_payload = {
        "verification_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": {
            "baseline": {
                "detector_path": str(p3_det),
                "detector_sha256": baseline_summary["detector_sha256"],
                "tracker_path": str(p3_trk),
                "tracker_sha256": baseline_summary["tracker_sha256"],
            },
            "candidate_config_b": {
                "detector_path": str(cand_det),
                "detector_sha256": candidate_summary["detector_sha256"],
                "tracker_path": str(cand_trk),
                "tracker_sha256": candidate_summary["tracker_sha256"],
            },
        },
        "baseline_evaluation": baseline_summary,
        "candidate_evaluation": candidate_summary,
        "macro_comparison": {
            "baseline_macro_final_score": baseline_summary["macro_final_score"],
            "candidate_macro_final_score": candidate_summary["macro_final_score"],
            "delta_macro_final_score": candidate_summary["macro_final_score"] - baseline_summary["macro_final_score"],
            "relative_improvement_pct": ((candidate_summary["macro_final_score"] - baseline_summary["macro_final_score"]) / max(1e-6, baseline_summary["macro_final_score"])) * 100.0,
        },
    }
    
    out_file = Path("reports/promotion_gate_verification.json")
    with open(out_file, "w") as f:
        json.dump(verification_payload, f, indent=2)
    logger.info(f"\nVerification artifact written to: {out_file}")

if __name__ == "__main__":
    main()

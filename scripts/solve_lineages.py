#!/usr/bin/env python3
"""End-to-end lineage solver CLI: Detection -> Embeddings -> Tracker -> Candidate Graph -> ILP -> Competition Evaluation."""

import argparse
import json
import logging
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl
import torch

from src.data.zarr_reader import open_dataset
from src.detection.peak_detector import PeakDetector3D
from src.detection.unet3d import TemporalUNet3D
from src.evaluation.tracking_metrics import CompetitionEvaluator
from src.graph.candidate_graph import CandidateGraphBuilder, candidate_graph_to_dataframes
from src.optimization.ilp_solver import LineageILPSolver
from src.representation.node_embedding import CellNodeEmbedding, extract_node_embeddings_for_dataset
from src.tracking.transformer import SpatioTemporalTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end cell lineage reconstruction and evaluation.")
    parser.add_argument("--dataset", type=str, default="44b6_0b24845f", help="Dataset name or path")
    parser.add_argument("--detector-checkpoint", type=str, default="checkpoints/best_detector.pt", help="Path to detector checkpoint")
    parser.add_argument("--tracker-checkpoint", type=str, default="checkpoints/best_tracker.pt", help="Path to tracker checkpoint")
    parser.add_argument("--threshold", type=float, default=0.3, help="Detection threshold")
    parser.add_argument("--use-gt-nodes", action="store_true", help="Use ground-truth nodes instead of raw detections")
    parser.add_argument("--allow-skips", action="store_true", help="Allow 2-frame skips")
    parser.add_argument("--max-distance", type=float, default=7.0, help="Max distance cutoff (µm)")
    parser.add_argument("--weight-edge", type=float, default=1.0, help="Weight on edge costs")
    parser.add_argument("--weight-node", type=float, default=0.5, help="Weight on node costs")
    parser.add_argument("--weight-division", type=float, default=1.0, help="Weight on mitosis costs")
    parser.add_argument("--cost-appear", type=float, default=3.0, help="Track start penalty")
    parser.add_argument("--cost-disappear", type=float, default=3.0, help="Track end penalty")
    parser.add_argument("--output-dir", type=str, default="reports", help="Directory for reports and geff exports")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load dataset
    ds_path = Path(args.dataset)
    if not (ds_path.parent / f"{ds_path.name}.zarr").exists():
        if (Path("data/train") / f"{args.dataset}.zarr").exists():
            ds_path = Path("data/train") / args.dataset
        elif (Path("data/fixtures") / f"{args.dataset}.zarr").exists():
            ds_path = Path("data/fixtures") / args.dataset

    dataset = open_dataset(ds_path, load_tracks=True, require_tracks=True)
    logger.info(f"Loaded dataset {dataset.name} | Shape: {dataset.shape} | GT Nodes: {dataset.tracks.num_nodes()} | GT Edges: {dataset.tracks.num_edges()}")

    # 1. Detections
    ckpt_det = torch.load(args.detector_checkpoint, map_location=device)
    model = TemporalUNet3D(
        in_channels=1,
        base_channels=ckpt_det.get("config", {}).get("base_channels", 16),
        feature_dim=ckpt_det.get("config", {}).get("feature_dim", 32),
        use_temporal_attention=True,
    ).to(device)
    model.load_state_dict(ckpt_det["model_state_dict"])
    model.eval()

    if args.use_gt_nodes:
        logger.info("Using ground-truth nodes for candidate tracking graph.")
        nodes_df = dataset.tracks.node_attrs().with_columns(pl.lit(1.0).alias("score"))
    else:
        logger.info(f"Running peak detection at threshold {args.threshold}...")
        detector = PeakDetector3D(threshold=args.threshold, scale=dataset.scale)
        active_t = sorted(dataset.tracks.node_attrs()["t"].unique().to_list())
        nodes_df = detector.detect_dataset_nodes(model=model, dataset=dataset, device=device, timepoints=active_t)

    # 2. Node Embeddings
    ckpt_trk = torch.load(args.tracker_checkpoint, map_location=device)
    embedder = CellNodeEmbedding(
        in_visual_dim=32,
        embedding_dim=128,
        scale=dataset.scale,
    ).to(device)
    if "embedder_state_dict" in ckpt_trk:
        embedder.load_state_dict(ckpt_trk["embedder_state_dict"])
    embedder.eval()

    tracker = SpatioTemporalTracker(
        node_dim=128,
        rel_dim=32,
        hidden_dim=ckpt_trk.get("config", {}).get("hidden_dim", 128),
        num_layers=ckpt_trk.get("config", {}).get("num_layers", 2),
        scale=dataset.scale,
    ).to(device)
    tracker.load_state_dict(ckpt_trk["tracker_state_dict"])
    tracker.eval()

    logger.info("Extracting node embeddings...")
    sorted_nodes_df, node_embeddings = extract_node_embeddings_for_dataset(
        model=model,
        embedder=embedder,
        dataset=dataset,
        nodes_df=nodes_df,
        device=device,
    )

    # 3. Candidate Graph
    logger.info("Building candidate tracking graph...")
    graph_builder = CandidateGraphBuilder(
        max_distance_um=args.max_distance,
        scale=dataset.scale,
        allow_frame_skips=args.allow_skips,
    )
    graph = graph_builder.build_candidate_graph(
        nodes_df=sorted_nodes_df,
        tracker=tracker,
        node_embeddings=node_embeddings,
        device=device,
    )
    logger.info(f"Candidate graph assembled: {graph.num_nodes} nodes, {graph.num_edges} edges.")

    # 4. Global ILP Solver
    logger.info("Solving global Integer Linear Program (SCIP)...")
    solver = LineageILPSolver(
        weight_edge=args.weight_edge,
        weight_node=args.weight_node,
        weight_division=args.weight_division,
        cost_appear=args.cost_appear,
        cost_disappear=args.cost_disappear,
    )
    solution = solver.solve(graph)
    logger.info(
        f"SCIP solve finished in {solution.solve_time_sec:.2f}s | "
        f"Status: {solution.status} | Objective: {solution.objective_value:.2f} | "
        f"Active Edges: {len(solution.selected_edge_ids)} | Active Nodes: {len(solution.selected_node_ids)} | "
        f"Divisions: {len(solution.dividing_node_ids)}"
    )

    # 5. Export Reconstructed Lineage
    solved_tracks = solution.to_tracksdata(graph)

    # 6. Evaluate Against Ground Truth
    logger.info("Computing official competition tracking metrics (J_adj + 0.1 * J_div)...")
    evaluator = CompetitionEvaluator(max_distance_um=args.max_distance, scale=dataset.scale)
    pred_nodes_df = solved_tracks.node_attrs()
    pred_edges_df = solved_tracks.edge_attrs()
    gt_nodes_df = dataset.tracks.node_attrs()
    gt_edges_df = dataset.tracks.edge_attrs()

    metrics = evaluator.evaluate_lineages(
        pred_nodes=pred_nodes_df,
        pred_edges=pred_edges_df,
        gt_nodes=gt_nodes_df,
        gt_edges=gt_edges_df,
    )

    print("\n==================================================")
    print(f"Official Competition Evaluation — {dataset.name}")
    print("==================================================")
    print(f"Ground Truth Nodes       : {gt_nodes_df.height}")
    print(f"Predicted Active Nodes   : {pred_nodes_df.height}")
    print(f"Node Jaccard (J)         : {metrics.node_jaccard:.4f}")
    print(f"Pred/GT Node Ratio       : {metrics.pred_to_gt_ratio:.4f}")
    print(f"Adjusted Jaccard (J_adj) : {metrics.j_adj:.4f}")
    print("--------------------------------------------------")
    print(f"Ground Truth Divisions   : {metrics.gt_divisions}")
    print(f"Predicted Divisions      : {metrics.pred_divisions}")
    print(f"Division Jaccard (J_div) : {metrics.j_div:.4f}")
    print("==================================================")
    print(f"FINAL COMPETITION SCORE  : {metrics.final_score:.4f}  (J_adj + 0.1 * J_div)")
    print("==================================================")
    print(f"Edge Precision           : {metrics.edge_precision*100:.1f}%")
    print(f"Edge Recall              : {metrics.edge_recall*100:.1f}%")
    print(f"Edge F1 Score            : {metrics.edge_f1:.4f}")
    print("==================================================")

    # Save outputs
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / f"lineage_eval_{dataset.name}.json"
    with open(report_file, "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    logger.info(f"Saved evaluation report to {report_file}")

    geff_out = out_dir / f"solved_lineages_{dataset.name}.geff"
    try:
        solved_tracks.to_geff(str(geff_out), overwrite=True)
    except TypeError:
        import shutil
        if geff_out.exists():
            shutil.rmtree(geff_out)
        solved_tracks.to_geff(str(geff_out))
    logger.info(f"Saved reconstructed lineages to {geff_out}")


if __name__ == "__main__":
    main()

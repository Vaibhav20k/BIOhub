#!/usr/bin/env python3
"""CLI utility to construct and inspect the Spatio-Temporal Candidate Graph for a dataset."""

import argparse
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
from src.graph.candidate_graph import CandidateGraphBuilder, candidate_graph_to_dataframes
from src.graph.graph_export import validate_dag_temporal_monotonicity
from src.representation.node_embedding import CellNodeEmbedding, extract_node_embeddings_for_dataset
from src.tracking.transformer import SpatioTemporalTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Spatio-Temporal Candidate Graph.")
    parser.add_argument("--dataset", type=str, default="44b6_0b24845f", help="Dataset name or path")
    parser.add_argument("--detector-checkpoint", type=str, default="checkpoints/best_detector.pt", help="Path to detector checkpoint")
    parser.add_argument("--tracker-checkpoint", type=str, default="checkpoints/best_tracker.pt", help="Path to tracker checkpoint")
    parser.add_argument("--threshold", type=float, default=0.3, help="Detection threshold")
    parser.add_argument("--use-gt-nodes", action="store_true", help="Use ground-truth nodes instead of running detector")
    parser.add_argument("--allow-skips", action="store_true", help="Allow 2-frame skips")
    parser.add_argument("--max-distance", type=float, default=7.0, help="Max distance cutoff (µm)")
    parser.add_argument("--output-dir", type=str, default="reports", help="Directory to save graph exports")

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
    logger.info(f"Loaded dataset {dataset.name} | Shape: {dataset.shape} | GT Nodes: {dataset.tracks.num_nodes()}")

    # Load detector
    ckpt_det = torch.load(args.detector_checkpoint, map_location=device)
    model = TemporalUNet3D(
        in_channels=1,
        base_channels=ckpt_det.get("config", {}).get("base_channels", 16),
        feature_dim=ckpt_det.get("config", {}).get("feature_dim", 32),
        use_temporal_attention=True,
    ).to(device)
    model.load_state_dict(ckpt_det["model_state_dict"])
    model.eval()

    # Get nodes
    if args.use_gt_nodes:
        logger.info("Using ground-truth nodes for candidate graph assembly.")
        nodes_df = dataset.tracks.node_attrs().with_columns(pl.lit(1.0).alias("score"))
    else:
        logger.info(f"Running 3D sub-pixel peak detection at threshold {args.threshold}...")
        detector = PeakDetector3D(threshold=args.threshold, scale=dataset.scale)
        active_t = sorted(dataset.tracks.node_attrs()["t"].unique().to_list())
        nodes_df = detector.detect_dataset_nodes(model=model, dataset=dataset, device=device, timepoints=active_t)
        logger.info(f"Detected {nodes_df.height} candidate cell centroids across {len(active_t)} timepoints.")

    # Load Tracker
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

    # Extract node embeddings
    logger.info("Extracting dense 128D node embeddings...")
    sorted_nodes_df, node_embeddings = extract_node_embeddings_for_dataset(
        model=model,
        embedder=embedder,
        dataset=dataset,
        nodes_df=nodes_df,
        device=device,
    )

    # Build Candidate Graph
    logger.info("Building Spatio-Temporal Candidate Graph...")
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

    summary = graph.get_summary()
    print("\n==========================================")
    print(f"Candidate Graph Summary — {dataset.name}")
    print("==========================================")
    for k, v in summary.items():
        print(f"{k:<25}: {v}")
    print("==========================================")

    # Verify DAG validity
    is_monotonic = validate_dag_temporal_monotonicity(graph)
    print(f"Strict Temporal Monotonicity (DAG) : {'PASSED [t_dst > t_src]' if is_monotonic else 'FAILED'}")

    # Export to DataFrames
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_nodes, df_edges = candidate_graph_to_dataframes(graph)
    nodes_csv = out_dir / f"candidate_nodes_{dataset.name}.csv"
    edges_csv = out_dir / f"candidate_edges_{dataset.name}.csv"
    df_nodes.write_csv(nodes_csv)
    df_edges.write_csv(edges_csv)
    logger.info(f"Saved candidate nodes to {nodes_csv} ({df_nodes.height} rows)")
    logger.info(f"Saved candidate edges to {edges_csv} ({df_edges.height} rows)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CLI tool to evaluate trained 3D U-Net cell detection models against ground-truth annotations."""

import argparse
import json
import logging
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
import torch

from src.data.zarr_reader import open_dataset
from src.detection.peak_detector import PeakDetector3D
from src.detection.unet3d import TemporalUNet3D
from src.evaluation.node_matching import BipartiteNodeEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 3D U-Net cell detector via Hungarian bipartite matching.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_detector.pt", help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default="44b6_0b24845f", help="Dataset name or path to evaluate")
    parser.add_argument("--threshold", type=float, default=0.3, help="Default detection confidence threshold")
    parser.add_argument("--max-distance", type=float, default=7.0, help="Maximum matching distance cutoff (µm)")
    parser.add_argument("--reports-dir", type=str, default="reports", help="Directory to save evaluation reports")
    parser.add_argument("--sweep", action="store_true", help="Perform threshold sweep across multiple cutoffs")

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

    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found at {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt.get("config", {})
    base_channels = config.get("base_channels", 16)
    feature_dim = config.get("feature_dim", 32)

    model = TemporalUNet3D(
        in_channels=1,
        base_channels=base_channels,
        feature_dim=feature_dim,
        use_temporal_attention=True,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"Loaded detector weights from epoch {ckpt.get('epoch', 'N/A')} (loss: {ckpt.get('best_loss', 'N/A')})")

    # Evaluate
    evaluator = BipartiteNodeEvaluator(max_distance=args.max_distance, scale=dataset.scale)
    gt_nodes = dataset.tracks.node_attrs()
    active_timepoints = sorted(gt_nodes["t"].unique().to_list())
    logger.info(f"Evaluating across {len(active_timepoints)} annotated timepoints: [{min(active_timepoints)} ... {max(active_timepoints)}]")

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        logger.info("Running threshold sweep across [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]...")
        sweep_df = evaluator.threshold_sweep(
            model=model,
            dataset=dataset,
            device=device,
            thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            timepoints=active_timepoints,
        )
        print("\nThreshold Sweep Results:")
        print(sweep_df)

        # Plot threshold sweep curves
        plt.figure(figsize=(9, 5))
        plt.plot(sweep_df["threshold"].to_list(), sweep_df["precision"].to_list(), label="Precision", marker="o", color="#1f77b4")
        plt.plot(sweep_df["threshold"].to_list(), sweep_df["recall"].to_list(), label="Recall", marker="s", color="#2ca02c")
        plt.plot(sweep_df["threshold"].to_list(), sweep_df["f1"].to_list(), label="F1 Score", marker="^", color="#d62728", lw=2)
        plt.plot(sweep_df["threshold"].to_list(), sweep_df["node_ratio"].to_list(), label="Node Ratio (Pred/GT)", linestyle="--", color="#ff7f0e")
        plt.axhline(1.0, color="gray", linestyle=":", alpha=0.6)
        plt.title(f"Cell Detection Performance vs Confidence Threshold ({dataset.name})")
        plt.xlabel("Confidence Threshold (tau)")
        plt.ylabel("Metric Value")
        plt.ylim(0, 1.2)
        plt.grid(True, alpha=0.3)
        plt.legend()
        sweep_plot = reports_dir / f"detection_sweep_{dataset.name}.png"
        plt.savefig(sweep_plot, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved threshold sweep plot to {sweep_plot}")

    # Single threshold evaluation
    detector = PeakDetector3D(threshold=args.threshold, scale=dataset.scale)
    pred_df = detector.detect_dataset_nodes(
        model=model,
        dataset=dataset,
        device=device,
        timepoints=active_timepoints,
    )

    result = evaluator.evaluate(pred_df, gt_nodes, timepoints=active_timepoints)

    print("\n==========================================")
    print(f"Detection Evaluation Summary — {dataset.name} (Threshold: {args.threshold})")
    print("==========================================")
    print(f"Ground Truth Nodes : {result.num_gt_nodes}")
    print(f"Predicted Nodes    : {result.num_pred_nodes}")
    print(f"True Positives     : {result.tp}")
    print(f"False Positives    : {result.fp}")
    print(f"False Negatives    : {result.fn}")
    print(f"Node Precision     : {result.precision:.4f}")
    print(f"Node Recall        : {result.recall:.4f}")
    print(f"Node F1 Score      : {result.f1:.4f}")
    print(f"Total Node Ratio   : {result.total_node_ratio:.4f}")
    print(f"Mean Error (um)    : {result.mean_error_um:.3f} µm")
    print(f"Median Error (um)  : {result.median_error_um:.3f} µm")
    print(f"P90 Error (um)     : {result.p90_error_um:.3f} µm")
    print("==========================================")

    # Save JSON report
    report_file = reports_dir / f"detection_eval_{dataset.name}.json"
    with open(report_file, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info(f"Saved evaluation metrics to {report_file}")


if __name__ == "__main__":
    main()

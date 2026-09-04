#!/usr/bin/env python3
"""CLI utility to generate and validate official Kaggle Biohub submission.csv."""

import argparse
import logging
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.submission.pipeline import EndToEndSubmissionPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Kaggle Biohub submission.csv file.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["44b6_0b24845f"],
        help="List of dataset names or paths to process for submission",
    )
    parser.add_argument(
        "--detector-checkpoint",
        type=str,
        default="checkpoints/best_detector.pt",
        help="Path to trained 3D Temporal U-Net detector",
    )
    parser.add_argument(
        "--tracker-checkpoint",
        type=str,
        default="checkpoints/best_tracker.pt",
        help="Path to trained Spatio-Temporal Transformer tracker",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Peak detection confidence threshold",
    )
    parser.add_argument(
        "--max-peaks-per-frame",
        type=int,
        default=50,
        help="Maximum candidate peaks to retain per frame (default: 50)",
    )
    parser.add_argument(
        "--time-range",
        nargs=2,
        type=int,
        default=None,
        metavar=("START", "END"),
        help="Optional explicit time range [START, END) to evaluate",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=7.0,
        help="Maximum matching cutoff in micrometers",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submission.csv",
        help="Path to save submission.csv",
    )

    args = parser.parse_args()

    # Resolve dataset paths
    resolved_paths = []
    for d in args.datasets:
        p = Path(d)
        if not (p.parent / f"{p.name}.zarr").exists() and not p.is_dir():
            if (Path("data/train") / f"{d}.zarr").exists():
                p = Path("data/train") / d
            elif (Path("data/fixtures") / f"{d}.zarr").exists():
                p = Path("data/fixtures") / d
        resolved_paths.append(p)

    logger.info(f"Generating submission for {len(resolved_paths)} datasets: {[p.name for p in resolved_paths]}")

    t_list = None
    if args.time_range:
        t_list = list(range(args.time_range[0], args.time_range[1]))
        logger.info(f"Restricting evaluation to explicit time range: {args.time_range[0]}..{args.time_range[1]}")

    pipeline = EndToEndSubmissionPipeline(
        detector_checkpoint=args.detector_checkpoint,
        tracker_checkpoint=args.tracker_checkpoint,
        detection_threshold=args.threshold,
        max_peaks_per_frame=args.max_peaks_per_frame,
        max_distance_um=args.max_distance,
    )

    sub_df, val_res = pipeline.generate_submission(
        datasets=resolved_paths,
        output_csv=args.output,
        timepoints=t_list,
    )

    print("\n==================================================")
    print("Kaggle Submission Generation & Validation Summary")
    print("==================================================")
    print(f"Total Rows       : {val_res.num_rows}")
    print(f"Total Datasets   : {val_res.num_datasets} ({val_res.datasets})")
    print(f"Total Nodes      : {val_res.num_nodes}")
    print(f"Total Edges      : {val_res.num_edges}")
    print(f"Validation Status: {'PASSED [COMPLIANT]' if val_res.is_valid else 'FAILED'}")
    if val_res.errors:
        print("Errors:")
        for err in val_res.errors:
            print(f"  - {err}")
    print(f"Saved Output To  : {args.output}")
    print("==================================================")

    if not val_res.is_valid:
        sys.exit(1)


if __name__ == "__main__":
    main()

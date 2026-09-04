#!/usr/bin/env python3
"""CLI tool to run comprehensive Exploratory Data Analysis (EDA) across datasets."""

import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.zarr_reader import open_dataset
from src.eda.dataset_stats import profile_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EDA profiling across datasets.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/train",
        help="Directory containing .zarr and .geff datasets (or specific dataset name).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="reports",
        help="Directory to save JSON profiles and EDA distribution figures.",
    )

    args = parser.parse_args()
    data_path = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_profile = []
    if data_path.is_dir():
        # Find all .zarr directories
        for zarr_dir in sorted(data_path.glob("*.zarr")):
            stem = zarr_dir.stem
            datasets_to_profile.append(data_path / stem)
    elif (data_path.parent / f"{data_path.stem}.zarr").exists():
        datasets_to_profile.append(data_path.parent / data_path.stem)
    else:
        print(f"Error: No datasets found at {data_path}")
        sys.exit(1)

    print(f"Found {len(datasets_to_profile)} dataset(s) to profile:")
    for d in datasets_to_profile:
        print(f"  - {d.name}")

    for ds_path in datasets_to_profile:
        print(f"\n==========================================")
        print(f"Profiling: {ds_path.name}")
        print(f"==========================================")
        dataset = open_dataset(ds_path, load_tracks=True)
        profile = profile_dataset(dataset, out_dir=out_dir)

        print(f"Volume Shape: {profile.volume_shape}")
        print(f"Physical Scale: {profile.scale_um} µm/voxel")
        if profile.displacement:
            print(f"Displacement: Mean={profile.displacement.mean_um:.2f} µm | P95={profile.displacement.p95_um:.2f} µm | Max={profile.displacement.max_um:.2f} µm")
            print(f"Recommended Search Radius R_max: {profile.displacement.recommended_search_radius_um} µm")
        if profile.density:
            print(f"Density: {profile.density.total_nodes} nodes across {profile.density.num_timepoints_with_nodes} timepoints (Mean {profile.density.mean_nodes_per_timepoint:.1f}/tp)")
            print(f"Nearest Neighbor Dist: Mean={profile.density.mean_nearest_neighbor_dist_um:.2f} µm | P05={profile.density.p05_nearest_neighbor_dist_um:.2f} µm")
        if profile.mitosis and profile.mitosis.division_count > 0:
            print(f"Mitosis: {profile.mitosis.division_count} divisions (Rate {profile.mitosis.division_rate_per_frame:.3f}/frame)")
            print(f"Daughter Separation: Mean={profile.mitosis.mean_daughter_separation_um:.2f} µm")

    print("\nEDA Profiling complete! Reports written to:", out_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CLI tool to visualize OME-Zarr 4D datasets and GEFF tracks via Napari or headless export."""

import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.zarr_reader import open_dataset
from src.visualization.napari_viewer import (
    create_napari_viewer,
    export_mip_gallery,
    export_ortho_projections,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="View or export 4D microscopy datasets and tracks.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="44b6_0b24845f",
        help="Dataset name or path (e.g. '44b6_0b24845f' or 'data/fixtures/synthetic_clip')",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode: generate and save summary images instead of launching GUI.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="reports",
        help="Directory to save exported figures.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=4,
        help="Number of frames in MIP gallery.",
    )

    args = parser.parse_args()

    # Resolve dataset path
    ds_path = Path(args.dataset)
    if not ds_path.exists():
        # Check in data/train or data/fixtures
        train_candidate = Path("data/train") / args.dataset
        fix_candidate = Path("data/fixtures") / args.dataset
        if (Path("data/train") / f"{args.dataset}.zarr").exists():
            ds_path = train_candidate
        elif (Path("data/fixtures") / f"{args.dataset}.zarr").exists():
            ds_path = fix_candidate
        else:
            print(f"Error: Dataset {args.dataset} not found in current directory, data/train, or data/fixtures.")
            sys.exit(1)

    print(f"Opening dataset: {ds_path}")
    dataset = open_dataset(ds_path, load_tracks=True)
    print(f"Loaded {dataset.name} | Shape: {dataset.shape} | Scale: {dataset.scale} µm/vox")
    if dataset.tracks is not None:
        print(f"Tracks: {dataset.tracks.num_nodes()} nodes, {dataset.tracks.num_edges()} edges")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.headless:
        print(f"Exporting headless visualization figures to {out_dir}...")
        gallery_file = out_dir / f"gallery_{dataset.name}.png"
        export_mip_gallery(dataset, gallery_file, num_frames=args.num_frames)
        print(f"  -> Saved MIP gallery to {gallery_file}")

        # Choose a representative timepoint with nodes
        t_rep = 0
        if dataset.tracks is not None and dataset.tracks.num_nodes() > 0:
            nodes_df = dataset.tracks.node_attrs()
            t_rep = int(nodes_df["t"].median())

        ortho_file = out_dir / f"ortho_{dataset.name}.png"
        export_ortho_projections(dataset, t_rep, ortho_file)
        print(f"  -> Saved orthogonal projections to {ortho_file}")
        print("Done!")
    else:
        print("Launching interactive Napari viewer...")
        viewer = create_napari_viewer(dataset, show=True)
        if viewer is None:
            print("Notice: GUI could not be initialized. Defaulting to headless export.")
            export_mip_gallery(dataset, out_dir / f"gallery_{dataset.name}.png", num_frames=args.num_frames)
        else:
            import napari
            napari.run()


if __name__ == "__main__":
    main()

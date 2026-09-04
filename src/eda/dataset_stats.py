"""Statistical analysis and profiling of cell trajectories, velocities, and density."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
from scipy.spatial import KDTree

from src.data.zarr_reader import DatasetVolume, DEFAULT_SCALE


@dataclass
class DisplacementProfile:
    count: int
    mean_um: float
    std_um: float
    median_um: float
    p90_um: float
    p95_um: float
    p99_um: float
    max_um: float
    recommended_search_radius_um: float


@dataclass
class DensityProfile:
    total_nodes: int
    num_timepoints_with_nodes: int
    mean_nodes_per_timepoint: float
    max_nodes_per_timepoint: int
    mean_nearest_neighbor_dist_um: float
    median_nearest_neighbor_dist_um: float
    p05_nearest_neighbor_dist_um: float  # 5th percentile (cell crowding / minimum cell radius)


@dataclass
class MitosisProfile:
    division_count: int
    division_rate_per_frame: float
    mean_daughter_separation_um: float
    median_daughter_separation_um: float


@dataclass
class DatasetProfile:
    dataset_name: str
    volume_shape: Tuple[int, int, int, int]
    scale_um: Tuple[float, float, float]
    displacement: Optional[DisplacementProfile]
    density: Optional[DensityProfile]
    mitosis: Optional[MitosisProfile]
    quantiles: Dict[str, float]


def compute_displacement_profile(
    dataset: DatasetVolume,
) -> Optional[DisplacementProfile]:
    """Compute physical displacement distribution between connected cell nodes."""
    if dataset.tracks is None or dataset.tracks.num_edges() == 0:
        return None

    nodes_df = dataset.tracks.node_attrs()
    edges_df = dataset.tracks.edge_attrs()

    # Join edges with source node coordinates
    joined = (
        edges_df.join(
            nodes_df.select(
                pl.col("node_id").alias("source_id"),
                pl.col("t").alias("t_src"),
                pl.col("z").alias("z_src"),
                pl.col("y").alias("y_src"),
                pl.col("x").alias("x_src"),
            ),
            on="source_id",
        )
        .join(
            nodes_df.select(
                pl.col("node_id").alias("target_id"),
                pl.col("t").alias("t_tgt"),
                pl.col("z").alias("z_tgt"),
                pl.col("y").alias("y_tgt"),
                pl.col("x").alias("x_tgt"),
            ),
            on="target_id",
        )
    )

    if joined.height == 0:
        return None

    scale_z, scale_y, scale_x = dataset.scale
    dz = (joined["z_tgt"] - joined["z_src"]).to_numpy() * scale_z
    dy = (joined["y_tgt"] - joined["y_src"]).to_numpy() * scale_y
    dx = (joined["x_tgt"] - joined["x_src"]).to_numpy() * scale_x

    dists = np.sqrt(dz**2 + dy**2 + dx**2)

    p90 = float(np.percentile(dists, 90))
    p95 = float(np.percentile(dists, 95))
    p99 = float(np.percentile(dists, 99))
    max_d = float(np.max(dists))

    # Recommended candidate search radius R_max: 99th percentile + safety buffer
    # ensuring >= 99.5% of true links are candidates while avoiding dense false edge explosions
    r_max = max(p99 * 1.25, 7.0)

    return DisplacementProfile(
        count=len(dists),
        mean_um=float(np.mean(dists)),
        std_um=float(np.std(dists)),
        median_um=float(np.median(dists)),
        p90_um=p90,
        p95_um=p95,
        p99_um=p99,
        max_um=max_d,
        recommended_search_radius_um=round(r_max, 2),
    )


def compute_density_profile(
    dataset: DatasetVolume,
) -> Optional[DensityProfile]:
    """Compute cell density and nearest-neighbor distance (NND) distribution."""
    if dataset.tracks is None or dataset.tracks.num_nodes() == 0:
        return None

    nodes_df = dataset.tracks.node_attrs()
    scale = np.array(dataset.scale, dtype=np.float32)

    # Group by timepoint
    nodes_per_t = nodes_df.group_by("t").len()
    counts = nodes_per_t["len"].to_numpy()

    # Nearest neighbor distance per timepoint
    nnd_list: List[float] = []
    for (t,), group in nodes_df.group_by("t"):
        if group.height >= 2:
            coords = group.select(["z", "y", "x"]).to_numpy() * scale
            tree = KDTree(coords)
            # Query 2 nearest neighbors (first neighbor is self, dist 0)
            dists, _ = tree.query(coords, k=2)
            nnd_list.extend(dists[:, 1].tolist())

    if nnd_list:
        nnd_arr = np.array(nnd_list)
        mean_nnd = float(np.mean(nnd_arr))
        median_nnd = float(np.median(nnd_arr))
        p05_nnd = float(np.percentile(nnd_arr, 5))
    else:
        mean_nnd = median_nnd = p05_nnd = float("nan")

    return DensityProfile(
        total_nodes=nodes_df.height,
        num_timepoints_with_nodes=len(counts),
        mean_nodes_per_timepoint=float(np.mean(counts)),
        max_nodes_per_timepoint=int(np.max(counts)),
        mean_nearest_neighbor_dist_um=mean_nnd,
        median_nearest_neighbor_dist_um=median_nnd,
        p05_nearest_neighbor_dist_um=p05_nnd,
    )


def compute_mitosis_profile(
    dataset: DatasetVolume,
) -> Optional[MitosisProfile]:
    """Identify and profile mitotic division events."""
    if dataset.tracks is None or dataset.tracks.num_edges() == 0:
        return None

    nodes_df = dataset.tracks.node_attrs()
    edges_df = dataset.tracks.edge_attrs()

    # Find nodes with out-degree >= 2
    out_degrees = edges_df.group_by("source_id").len()
    div_sources = out_degrees.filter(pl.col("len") >= 2)["source_id"].to_list()

    daughter_separations: List[float] = []
    scale = np.array(dataset.scale, dtype=np.float32)

    for src_id in div_sources:
        children = edges_df.filter(pl.col("source_id") == src_id)["target_id"].to_list()
        if len(children) >= 2:
            c1_info = nodes_df.filter(pl.col("node_id") == children[0])
            c2_info = nodes_df.filter(pl.col("node_id") == children[1])
            if c1_info.height > 0 and c2_info.height > 0:
                p1 = c1_info.select(["z", "y", "x"]).to_numpy()[0] * scale
                p2 = c2_info.select(["z", "y", "x"]).to_numpy()[0] * scale
                d_sep = float(np.linalg.norm(p1 - p2))
                daughter_separations.append(d_sep)

    div_count = len(div_sources)
    T = dataset.shape[0]
    rate = div_count / max(1, T)

    if daughter_separations:
        mean_sep = float(np.mean(daughter_separations))
        median_sep = float(np.median(daughter_separations))
    else:
        mean_sep = median_sep = float("nan")

    return MitosisProfile(
        division_count=div_count,
        division_rate_per_frame=float(rate),
        mean_daughter_separation_um=mean_sep,
        median_daughter_separation_um=median_sep,
    )


def plot_eda_distributions(
    dataset: DatasetVolume,
    out_path: Union[str, Path],
) -> Optional[Path]:
    """Generate and save a 4-panel EDA diagnostic dashboard for a dataset.

    Panels:
        1. Cell displacement distribution with critical percentiles & R_max
        2. Nearest-neighbor distance (NND) distribution across time
        3. Annotated cell count timeline across timepoints
        4. Directional displacement components (dz, dy, dx) in physical units
    """
    if dataset.tracks is None or dataset.tracks.num_nodes() == 0:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nodes_df = dataset.tracks.node_attrs()
    edges_df = dataset.tracks.edge_attrs()
    scale_z, scale_y, scale_x = dataset.scale

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    # 1. Displacement distribution
    if edges_df.height > 0:
        joined = (
            edges_df.join(
                nodes_df.select(
                    pl.col("node_id").alias("source_id"),
                    pl.col("z").alias("z_src"),
                    pl.col("y").alias("y_src"),
                    pl.col("x").alias("x_src"),
                ),
                on="source_id",
            )
            .join(
                nodes_df.select(
                    pl.col("node_id").alias("target_id"),
                    pl.col("z").alias("z_tgt"),
                    pl.col("y").alias("y_tgt"),
                    pl.col("x").alias("x_tgt"),
                ),
                on="target_id",
            )
        )

        dz = (joined["z_tgt"] - joined["z_src"]).to_numpy() * scale_z
        dy = (joined["y_tgt"] - joined["y_src"]).to_numpy() * scale_y
        dx = (joined["x_tgt"] - joined["x_src"]).to_numpy() * scale_x
        dists = np.sqrt(dz**2 + dy**2 + dx**2)

        axes[0, 0].hist(dists, bins=25, color="#1f77b4", edgecolor="black", alpha=0.7, density=True)
        axes[0, 0].axvline(np.mean(dists), color="red", linestyle="--", label=f"Mean: {np.mean(dists):.2f} µm")
        axes[0, 0].axvline(np.percentile(dists, 95), color="orange", linestyle="--", label=f"P95: {np.percentile(dists, 95):.2f} µm")
        axes[0, 0].axvline(np.percentile(dists, 99), color="purple", linestyle="--", label=f"P99: {np.percentile(dists, 99):.2f} µm")
        axes[0, 0].axvline(max(np.percentile(dists, 99) * 1.25, 7.0), color="green", linestyle="-", lw=2, label=f"R_max cutoff: {max(np.percentile(dists, 99) * 1.25, 7.0):.2f} µm")
        axes[0, 0].set_title("Frame-to-Frame Cell Displacement (||Δr||)", fontweight="bold")
        axes[0, 0].set_xlabel("Physical Distance (µm)")
        axes[0, 0].set_ylabel("Probability Density")
        axes[0, 0].legend(fontsize=8)
    else:
        axes[0, 0].text(0.5, 0.5, "No temporal edges available", ha="center", va="center")

    # 2. Nearest Neighbor Distance
    scale = np.array(dataset.scale, dtype=np.float32)
    nnd_list = []
    for (t,), group in nodes_df.group_by("t"):
        if group.height >= 2:
            coords = group.select(["z", "y", "x"]).to_numpy() * scale
            tree = KDTree(coords)
            dists, _ = tree.query(coords, k=2)
            nnd_list.extend(dists[:, 1].tolist())

    if nnd_list:
        axes[0, 1].hist(nnd_list, bins=25, color="#ff7f0e", edgecolor="black", alpha=0.7, density=True)
        axes[0, 1].axvline(np.mean(nnd_list), color="red", linestyle="--", label=f"Mean: {np.mean(nnd_list):.2f} µm")
        axes[0, 1].axvline(np.percentile(nnd_list, 5), color="blue", linestyle=":", label=f"P05 (Crowding): {np.percentile(nnd_list, 5):.2f} µm")
        axes[0, 1].set_title("Nearest-Neighbor Distance (Spatial Crowding)", fontweight="bold")
        axes[0, 1].set_xlabel("Inter-cell Distance (µm)")
        axes[0, 1].set_ylabel("Probability Density")
        axes[0, 1].legend(fontsize=8)
    else:
        axes[0, 1].text(0.5, 0.5, "Isolated single cells per frame", ha="center", va="center")

    # 3. Cell Count Timeline
    nodes_per_t = nodes_df.group_by("t").len().sort("t")
    axes[1, 0].plot(nodes_per_t["t"].to_numpy(), nodes_per_t["len"].to_numpy(), marker="o", color="#2ca02c", lw=2)
    axes[1, 0].set_title("Annotated Cell Count vs Timepoint", fontweight="bold")
    axes[1, 0].set_xlabel("Timepoint (t)")
    axes[1, 0].set_ylabel("Annotated Cells")
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Directional Displacement Breakdown
    if edges_df.height > 0:
        axes[1, 1].boxplot([np.abs(dz), np.abs(dy), np.abs(dx)], tick_labels=["|Δz| (Axial)", "|Δy| (Lateral)", "|Δx| (Lateral)"])
        axes[1, 1].set_title("Physical Motion Anisotropy (|Δr| by axis)", fontweight="bold")
        axes[1, 1].set_ylabel("Displacement (µm)")
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, "No edge data", ha="center", va="center")

    fig.suptitle(f"EDA Distribution Profile — {dataset.name}", fontsize=14, fontweight="bold")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def profile_dataset(
    dataset: DatasetVolume,
    out_dir: Optional[Path] = None,
) -> DatasetProfile:
    """Generate comprehensive statistical profile of a dataset volume."""
    disp_prof = compute_displacement_profile(dataset)
    density_prof = compute_density_profile(dataset)
    mitosis_prof = compute_mitosis_profile(dataset)

    profile = DatasetProfile(
        dataset_name=dataset.name,
        volume_shape=dataset.shape,
        scale_um=dataset.scale,
        displacement=disp_prof,
        density=density_prof,
        mitosis=mitosis_prof,
        quantiles=dataset.quantiles,
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        report_file = out_dir / f"profile_{dataset.name}.json"
        report_file.write_text(json.dumps(asdict(profile), indent=2))
        print(f"Saved dataset profile report to {report_file}")

        plot_path = out_dir / f"eda_distributions_{dataset.name}.png"
        plot_eda_distributions(dataset, plot_path)
        print(f"Saved EDA distribution plot to {plot_path}")

    return profile

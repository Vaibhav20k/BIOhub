"""Interactive Napari 4D visualization and headless MIP rendering for cell tracking."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.data.zarr_reader import DatasetVolume

logger = logging.getLogger(__name__)


def geff_to_napari_tracks(
    dataset: DatasetVolume,
) -> Tuple[np.ndarray, Dict[int, List[int]], Dict[int, int]]:
    """Convert a GEFF lineage graph into Napari-compatible tracks array and graph dictionary.

    Napari Tracks layer expects:
        - data: (N, 5) ndarray with columns [track_id, t, z, y, x]
        - graph: dict mapping child_track_id -> [parent_track_ids]

    Args:
        dataset: DatasetVolume containing loaded tracks.

    Returns:
        tracks_data: np.ndarray of shape (N, 5)
        napari_graph: dict of {int: list of int}
        node_to_track: dict mapping original node_id to track_id
    """
    if dataset.tracks is None or dataset.tracks.num_nodes() == 0:
        return np.empty((0, 5), dtype=np.float32), {}, {}

    nodes_df = dataset.tracks.node_attrs()
    edges_df = dataset.tracks.edge_attrs()

    # Map node_id to coordinates (t, z, y, x)
    # Note: safely accessing columns by name to handle arbitrary column ordering
    nodes_dict: Dict[int, Tuple[float, float, float, float]] = {}
    for row in nodes_df.iter_rows(named=True):
        nid = int(row["node_id"])
        t = float(row["t"])
        z = float(row["z"])
        y = float(row["y"])
        x = float(row["x"])
        nodes_dict[nid] = (t, z, y, x)

    # Build adjacency lists for incoming and outgoing edges
    in_edges: Dict[int, List[int]] = {}
    out_edges: Dict[int, List[int]] = {}
    if edges_df.height > 0:
        for row in edges_df.iter_rows(named=True):
            s = int(row["source_id"])
            tgt = int(row["target_id"])
            out_edges.setdefault(s, []).append(tgt)
            in_edges.setdefault(tgt, []).append(s)

    # Identify segment starts:
    # A node starts a new track segment if:
    # 1. It has in-degree == 0 (root/birth)
    # 2. Its parent has out-degree > 1 (mitosis daughter branch)
    # 3. It has in-degree > 1 (merge)
    start_nodes: List[int] = []
    for nid in sorted(nodes_dict.keys(), key=lambda k: (nodes_dict[k][0], k)):
        parents = in_edges.get(nid, [])
        if len(parents) == 0 or len(parents) > 1:
            start_nodes.append(nid)
        else:
            parent_out = out_edges.get(parents[0], [])
            if len(parent_out) > 1:
                start_nodes.append(nid)

    visited_nodes = set()
    node_to_track: Dict[int, int] = {}
    tracks_list: List[List[float]] = []
    current_track_id = 1

    for s_nid in start_nodes:
        if s_nid in visited_nodes:
            continue
        tid = current_track_id
        current_track_id += 1

        curr: Optional[int] = s_nid
        while curr is not None and curr not in visited_nodes:
            visited_nodes.add(curr)
            node_to_track[curr] = tid
            t, z, y, x = nodes_dict[curr]
            tracks_list.append([float(tid), t, z, y, x])

            children = out_edges.get(curr, [])
            if len(children) == 1 and len(in_edges.get(children[0], [])) == 1:
                curr = children[0]
            else:
                break

    # Defensive fallback for any orphaned nodes not reached by above traversal
    for nid, (t, z, y, x) in nodes_dict.items():
        if nid not in visited_nodes:
            tid = current_track_id
            current_track_id += 1
            node_to_track[nid] = tid
            visited_nodes.add(nid)
            tracks_list.append([float(tid), t, z, y, x])

    # Build Napari lineage graph (child_track_id -> [parent_track_ids])
    napari_graph: Dict[int, List[int]] = {}
    for s_nid in start_nodes:
        tid = node_to_track[s_nid]
        parents = in_edges.get(s_nid, [])
        parent_tracks = [
            node_to_track[p]
            for p in parents
            if p in node_to_track and node_to_track[p] != tid
        ]
        if parent_tracks:
            napari_graph[tid] = sorted(list(set(parent_tracks)))

    # Sort tracks array by track_id, then time t (Napari requirement)
    tracks_arr = np.array(tracks_list, dtype=np.float32)
    if len(tracks_arr) > 0:
        sort_indices = np.lexsort((tracks_arr[:, 1], tracks_arr[:, 0]))
        tracks_arr = tracks_arr[sort_indices]

    return tracks_arr, napari_graph, node_to_track


def create_napari_viewer(
    dataset: DatasetVolume,
    show: bool = True,
    title: Optional[str] = None,
) -> Any:
    """Create an interactive Napari 4D viewer with volume and tracks layers.

    Args:
        dataset: DatasetVolume instance.
        show: Whether to open the Napari GUI window immediately.
        title: Viewer window title.

    Returns:
        napari.Viewer instance, or None if GUI initialization fails in headless mode.
    """
    try:
        import napari
    except ImportError:
        logger.error("Napari is not installed in the current environment.")
        return None

    try:
        viewer = napari.Viewer(title=title or f"Biohub - {dataset.name}", show=show)
    except Exception as e:
        logger.warning(
            f"Failed to initialize Napari Qt window (likely headless environment): {e}"
        )
        return None

    scale_4d = (1.0,) + tuple(dataset.scale)

    # Determine sensible contrast limits from volume quantiles if available
    clow = dataset.quantiles.get("q0.01", 0.0)
    chigh = dataset.quantiles.get("q0.999", 2000.0)
    if chigh <= clow:
        chigh = clow + 100.0

    # Add 4D lazy Zarr volume
    viewer.add_image(
        dataset.zarr_group["0"],
        name=f"Volume ({dataset.name})",
        scale=scale_4d,
        contrast_limits=[clow, chigh],
        colormap="gray",
        blending="translucent",
    )

    # Add Points & Tracks if GEFF graph is available
    if dataset.tracks is not None and dataset.tracks.num_nodes() > 0:
        nodes_df = dataset.tracks.node_attrs()
        coords_list = []
        node_ids = []
        for row in nodes_df.iter_rows(named=True):
            coords_list.append([row["t"], row["z"], row["y"], row["x"]])
            node_ids.append(row["node_id"])

        coords_arr = np.array(coords_list, dtype=np.float32)

        viewer.add_points(
            coords_arr,
            name="Cell Detections",
            scale=scale_4d,
            size=6,
            properties={"node_id": node_ids},
            face_color="cyan",
            border_color="white",
            border_width=0.2,
        )

        tracks_arr, napari_graph, _ = geff_to_napari_tracks(dataset)
        if len(tracks_arr) > 0:
            viewer.add_tracks(
                tracks_arr,
                name="Lineage Tracks",
                graph=napari_graph,
                scale=scale_4d,
                tail_length=15,
                tail_width=3.0,
                colormap="turbo",
            )

    return viewer


def render_mip_overlay(
    dataset: DatasetVolume,
    timepoint: int,
    proj_axis: str = "z",
) -> Tuple[np.ndarray, pl.DataFrame, List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
    """Compute 2D Maximum Intensity Projection and extract projected cell nodes & edges.

    Args:
        dataset: DatasetVolume instance.
        timepoint: Time index to project.
        proj_axis: Axis to project along ('z', 'y', or 'x').

    Returns:
        mip: 2D float32 image array.
        nodes_at_t: Polars DataFrame of nodes at timepoint with coordinates.
        active_edge_coords: List of 2D line segments [((u_col, u_row), (v_col, v_row))] for edges from t to t+1.
    """
    vol = dataset.read_spatial_crop(
        slice(timepoint, timepoint + 1),
        slice(None),
        slice(None),
        slice(None),
    )[0]  # Shape: (Z, Y, X)

    axis_map = {"z": (0, "x", "y"), "y": (1, "x", "z"), "x": (2, "y", "z")}
    if proj_axis not in axis_map:
        raise ValueError(f"proj_axis must be one of {list(axis_map.keys())}, got {proj_axis}")

    axis_idx, col_name, row_name = axis_map[proj_axis]
    mip = np.max(vol, axis=axis_idx)

    nodes_at_t = pl.DataFrame()
    active_edge_coords: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

    if dataset.tracks is not None and dataset.tracks.num_nodes() > 0:
        nodes_df = dataset.tracks.node_attrs()
        edges_df = dataset.tracks.edge_attrs()

        nodes_at_t = nodes_df.filter(pl.col("t") == timepoint)

        # Look for edges from t to t+1
        if edges_df.height > 0:
            joined = (
                edges_df.join(
                    nodes_df.select(
                        pl.col("node_id").alias("source_id"),
                        pl.col("t").alias("t_src"),
                        pl.col(col_name).alias("c_src"),
                        pl.col(row_name).alias("r_src"),
                    ),
                    on="source_id",
                )
                .join(
                    nodes_df.select(
                        pl.col("node_id").alias("target_id"),
                        pl.col("t").alias("t_tgt"),
                        pl.col(col_name).alias("c_tgt"),
                        pl.col(row_name).alias("r_tgt"),
                    ),
                    on="target_id",
                )
                .filter(pl.col("t_src") == timepoint)
            )

            for r in joined.iter_rows(named=True):
                active_edge_coords.append(
                    ((float(r["c_src"]), float(r["r_src"])), (float(r["c_tgt"]), float(r["r_tgt"])))
                )

    return mip, nodes_at_t, active_edge_coords


def export_mip_gallery(
    dataset: DatasetVolume,
    out_path: Union[str, Path],
    num_frames: int = 4,
    timepoints: Optional[List[int]] = None,
) -> Path:
    """Generate a multi-frame Maximum Intensity Projection gallery with ground-truth track overlays.

    Args:
        dataset: DatasetVolume to visualize.
        out_path: Output image filepath (.png).
        num_frames: Number of frames to display if timepoints not specified.
        timepoints: Specific time indices to plot.

    Returns:
        Path to the saved figure.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if timepoints is None:
        if dataset.tracks is not None and dataset.tracks.num_nodes() > 0:
            active_t = sorted(dataset.tracks.node_attrs()["t"].unique().to_list())
            if len(active_t) <= num_frames:
                timepoints = active_t
            else:
                indices = np.linspace(0, len(active_t) - 1, num_frames, dtype=int)
                timepoints = [active_t[i] for i in indices]
        else:
            timepoints = np.linspace(0, dataset.shape[0] - 1, num_frames, dtype=int).tolist()

    n = len(timepoints)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), constrained_layout=True)
    if n == 1:
        axes = [axes]

    clow = dataset.quantiles.get("q0.01", None)
    chigh = dataset.quantiles.get("q0.999", None)

    for ax, t in zip(axes, timepoints):
        mip, nodes_t, edge_coords = render_mip_overlay(dataset, t, proj_axis="z")
        vmin = clow if clow is not None else np.percentile(mip, 1)
        vmax = chigh if chigh is not None else np.percentile(mip, 99.5)

        ax.imshow(mip, cmap="magma", vmin=vmin, vmax=vmax, origin="upper")

        # Overlay cell centroids
        if nodes_t.height > 0:
            xs = nodes_t["x"].to_numpy()
            ys = nodes_t["y"].to_numpy()
            ax.scatter(
                xs,
                ys,
                s=50,
                c="cyan",
                edgecolors="white",
                linewidths=1.2,
                label=f"Cells (N={len(xs)})",
                zorder=3,
            )

        # Overlay displacement vectors to t+1
        for (c1, r1), (c2, r2) in edge_coords:
            ax.annotate(
                "",
                xy=(c2, r2),
                xytext=(c1, r1),
                arrowprops=dict(
                    arrowstyle="->",
                    color="lime",
                    lw=2.0,
                    mutation_scale=12,
                ),
                zorder=4,
            )

        ax.set_title(f"Frame t={t}", fontsize=11, fontweight="bold")
        ax.set_xlabel("X (voxels)", fontsize=9)
        ax.set_ylabel("Y (voxels)", fontsize=9)
        if nodes_t.height > 0:
            ax.legend(loc="lower right", fontsize=8, framealpha=0.6)

    fig.suptitle(
        f"Z-MIP Cell Lineage Tracking: {dataset.name} (Scale: {dataset.scale} µm/vox)",
        fontsize=13,
        fontweight="bold",
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved MIP gallery to {out_path}")
    return out_path


def export_ortho_projections(
    dataset: DatasetVolume,
    timepoint: int,
    out_path: Union[str, Path],
) -> Path:
    """Export orthogonal 3-plane (XY, XZ, YZ) Maximum Intensity Projections for 3D inspection."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vol = dataset.read_spatial_crop(
        slice(timepoint, timepoint + 1),
        slice(None),
        slice(None),
        slice(None),
    )[0]  # (Z, Y, X)

    mip_xy = np.max(vol, axis=0)  # (Y, X)
    mip_xz = np.max(vol, axis=1)  # (Z, X)
    mip_yz = np.max(vol, axis=2)  # (Z, Y)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)

    aspect_z_yx = dataset.scale[0] / dataset.scale[1]  # 1.625 / 0.40625 = 4.0

    axes[0].imshow(mip_xy, cmap="viridis", origin="upper")
    axes[0].set_title(f"XY Plane (t={timepoint})")
    axes[0].set_xlabel("X (voxels)")
    axes[0].set_ylabel("Y (voxels)")

    axes[1].imshow(mip_xz, cmap="viridis", aspect=aspect_z_yx, origin="upper")
    axes[1].set_title(f"XZ Plane (Anisotropy {aspect_z_yx:.1f}x)")
    axes[1].set_xlabel("X (voxels)")
    axes[1].set_ylabel("Z (axial voxels)")

    axes[2].imshow(mip_yz, cmap="viridis", aspect=aspect_z_yx, origin="upper")
    axes[2].set_title(f"YZ Plane (Anisotropy {aspect_z_yx:.1f}x)")
    axes[2].set_xlabel("Y (voxels)")
    axes[2].set_ylabel("Z (axial voxels)")

    fig.suptitle(f"Orthogonal Projections - {dataset.name}", fontsize=12, fontweight="bold")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path

"""OME-Zarr and GEFF lazy dataset reader for 4D microscopy volumes."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import polars as pl
import tracksdata as td
import zarr

DEFAULT_SCALE: Tuple[float, float, float] = (1.625, 0.40625, 0.40625)


@dataclass
class DatasetVolume:
    """Represents a 4D microscopy dataset with paired graph tracks."""

    name: str
    zarr_path: Path
    zarr_group: Any
    shape: Tuple[int, int, int, int]  # (T, Z, Y, X)
    scale: Tuple[float, float, float]  # (scale_z, scale_y, scale_x) in micrometers
    quantiles: Dict[str, float] = field(default_factory=dict)
    tracks_path: Optional[Path] = None
    tracks: Optional[td.graph.BaseGraph] = None

    def read_timepoints(self, t_start: int, t_end: int) -> np.ndarray:
        """Read a slice of timepoints [t_start, t_end) into memory as float32."""
        arr = self.zarr_group["0"][t_start:t_end]
        return np.asarray(arr, dtype=np.float32)

    def read_spatial_crop(
        self,
        t_slice: slice,
        z_slice: slice,
        y_slice: slice,
        x_slice: slice,
    ) -> np.ndarray:
        """Read a 4D spatial-temporal bounding box into memory as float32."""
        arr = self.zarr_group["0"][t_slice, z_slice, y_slice, x_slice]
        return np.asarray(arr, dtype=np.float32)


def _parse_scale_from_attrs(attrs: Dict[str, Any]) -> Tuple[float, float, float]:
    """Extract (Z, Y, X) physical scale from OME-NGFF multiscales metadata."""
    multiscales = attrs.get("multiscales", [])
    if multiscales:
        first_ms = multiscales[0]
        datasets = first_ms.get("datasets", [])
        if datasets:
            for transform in datasets[0].get("coordinateTransformations", []):
                if transform.get("type") == "scale":
                    scale_vec = transform.get("scale", [])
                    # Shape is typically 4D [scale_t, scale_z, scale_y, scale_x]
                    if len(scale_vec) == 4:
                        return (float(scale_vec[1]), float(scale_vec[2]), float(scale_vec[3]))
                    elif len(scale_vec) == 3:
                        return (float(scale_vec[0]), float(scale_vec[1]), float(scale_vec[2]))
    return DEFAULT_SCALE


def open_dataset(
    path: Union[str, Path],
    require_tracks: bool = False,
    load_tracks: bool = True,
) -> DatasetVolume:
    """Open an OME-Zarr dataset and optionally its paired .geff tracks graph.

    Args:
        path: Path to dataset directory or file without extension (e.g. data/train/sample_1)
        require_tracks: If True, raises FileNotFoundError if .geff tracks are absent.
        load_tracks: If True and .geff exists, loads the graph into memory.

    Returns:
        DatasetVolume instance with lazy access to volume chunks.
    """
    path = Path(path)
    if path.suffix in (".zarr", ".geff"):
        stem = path.stem
        base_dir = path.parent
    else:
        stem = path.name
        base_dir = path.parent

    zarr_path = base_dir / f"{stem}.zarr"
    geff_path = base_dir / f"{stem}.geff"

    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr volume not found at {zarr_path}")

    # Open root group (supports both Zarr v2 and Zarr v3)
    z_group = zarr.open_group(str(zarr_path), mode="r")
    if "0" not in z_group:
        raise KeyError(f"Expected array '0' inside OME-Zarr group at {zarr_path}")

    raw_shape = tuple(int(s) for s in z_group["0"].shape)
    if len(raw_shape) != 4:
        raise ValueError(f"Expected 4D array (T, Z, Y, X), got shape {raw_shape}")

    attrs = dict(z_group.attrs)
    scale = _parse_scale_from_attrs(attrs)
    quantiles = attrs.get("image_statistics", {}).get("quantiles", {})

    tracks = None
    if geff_path.exists() and load_tracks:
        res = td.graph.IndexedRXGraph.from_geff(str(geff_path))
        tracks = res[0] if isinstance(res, tuple) else res
    elif require_tracks:
        raise FileNotFoundError(f"Required tracks file not found at {geff_path}")

    return DatasetVolume(
        name=stem,
        zarr_path=zarr_path,
        zarr_group=z_group,
        shape=raw_shape,
        scale=scale,
        quantiles=quantiles,
        tracks_path=geff_path if geff_path.exists() else None,
        tracks=tracks,
    )

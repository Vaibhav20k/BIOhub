"""3D sub-pixel peak detection and non-maximum suppression (NMS) for cell centroids."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
from scipy.ndimage import maximum_filter
import torch
import torch.nn as nn
from torch.amp import autocast

from src.data.preprocessing import normalize_intensity
from src.data.zarr_reader import DatasetVolume, DEFAULT_SCALE


@dataclass
class DetectedPeak:
    t: int
    node_id: int
    z: float
    y: float
    x: float
    score: float


class PeakDetector3D:
    """Extracts discrete cell centroids from continuous 3D heatmaps with sub-pixel precision."""

    def __init__(
        self,
        threshold: float = 0.3,
        min_distance: Tuple[int, int, int] = (3, 7, 7),  # (k_z, k_y, k_x) accounting for 4x anisotropy
        subpixel: bool = True,
        scale: Tuple[float, float, float] = DEFAULT_SCALE,
        max_peaks_per_frame: Optional[int] = None,
    ):
        """
        Args:
            threshold: Minimum probability to consider a local maximum as a valid cell detection.
            min_distance: 3D NMS footprint size in voxels (k_z, k_y, k_x).
            subpixel: Whether to apply 3D quadratic Taylor refinement for sub-voxel accuracy.
            scale: Physical scale (s_z, s_y, s_x) in micrometers.
            max_peaks_per_frame: Optional maximum number of highest-scoring peaks to retain per frame.
        """
        self.threshold = threshold
        self.min_distance = min_distance
        self.subpixel = subpixel
        self.scale = scale
        self.max_peaks_per_frame = max_peaks_per_frame

    def extract_peaks_3d(
        self,
        heatmap: np.ndarray,
        threshold: Optional[float] = None,
        max_peaks: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract local maxima coordinates and confidence scores from a 3D heatmap.

        Args:
            heatmap: (Z, Y, X) float32 probability volume.
            threshold: Optional threshold override.
            max_peaks: Optional limit on the number of highest-scoring peaks to retain.

        Returns:
            coords: (N, 3) float32 array of [z, y, x] peak coordinates.
            scores: (N,) float32 array of peak values.
        """
        thresh = threshold if threshold is not None else self.threshold
        limit = max_peaks if max_peaks is not None else self.max_peaks_per_frame
        Z, Y, X = heatmap.shape

        # 3D Maximum Filter (Non-Maximum Suppression)
        footprint = np.ones(self.min_distance, dtype=bool)
        max_filtered = maximum_filter(heatmap, footprint=footprint, mode="constant", cval=0.0)

        # Local maxima mask
        is_peak = (heatmap == max_filtered) & (heatmap >= thresh)
        peak_indices = np.argwhere(is_peak)  # (N, 3)

        if len(peak_indices) == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)

        scores = heatmap[is_peak].astype(np.float32)

        # Retain top-K highest scoring peaks if requested
        if limit is not None and len(peak_indices) > limit:
            top_k_indices = np.argpartition(-scores, limit)[:limit]
            sorted_order = np.argsort(-scores[top_k_indices])
            best_idx = top_k_indices[sorted_order]
            peak_indices = peak_indices[best_idx]
            scores = scores[best_idx]

        if not self.subpixel:
            return peak_indices.astype(np.float32), scores

        # Vectorized sub-voxel 3D quadratic interpolation
        cz, cy, cx = peak_indices[:, 0], peak_indices[:, 1], peak_indices[:, 2]

        # Z axis
        zm = np.clip(cz - 1, 0, Z - 1)
        zp = np.clip(cz + 1, 0, Z - 1)
        v_zm, v_z0, v_zp = heatmap[zm, cy, cx], heatmap[cz, cy, cx], heatmap[zp, cy, cx]
        denom_z = 2.0 * (v_zm - 2.0 * v_z0 + v_zp)
        valid_z = (cz > 0) & (cz < Z - 1) & (np.abs(denom_z) > 1e-5)
        dz = np.zeros_like(cz, dtype=np.float32)
        dz[valid_z] = np.clip(-(v_zp[valid_z] - v_zm[valid_z]) / denom_z[valid_z], -0.5, 0.5)

        # Y axis
        ym = np.clip(cy - 1, 0, Y - 1)
        yp = np.clip(cy + 1, 0, Y - 1)
        v_ym, v_y0, v_yp = heatmap[cz, ym, cx], heatmap[cz, cy, cx], heatmap[cz, yp, cx]
        denom_y = 2.0 * (v_ym - 2.0 * v_y0 + v_yp)
        valid_y = (cy > 0) & (cy < Y - 1) & (np.abs(denom_y) > 1e-5)
        dy = np.zeros_like(cy, dtype=np.float32)
        dy[valid_y] = np.clip(-(v_yp[valid_y] - v_ym[valid_y]) / denom_y[valid_y], -0.5, 0.5)

        # X axis
        xm = np.clip(cx - 1, 0, X - 1)
        xp = np.clip(cx + 1, 0, X - 1)
        v_xm, v_x0, v_xp = heatmap[cz, cy, xm], heatmap[cz, cy, cx], heatmap[cz, cy, xp]
        denom_x = 2.0 * (v_xm - 2.0 * v_x0 + v_xp)
        valid_x = (cx > 0) & (cx < X - 1) & (np.abs(denom_x) > 1e-5)
        dx = np.zeros_like(cx, dtype=np.float32)
        dx[valid_x] = np.clip(-(v_xp[valid_x] - v_xm[valid_x]) / denom_x[valid_x], -0.5, 0.5)

        refined = np.stack([cz + dz, cy + dy, cx + dx], axis=-1).astype(np.float32)
        return refined, scores

    def detect_dataset_nodes(
        self,
        model: nn.Module,
        dataset: DatasetVolume,
        device: torch.device,
        timepoints: Optional[List[int]] = None,
        threshold: Optional[float] = None,
        batch_size: int = 1,
    ) -> pl.DataFrame:
        """Run full-volume detector inference across specified timepoints and extract peaks.

        Args:
            model: Trained TemporalUNet3D model.
            dataset: DatasetVolume instance.
            device: Compute device (cuda or cpu).
            timepoints: List of frame indices to detect (defaults to all timepoints).
            threshold: Detection probability cutoff.
            batch_size: Batch size of frames to evaluate.

        Returns:
            Polars DataFrame with columns: ['t', 'node_id', 'z', 'y', 'x', 'score']
        """
        model.eval()
        thresh = threshold if threshold is not None else self.threshold

        if timepoints is None:
            timepoints = list(range(dataset.shape[0]))

        detected_records: List[Dict[str, Union[int, float]]] = []
        global_node_id = 0

        with torch.no_grad():
            for t in timepoints:
                # Read 3D volume at timepoint t
                vol_crop = dataset.read_spatial_crop(
                    slice(t, t + 1),
                    slice(None),
                    slice(None),
                    slice(None),
                )  # (1, Z, Y, X)

                vol_norm = normalize_intensity(vol_crop, quantiles=dataset.quantiles)
                # Shape: (1, 1, Z, Y, X)
                inp = torch.from_numpy(vol_norm).unsqueeze(0).to(device)

                with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    pred_hm, _ = model(inp)

                hm_3d = pred_hm[0, 0].cpu().numpy().astype(np.float32)  # (Z, Y, X)

                peaks, scores = self.extract_peaks_3d(hm_3d, threshold=thresh)

                for (z, y, x), score in zip(peaks, scores):
                    detected_records.append(
                        {
                            "t": int(t),
                            "node_id": int(global_node_id),
                            "z": float(z),
                            "y": float(y),
                            "x": float(x),
                            "score": float(score),
                        }
                    )
                    global_node_id += 1

        if not detected_records:
            return pl.DataFrame(
                schema={
                    "t": pl.Int64,
                    "node_id": pl.Int64,
                    "z": pl.Float64,
                    "y": pl.Float64,
                    "x": pl.Float64,
                    "score": pl.Float64,
                }
            )

        return pl.DataFrame(detected_records)

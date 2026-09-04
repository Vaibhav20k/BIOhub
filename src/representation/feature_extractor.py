"""Trilinear RoI feature pooling and local morphology extraction for 3D cell detections."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.zarr_reader import DEFAULT_SCALE


class TrilinearRoIPooler(nn.Module):
    """Samples continuous 3D feature representations using trilinear grid interpolation."""

    def __init__(
        self,
        scale: Tuple[float, float, float] = DEFAULT_SCALE,
        sample_neighborhood: bool = True,
        neighborhood_radius_um: float = 1.0,
        in_channels: int = 32,
        out_channels: Optional[int] = None,
    ):
        """
        Args:
            scale: Physical scale (s_z, s_y, s_x) in micrometers per voxel.
            sample_neighborhood: Whether to sample a 7-point morphological neighborhood
                                 (center + 6 directional offsets in physical space).
            neighborhood_radius_um: Physical sampling offset radius in micrometers.
            in_channels: Number of channels in the input feature volume (default: 32).
            out_channels: Optional projected output channel dimension.
        """
        super().__init__()
        self.scale = scale
        self.sample_neighborhood = sample_neighborhood
        self.neighborhood_radius_um = neighborhood_radius_um
        self.in_channels = in_channels

        # Calculate voxel offset radii accounting for 4x Z-anisotropy
        sz, sy, sx = scale
        dz = neighborhood_radius_um / sz
        dy = neighborhood_radius_um / sy
        dx = neighborhood_radius_um / sx

        # 7 sampling offsets: [center, +z, -z, +y, -y, +x, -x] in (z, y, x) voxels
        if sample_neighborhood:
            offsets = [
                [0.0, 0.0, 0.0],
                [+dz, 0.0, 0.0],
                [-dz, 0.0, 0.0],
                [0.0, +dy, 0.0],
                [0.0, -dy, 0.0],
                [0.0, 0.0, +dx],
                [0.0, 0.0, -dx],
            ]
        else:
            offsets = [[0.0, 0.0, 0.0]]

        offsets_tensor = torch.tensor(offsets, dtype=torch.float32)  # (K, 3) where [z, y, x]
        self.register_buffer("offsets", offsets_tensor)
        self.num_sample_points = len(offsets)

        # Feature dimension after neighborhood aggregation
        # We concatenate center feature (C) + mean of directional offsets (C) + std of directional offsets (C)
        if sample_neighborhood:
            pooled_in_dim = in_channels * 3
        else:
            pooled_in_dim = in_channels

        self.out_channels = out_channels if out_channels is not None else pooled_in_dim

        if out_channels is not None:
            self.proj = nn.Sequential(
                nn.Linear(pooled_in_dim, out_channels),
                nn.LayerNorm(out_channels),
                nn.GELU(),
                nn.Linear(out_channels, out_channels),
            )
        else:
            self.proj = nn.Identity()

    def pool_features_single_volume(
        self,
        feature_map: torch.Tensor,
        coords_zyx: torch.Tensor,
    ) -> torch.Tensor:
        """Sample features for a batch of continuous 3D coordinates from a single 3D feature volume.

        Args:
            feature_map: Tensor of shape (1, C, Z, Y, X) or (C, Z, Y, X).
            coords_zyx: Tensor of shape (N, 3) containing continuous [z, y, x] coordinates in voxel units.

        Returns:
            pooled_features: Tensor of shape (N, out_channels).
        """
        if feature_map.ndim == 4:
            feature_map = feature_map.unsqueeze(0)  # (1, C, Z, Y, X)

        assert feature_map.ndim == 5, f"Expected 5D feature map (1, C, Z, Y, X), got shape {feature_map.shape}"
        _, C, Z, Y, X = feature_map.shape
        N = coords_zyx.shape[0]

        if N == 0:
            return torch.empty((0, self.out_channels), device=feature_map.device, dtype=feature_map.dtype)

        # Expand coordinates with 7 neighborhood offsets:
        # coords_zyx: (N, 1, 3) + offsets: (1, K, 3) -> (N, K, 3) in [z, y, x]
        sampled_coords = coords_zyx.unsqueeze(1) + self.offsets.unsqueeze(0).to(coords_zyx.device)

        # Normalize to [-1, 1] for grid_sample.
        # Note: grid_sample expects coordinates in (x, y, z) order!
        z_norm = 2.0 * sampled_coords[..., 0] / max(1, Z - 1) - 1.0
        y_norm = 2.0 * sampled_coords[..., 1] / max(1, Y - 1) - 1.0
        x_norm = 2.0 * sampled_coords[..., 2] / max(1, X - 1) - 1.0

        # Grid shape for grid_sample: (1, N, K, 1, 3) where last dimension is (x, y, z)
        grid = torch.stack([x_norm, y_norm, z_norm], dim=-1).unsqueeze(0).unsqueeze(3)  # (1, N, K, 1, 3)

        # Trilinear sampling
        # Output shape: (1, C, N, K, 1)
        sampled = F.grid_sample(
            feature_map,
            grid,
            mode="bilinear",  # "bilinear" in 5D means trilinear interpolation
            padding_mode="border",
            align_corners=True,
        )

        # Squeeze singleton dims -> (C, N, K) -> permute to (N, K, C)
        sampled = sampled.squeeze(0).squeeze(-1).permute(1, 2, 0)  # (N, K, C)

        if not self.sample_neighborhood:
            features = sampled[:, 0, :]  # (N, C)
        else:
            center_feat = sampled[:, 0, :]  # (N, C)
            neighbor_feats = sampled[:, 1:, :]  # (N, 6, C)
            mean_neighbor = neighbor_feats.mean(dim=1)  # (N, C)
            std_neighbor = neighbor_feats.std(dim=1)  # (N, C)
            features = torch.cat([center_feat, mean_neighbor, std_neighbor], dim=-1)  # (N, 3*C)

        return self.proj(features)

    def forward(
        self,
        feature_map: torch.Tensor,
        coords_zyx: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for feature pooling."""
        return self.pool_features_single_volume(feature_map, coords_zyx)


def extract_patch_intensity_stats(
    volume: np.ndarray,
    coords_zyx: np.ndarray,
    radius: Tuple[int, int, int] = (1, 3, 3),
) -> np.ndarray:
    """Extract local microscopy image intensity statistics around candidate centroids.

    Args:
        volume: 3D numpy array (Z, Y, X) of normalized raw pixel intensities.
        coords_zyx: (N, 3) array of [z, y, x] cell coordinates.
        radius: (rz, ry, rx) bounding box radius around centroid.

    Returns:
        stats: (N, 4) array containing [mean_intensity, max_intensity, min_intensity, std_intensity].
    """
    Z, Y, X = volume.shape
    N = len(coords_zyx)
    stats = np.zeros((N, 4), dtype=np.float32)

    rz, ry, rx = radius

    for i, (cz, cy, cx) in enumerate(coords_zyx):
        iz, iy, ix = int(round(cz)), int(round(cy)), int(round(cx))
        z0 = max(0, iz - rz)
        z1 = min(Z, iz + rz + 1)
        y0 = max(0, iy - ry)
        y1 = min(Y, iy + ry + 1)
        x0 = max(0, ix - rx)
        x1 = min(X, ix + rx + 1)

        patch = volume[z0:z1, y0:y1, x0:x1]
        if patch.size > 0:
            stats[i, 0] = float(np.mean(patch))
            stats[i, 1] = float(np.max(patch))
            stats[i, 2] = float(np.min(patch))
            stats[i, 3] = float(np.std(patch))

    return stats

"""Intensity normalization and coordinate transformation utilities for microscopy."""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

DEFAULT_SCALE: Tuple[float, float, float] = (1.625, 0.40625, 0.40625)


def normalize_intensity(
    image: Union[np.ndarray, torch.Tensor],
    quantiles: Optional[Dict[str, float]] = None,
    q_low: float = 0.01,
    q_high: float = 0.999,
    gamma: float = 1.0,
    eps: float = 1e-6,
) -> Union[np.ndarray, torch.Tensor]:
    """Normalize raw microscopy intensity to [0, 1] using robust quantiles.

    Args:
        image: Array or Tensor of shape (..., Z, Y, X).
        quantiles: Precomputed quantile dict from Zarr metadata.
        q_low: Lower quantile threshold (default 0.01 / 1%).
        q_high: Upper quantile threshold (default 0.999 / 99.9%).
        gamma: Exponential contrast curve (1.0 = linear).
        eps: Small epsilon to prevent division by zero.

    Returns:
        Normalized array/tensor in range [0, 1] with dtype float32.
    """
    is_torch = isinstance(image, torch.Tensor)

    # Resolve quantile values
    v_low = None
    v_high = None

    if quantiles:
        # Check direct string keys e.g. "0.01" and "0.999"
        k_low = str(q_low)
        k_high = str(q_high)
        if k_low in quantiles and k_high in quantiles:
            v_low = float(quantiles[k_low])
            v_high = float(quantiles[k_high])

    if v_low is None or v_high is None:
        if is_torch:
            flat = image.flatten().float()
            # Subsample for speed if very large
            if flat.numel() > 100_000:
                flat = flat[:: max(1, flat.numel() // 50_000)]
            v_low = float(torch.quantile(flat, q_low))
            v_high = float(torch.quantile(flat, q_high))
        else:
            flat = np.asarray(image).ravel().astype(np.float32)
            if flat.size > 100_000:
                flat = flat[:: max(1, flat.size // 50_000)]
            v_low = float(np.quantile(flat, q_low))
            v_high = float(np.quantile(flat, q_high))

    v_diff = max(v_high - v_low, eps)

    if is_torch:
        norm = torch.clamp((image.float() - v_low) / v_diff, 0.0, 1.0)
        if gamma != 1.0:
            norm = torch.pow(norm, gamma)
        return norm
    else:
        norm = np.clip((image.astype(np.float32) - v_low) / v_diff, 0.0, 1.0)
        if gamma != 1.0:
            norm = np.power(norm, gamma)
        return norm


def pixel_to_physical(
    coords: Union[np.ndarray, torch.Tensor],
    scale: Tuple[float, float, float] = DEFAULT_SCALE,
) -> Union[np.ndarray, torch.Tensor]:
    """Convert (Z, Y, X) pixel coordinates to physical micrometers."""
    scale_arr = np.array(scale, dtype=np.float32)
    if isinstance(coords, torch.Tensor):
        scale_t = torch.tensor(scale, device=coords.device, dtype=coords.dtype)
        return coords * scale_t
    return coords * scale_arr


def physical_to_pixel(
    coords: Union[np.ndarray, torch.Tensor],
    scale: Tuple[float, float, float] = DEFAULT_SCALE,
) -> Union[np.ndarray, torch.Tensor]:
    """Convert (Z, Y, X) physical micrometer coordinates to pixel coordinates."""
    scale_arr = np.array(scale, dtype=np.float32)
    if isinstance(coords, torch.Tensor):
        scale_t = torch.tensor(scale, device=coords.device, dtype=coords.dtype)
        return coords / scale_t
    return coords / scale_arr


def compute_physical_distance_matrix(
    coords_a: Union[np.ndarray, torch.Tensor],
    coords_b: Union[np.ndarray, torch.Tensor],
    scale: Tuple[float, float, float] = DEFAULT_SCALE,
) -> Union[np.ndarray, torch.Tensor]:
    """Compute pairwise Euclidean distance in physical micrometer space.

    Args:
        coords_a: Array/Tensor of shape (N, 3) representing [z, y, x] in pixels.
        coords_b: Array/Tensor of shape (M, 3) representing [z, y, x] in pixels.
        scale: Voxel physical dimensions (scale_z, scale_y, scale_x).

    Returns:
        Distance matrix of shape (N, M) with distances in micrometers.
    """
    is_torch = isinstance(coords_a, torch.Tensor)
    if is_torch:
        scale_t = torch.tensor(scale, device=coords_a.device, dtype=coords_a.dtype)
        ca = coords_a * scale_t  # (N, 3)
        cb = coords_b * scale_t  # (M, 3)
        return torch.cdist(ca, cb, p=2)
    else:
        scale_arr = np.array(scale, dtype=np.float32)
        ca = coords_a * scale_arr
        cb = coords_b * scale_arr
        diff = ca[:, np.newaxis, :] - cb[np.newaxis, :, :]  # (N, M, 3)
        return np.linalg.norm(diff, axis=-1)

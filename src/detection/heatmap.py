"""3D Gaussian heatmap target generator for cell nucleus detection."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import polars as pl
import torch

from src.data.zarr_reader import DEFAULT_SCALE


class GaussianHeatmapGenerator:
    """Generates continuous 3D Gaussian probability heatmaps from discrete cell coordinates."""

    def __init__(
        self,
        scale: Tuple[float, float, float] = DEFAULT_SCALE,
        sigma_um: float = 1.5,
    ):
        """
        Args:
            scale: Physical voxel dimensions (s_z, s_y, s_x) in micrometers.
            sigma_um: Standard deviation of Gaussian cell representation in micrometers.
        """
        self.scale = scale
        self.sigma_um = sigma_um
        self.sz, self.sy, self.sx = scale

        # Calculate bounding box radii in voxels (3 * sigma rule)
        self.rz = max(1, int(np.ceil(3.0 * sigma_um / self.sz)))
        self.ry = max(1, int(np.ceil(3.0 * sigma_um / self.sy)))
        self.rx = max(1, int(np.ceil(3.0 * sigma_um / self.sx)))

    def generate_3d(
        self,
        shape: Tuple[int, int, int],
        centers: Union[np.ndarray, List[Tuple[float, float, float]]],
    ) -> np.ndarray:
        """Generate a 3D Gaussian heatmap for a single volume.

        Args:
            shape: (Z, Y, X) volume dimensions.
            centers: (K, 3) array or list of (z, y, x) cell coordinates in voxels.

        Returns:
            heatmap: (Z, Y, X) float32 numpy array with values in [0, 1].
        """
        Z, Y, X = shape
        heatmap = np.zeros(shape, dtype=np.float32)

        if len(centers) == 0:
            return heatmap

        centers_arr = np.asarray(centers, dtype=np.float32)

        for cz, cy, cx in centers_arr:
            z_min = max(0, int(np.floor(cz - self.rz)))
            z_max = min(Z, int(np.ceil(cz + self.rz + 1)))
            y_min = max(0, int(np.floor(cy - self.ry)))
            y_max = min(Y, int(np.ceil(cy + self.ry + 1)))
            x_min = max(0, int(np.floor(cx - self.rx)))
            x_max = min(X, int(np.ceil(cx + self.rx + 1)))

            if z_min >= z_max or y_min >= y_max or x_min >= x_max:
                continue

            zg = np.arange(z_min, z_max, dtype=np.float32)
            yg = np.arange(y_min, y_max, dtype=np.float32)
            xg = np.arange(x_min, x_max, dtype=np.float32)

            zz, yy, xx = np.meshgrid(zg, yg, xg, indexing="ij")
            dist_sq = (
                ((zz - cz) * self.sz) ** 2
                + ((yy - cy) * self.sy) ** 2
                + ((xx - cx) * self.sx) ** 2
            )
            g = np.exp(-0.5 * dist_sq / (self.sigma_um ** 2))

            sub_hm = heatmap[z_min:z_max, y_min:y_max, x_min:x_max]
            heatmap[z_min:z_max, y_min:y_max, x_min:x_max] = np.maximum(sub_hm, g)

        return heatmap

    def generate_spatiotemporal_4d(
        self,
        shape: Tuple[int, int, int, int],
        nodes_by_step: Dict[int, List[Dict[str, float]]],
    ) -> torch.Tensor:
        """Generate a 4D spatio-temporal heatmap tensor for a patch sample.

        Args:
            shape: (W, Z, Y, X) where W is window size in frames.
            nodes_by_step: Dict mapping tau -> list of dicts with 'z', 'y', 'x'.

        Returns:
            heatmap_tensor: (1, W, Z, Y, X) torch.FloatTensor in [0, 1].
        """
        W, Z, Y, X = shape
        heatmaps = []

        for tau in range(W):
            nodes_at_tau = nodes_by_step.get(tau, [])
            centers = [[n["z"], n["y"], n["x"]] for n in nodes_at_tau]
            hm_3d = self.generate_3d((Z, Y, X), centers)
            heatmaps.append(hm_3d)

        hm_4d = np.stack(heatmaps, axis=0)  # (W, Z, Y, X)
        tensor = torch.from_numpy(hm_4d).unsqueeze(0).float()  # (1, W, Z, Y, X)
        return tensor


def generate_gaussian_heatmap_3d(
    shape: Tuple[int, int, int],
    centers: Union[np.ndarray, List[Tuple[float, float, float]]],
    scale: Tuple[float, float, float] = DEFAULT_SCALE,
    sigma_um: float = 1.5,
) -> np.ndarray:
    """Convenience function to generate a single 3D Gaussian heatmap."""
    generator = GaussianHeatmapGenerator(scale=scale, sigma_um=sigma_um)
    return generator.generate_3d(shape, centers)

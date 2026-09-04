"""Spatio-temporal 3D+t patch sampler and PyTorch dataset for cell tracking."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import normalize_intensity
from src.data.zarr_reader import DatasetVolume


class SpatioTemporalPatchSampler(Dataset):
    """Samples 3D+t spatio-temporal crops and localized subgraphs from microscopy volumes."""

    def __init__(
        self,
        dataset: DatasetVolume,
        patch_size: Tuple[int, int, int] = (32, 128, 128),  # (Z, Y, X)
        window_size: int = 2,  # Number of frames W
        samples_per_epoch: int = 100,
        center_on_cell_prob: float = 0.8,
        normalize: bool = True,
        seed: Optional[int] = None,
    ):
        self.dataset = dataset
        self.patch_size = patch_size
        self.window_size = window_size
        self.samples_per_epoch = samples_per_epoch
        self.center_on_cell_prob = center_on_cell_prob
        self.normalize = normalize
        self.rng = np.random.RandomState(seed)

        self.T, self.Z, self.Y, self.X = dataset.shape
        self.pz, self.py, self.px = patch_size

        assert self.pz <= self.Z, f"Patch Z {self.pz} exceeds volume Z {self.Z}"
        assert self.py <= self.Y, f"Patch Y {self.py} exceeds volume Y {self.Y}"
        assert self.px <= self.X, f"Patch X {self.px} exceeds volume X {self.X}"
        assert self.window_size <= self.T, f"Window size {self.window_size} exceeds T {self.T}"

        # Cache ground truth nodes dataframe if tracks are present
        self.node_df: Optional[pl.DataFrame] = None
        self.edge_df: Optional[pl.DataFrame] = None
        if dataset.tracks is not None:
            self.node_df = dataset.tracks.node_attrs()
            self.edge_df = dataset.tracks.edge_attrs()

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _sample_crop_box(self) -> Tuple[int, int, int, int]:
        """Sample (t_start, z_start, y_start, x_start) bounding box."""
        # 1. Temporal start
        max_t = self.T - self.window_size
        t_start = int(self.rng.randint(0, max_t + 1)) if max_t > 0 else 0

        # 2. Decide whether to center on an annotated cell
        use_cell_center = (
            self.node_df is not None
            and self.node_df.height > 0
            and self.rng.rand() < self.center_on_cell_prob
        )

        if use_cell_center:
            row_idx = self.rng.randint(0, self.node_df.height)
            row = self.node_df.row(row_idx, named=True)
            node_t = int(row["t"])
            cz, cy, cx = float(row["z"]), float(row["y"]), float(row["x"])

            # Ensure node_t lies inside [t_start, t_start + window_size)
            max_offset = min(self.window_size - 1, node_t)
            t_offset = self.rng.randint(0, max_offset + 1) if max_offset >= 0 else 0
            t_start = int(np.clip(node_t - t_offset, 0, max(0, self.T - self.window_size)))

            # Add random jitter around cell centroid
            jz = self.rng.randint(-self.pz // 4, self.pz // 4 + 1)
            jy = self.rng.randint(-self.py // 4, self.py // 4 + 1)
            jx = self.rng.randint(-self.px // 4, self.px // 4 + 1)

            z_start = int(np.clip(cz - self.pz // 2 + jz, 0, max(0, self.Z - self.pz)))
            y_start = int(np.clip(cy - self.py // 2 + jy, 0, max(0, self.Y - self.py)))
            x_start = int(np.clip(cx - self.px // 2 + jx, 0, max(0, self.X - self.px)))
            return t_start, z_start, y_start, x_start

        # Uniform random spatial sampling
        z_start = int(self.rng.randint(0, self.Z - self.pz + 1)) if self.Z > self.pz else 0
        y_start = int(self.rng.randint(0, self.Y - self.py + 1)) if self.Y > self.py else 0
        x_start = int(self.rng.randint(0, self.X - self.px + 1)) if self.X > self.px else 0
        return t_start, z_start, y_start, x_start

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Dict, List]]:
        t_start, z_start, y_start, x_start = self._sample_crop_box()
        t_end = t_start + self.window_size
        z_end = z_start + self.pz
        y_end = y_start + self.py
        x_end = x_start + self.px

        # Read 4D volume patch
        patch = self.dataset.read_spatial_crop(
            slice(t_start, t_end),
            slice(z_start, z_end),
            slice(y_start, y_end),
            slice(x_start, x_end),
        )

        if self.normalize:
            patch = normalize_intensity(patch, quantiles=self.dataset.quantiles)

        # Shape: (1, W, Z, Y, X)
        patch_tensor = torch.from_numpy(patch).unsqueeze(0).float()

        # Extract localized nodes
        nodes_by_step: Dict[int, List[Dict[str, float]]] = {
            tau: [] for tau in range(self.window_size)
        }
        node_id_to_local_idx: Dict[int, Tuple[int, int]] = {}

        if self.node_df is not None and self.node_df.height > 0:
            in_box = self.node_df.filter(
                (pl.col("t") >= t_start)
                & (pl.col("t") < t_end)
                & (pl.col("z") >= z_start)
                & (pl.col("z") < z_end)
                & (pl.col("y") >= y_start)
                & (pl.col("y") < y_end)
                & (pl.col("x") >= x_start)
                & (pl.col("x") < x_end)
            )

            for row in in_box.iter_rows(named=True):
                nid = int(row["node_id"])
                tau = int(row["t"]) - t_start
                lz = float(row["z"]) - z_start
                ly = float(row["y"]) - y_start
                lx = float(row["x"]) - x_start
                local_idx = len(nodes_by_step[tau])
                nodes_by_step[tau].append(
                    {"node_id": nid, "tau": tau, "z": lz, "y": ly, "x": lx}
                )
                node_id_to_local_idx[nid] = (tau, local_idx)

        # Extract transition matrices across consecutive frames tau -> tau + 1
        transitions = []
        for tau in range(self.window_size - 1):
            n_src = len(nodes_by_step[tau])
            n_tgt = len(nodes_by_step[tau + 1])
            mat = torch.zeros((n_src, n_tgt), dtype=torch.float32)

            if n_src > 0 and n_tgt > 0 and self.edge_df is not None:
                src_ids = {n["node_id"]: idx for idx, n in enumerate(nodes_by_step[tau])}
                tgt_ids = {n["node_id"]: idx for idx, n in enumerate(nodes_by_step[tau + 1])}

                for row in self.edge_df.iter_rows(named=True):
                    s = int(row["source_id"])
                    tgt = int(row["target_id"])
                    if s in src_ids and tgt in tgt_ids:
                        mat[src_ids[s], tgt_ids[tgt]] = 1.0

            transitions.append(mat)

        return {
            "image": patch_tensor,
            "origin": (t_start, z_start, y_start, x_start),
            "nodes_by_step": nodes_by_step,
            "transitions": transitions,
        }

"""PyTorch Dataset for 3D/4D cell detection training with on-the-fly heatmap generation."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.patch_sampler import SpatioTemporalPatchSampler
from src.data.zarr_reader import DatasetVolume
from src.detection.heatmap import GaussianHeatmapGenerator


class DetectionDataset(Dataset):
    """Wraps SpatioTemporalPatchSampler to generate paired (image, heatmap, mask) tensors."""

    def __init__(
        self,
        dataset: DatasetVolume,
        patch_size: Tuple[int, int, int] = (32, 128, 128),
        window_size: int = 2,
        samples_per_epoch: int = 100,
        center_on_cell_prob: float = 0.8,
        sigma_um: float = 1.5,
        augment: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Args:
            dataset: DatasetVolume instance.
            patch_size: (Z, Y, X) spatial dimensions.
            window_size: Number of consecutive time frames W.
            samples_per_epoch: Length of an epoch in sampled patches.
            center_on_cell_prob: Probability of centering a crop on an annotated cell.
            sigma_um: Standard deviation of Gaussian target in micrometers.
            augment: Whether to apply spatial flips and lateral rotations.
            seed: Random seed.
        """
        self.sampler = SpatioTemporalPatchSampler(
            dataset=dataset,
            patch_size=patch_size,
            window_size=window_size,
            samples_per_epoch=samples_per_epoch,
            center_on_cell_prob=center_on_cell_prob,
            normalize=True,
            seed=seed,
        )
        self.heatmap_generator = GaussianHeatmapGenerator(
            scale=dataset.scale,
            sigma_um=sigma_um,
        )
        self.augment = augment
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Tuple, Dict]]:
        sample = self.sampler[idx]
        image = sample["image"]  # (1, W, Z, Y, X)
        nodes_by_step = sample["nodes_by_step"]
        origin = sample["origin"]  # (t_start, z_start, y_start, x_start)
        W, Z, Y, X = self.sampler.window_size, self.sampler.pz, self.sampler.py, self.sampler.px

        # Generate continuous Gaussian ground-truth heatmap: (1, W, Z, Y, X)
        heatmap = self.heatmap_generator.generate_spatiotemporal_4d(
            shape=(W, Z, Y, X),
            nodes_by_step=nodes_by_step,
        )

        # Synchronous Spatial Augmentations (applies identically to image and heatmap)
        if self.augment:
            # 1. Random horizontal/vertical lateral flips (dim 3=Y, dim 4=X)
            if self.rng.rand() < 0.5:
                image = torch.flip(image, dims=[4])  # Flip X
                heatmap = torch.flip(heatmap, dims=[4])

            if self.rng.rand() < 0.5:
                image = torch.flip(image, dims=[3])  # Flip Y
                heatmap = torch.flip(heatmap, dims=[3])

            # 2. Random axial flip (dim 2=Z)
            if self.rng.rand() < 0.5:
                image = torch.flip(image, dims=[2])  # Flip Z
                heatmap = torch.flip(heatmap, dims=[2])

            # 3. Random 90-degree lateral rotation in Y-X plane (k in {0, 1, 2, 3})
            k_rot = self.rng.randint(0, 4)
            if k_rot > 0:
                image = torch.rot90(image, k=k_rot, dims=[3, 4])
                heatmap = torch.rot90(heatmap, k=k_rot, dims=[3, 4])

            # 4. Intensity jitter on image only
            scale_factor = float(self.rng.uniform(0.85, 1.15))
            shift_factor = float(self.rng.uniform(-0.05, 0.05))
            image = torch.clamp(image * scale_factor + shift_factor, 0.0, 1.0)

        # Frame supervision mask: 1.0 if frame has annotations or is within annotated window
        frame_mask = torch.ones((1, W, 1, 1, 1, 1), dtype=torch.float32)

        return {
            "image": image,
            "heatmap": heatmap,
            "frame_mask": frame_mask,
            "origin": origin,
        }

"""3D Spatio-temporal data augmentations preserving coordinate and graph geometry."""

from typing import Dict, List, Tuple
import numpy as np
import torch


class SpatialMicroscopyAugmentation:
    """Applies valid 3D microscopy augmentations to image patches and node coordinates."""

    def __init__(
        self,
        flip_prob: float = 0.5,
        rot90_prob: float = 0.5,
        intensity_scale_prob: float = 0.5,
        noise_prob: float = 0.3,
        seed: int = 42,
    ):
        self.flip_prob = flip_prob
        self.rot90_prob = rot90_prob
        self.intensity_scale_prob = intensity_scale_prob
        self.noise_prob = noise_prob
        self.rng = np.random.RandomState(seed)

    def __call__(
        self,
        image: torch.Tensor,
        nodes_by_step: Dict[int, List[Dict[str, float]]],
    ) -> Tuple[torch.Tensor, Dict[int, List[Dict[str, float]]]]:
        """Apply augmentations.

        Args:
            image: Tensor of shape (C, W, Z, Y, X) where C is channel (usually 1).
            nodes_by_step: Dict mapping time step tau -> list of node dicts {'node_id', 'z', 'y', 'x'}.

        Returns:
            Augmented image tensor and updated nodes_by_step dict.
        """
        C, W, Z, Y, X = image.shape
        img = image.clone()
        aug_nodes: Dict[int, List[Dict[str, float]]] = {
            tau: [dict(n) for n in nodes_by_step[tau]] for tau in nodes_by_step
        }

        # 1. Lateral X-Flip (dim 4)
        if self.rng.rand() < self.flip_prob:
            img = torch.flip(img, dims=[4])
            for tau in aug_nodes:
                for n in aug_nodes[tau]:
                    n["x"] = float(X - 1.0 - n["x"])

        # 2. Lateral Y-Flip (dim 3)
        if self.rng.rand() < self.flip_prob:
            img = torch.flip(img, dims=[3])
            for tau in aug_nodes:
                for n in aug_nodes[tau]:
                    n["y"] = float(Y - 1.0 - n["y"])

        # 3. Axial Z-Flip (dim 2)
        if self.rng.rand() < self.flip_prob:
            img = torch.flip(img, dims=[2])
            for tau in aug_nodes:
                for n in aug_nodes[tau]:
                    n["z"] = float(Z - 1.0 - n["z"])

        # 4. In-plane 90-degree rotation (Y-X plane only, preserving Z anisotropy)
        if self.rng.rand() < self.rot90_prob and Y == X:
            k = int(self.rng.randint(1, 4))
            img = torch.rot90(img, k=k, dims=[3, 4])
            for tau in aug_nodes:
                for n in aug_nodes[tau]:
                    y, x = n["y"], n["x"]
                    if k == 1:
                        n["y"], n["x"] = float(x), float(Y - 1.0 - y)
                    elif k == 2:
                        n["y"], n["x"] = float(Y - 1.0 - y), float(X - 1.0 - x)
                    elif k == 3:
                        n["y"], n["x"] = float(X - 1.0 - x), float(y)

        # 5. Contrast/intensity scaling
        if self.rng.rand() < self.intensity_scale_prob:
            alpha = float(self.rng.uniform(0.8, 1.2))
            img = torch.clamp(img * alpha, 0.0, 1.0)

        # 6. Gaussian noise injection
        if self.rng.rand() < self.noise_prob:
            sigma = float(self.rng.uniform(0.005, 0.02))
            noise = torch.randn_like(img) * sigma
            img = torch.clamp(img + noise, 0.0, 1.0)

        return img, aug_nodes

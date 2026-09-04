"""Multi-frequency continuous Fourier spatio-temporal positional encodings for cell tracking."""

import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from src.data.zarr_reader import DEFAULT_SCALE


class SpatioTemporalFourierEncoding(nn.Module):
    """Continuous multi-scale Fourier sinusoidal positional encoding for 4D (t, z, y, x) coordinates.
    
    Transforms anisotropic continuous coordinates in physical units (micrometers)
    into rich frequency-domain representations via log-spaced sinusoidal bands.
    """

    def __init__(
        self,
        num_frequency_bands: int = 8,
        min_freq_log2: float = -2.0,
        max_freq_log2: float = 6.0,
        scale: Tuple[float, float, float] = DEFAULT_SCALE,
        dt_scale: float = 1.0,
        include_raw: bool = True,
        proj_dim: Optional[int] = None,
    ):
        """
        Args:
            num_frequency_bands: Number of frequency octaves per coordinate axis.
            min_freq_log2: Log2 of base frequency.
            max_freq_log2: Log2 of highest frequency.
            scale: Physical spatial scale (s_z, s_y, s_x) in micrometers per voxel.
            dt_scale: Temporal scale factor (time unit per frame step).
            include_raw: Whether to concatenate normalized raw coordinates to the encoding.
            proj_dim: Optional output projection dimension (via MLP). If None, raw Fourier dim is used.
        """
        super().__init__()
        self.num_frequency_bands = num_frequency_bands
        self.include_raw = include_raw
        self.scale = scale
        self.dt_scale = dt_scale

        # Register physical coordinate scale factors as buffer: [s_t, s_z, s_y, s_x]
        coord_scales = torch.tensor([dt_scale, scale[0], scale[1], scale[2]], dtype=torch.float32)
        self.register_buffer("coord_scales", coord_scales)

        # Log-spaced frequency bands: 2^[min ... max]
        freq_bands = 2.0 ** torch.linspace(
            min_freq_log2, max_freq_log2, num_frequency_bands, dtype=torch.float32
        )
        self.register_buffer("freq_bands", freq_bands)

        # Dimension calculation: 4 coordinates * num_bands * 2 (sin + cos) = 8 * num_bands
        raw_dim = 4 if include_raw else 0
        self.fourier_dim = 4 * num_frequency_bands * 2 + raw_dim

        if proj_dim is not None:
            self.output_dim = proj_dim
            self.projection = nn.Sequential(
                nn.Linear(self.fourier_dim, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
            )
        else:
            self.output_dim = self.fourier_dim
            self.projection = nn.Identity()

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Encode spatio-temporal coordinates into continuous Fourier embeddings.

        Args:
            coords: Tensor of shape (..., 4) containing [t, z, y, x] coordinates in voxel units.

        Returns:
            encodings: Tensor of shape (..., output_dim) with continuous positional embeddings.
        """
        assert coords.shape[-1] == 4, f"Expected last dimension 4 for (t, z, y, x), got {coords.shape[-1]}"

        # Convert to physical units (micrometers / seconds)
        # coords: (..., 4), coord_scales: (4,)
        phys_coords = coords * self.coord_scales

        # Expand dims for frequency multiplication:
        # phys_coords: (..., 4, 1), freq_bands: (num_bands,)
        scaled = phys_coords.unsqueeze(-1) * self.freq_bands * math.pi  # (..., 4, num_bands)

        # Compute sinusoidal responses
        sin_part = torch.sin(scaled)
        cos_part = torch.cos(scaled)

        # Flatten frequency dimension: (..., 4 * num_bands * 2)
        # Interleave or concatenate sin and cos
        fourier = torch.cat([sin_part, cos_part], dim=-1)  # (..., 4, 2 * num_bands)
        flat_shape = list(coords.shape[:-1]) + [4 * self.num_frequency_bands * 2]
        fourier = fourier.reshape(flat_shape)

        if self.include_raw:
            # Append normalized physical coordinates
            fourier = torch.cat([fourier, phys_coords], dim=-1)

        # Apply optional learnable projection
        return self.projection(fourier)


class RelativeSpatialEncoding(nn.Module):
    """Encodes continuous 3D relative displacements Delta p = p_v - p_u between cell pairs.
    
    Used by Transformer edge attention to condition edge affinity on anisotropic physical distance.
    """

    def __init__(
        self,
        num_frequency_bands: int = 6,
        scale: Tuple[float, float, float] = DEFAULT_SCALE,
        output_dim: int = 32,
    ):
        """
        Args:
            num_frequency_bands: Number of sinusoidal frequency bands for relative offsets.
            scale: Physical scale (s_z, s_y, s_x) in micrometers.
            output_dim: Dimension of projected relative embedding.
        """
        super().__init__()
        self.scale = scale
        self.num_frequency_bands = num_frequency_bands

        spatial_scales = torch.tensor([scale[0], scale[1], scale[2]], dtype=torch.float32)
        self.register_buffer("spatial_scales", spatial_scales)

        # Frequencies tuned for displacements from 0.1 µm to 15.0 µm
        freq_bands = 2.0 ** torch.linspace(-1.0, 4.0, num_frequency_bands, dtype=torch.float32)
        self.register_buffer("freq_bands", freq_bands)

        # Inputs: 3 coordinates * num_bands * 2 + 3 (raw Delta) + 1 (Euclidean norm)
        raw_dim = 3 * num_frequency_bands * 2 + 4
        self.output_dim = output_dim

        self.projection = nn.Sequential(
            nn.Linear(raw_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, delta_zyx: torch.Tensor) -> torch.Tensor:
        """Encode relative voxel displacements into invariant physical spatial embeddings.

        Args:
            delta_zyx: Tensor of shape (..., 3) containing [Delta z, Delta y, Delta x] in voxels.

        Returns:
            rel_embeddings: Tensor of shape (..., output_dim).
        """
        # Convert to physical micrometers
        phys_delta = delta_zyx * self.spatial_scales  # (..., 3)
        euclidean_dist = torch.linalg.norm(phys_delta, dim=-1, keepdim=True)  # (..., 1)

        # Fourier components
        scaled = phys_delta.unsqueeze(-1) * self.freq_bands * math.pi  # (..., 3, num_bands)
        sin_part = torch.sin(scaled)
        cos_part = torch.cos(scaled)
        fourier = torch.cat([sin_part, cos_part], dim=-1)  # (..., 3, 2 * num_bands)
        flat_shape = list(delta_zyx.shape[:-1]) + [3 * self.num_frequency_bands * 2]
        fourier = fourier.reshape(flat_shape)

        # Concatenate Fourier features, raw physical delta, and physical Euclidean distance
        features = torch.cat([fourier, phys_delta, euclidean_dist], dim=-1)
        return self.projection(features)

"""Unified cell node representation fusing 3D deep visual features, continuous coordinates, and confidence."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.amp import autocast

from src.data.preprocessing import normalize_intensity
from src.data.zarr_reader import DatasetVolume, DEFAULT_SCALE
from src.representation.feature_extractor import TrilinearRoIPooler, extract_patch_intensity_stats
from src.representation.positional_encoding import SpatioTemporalFourierEncoding


class CellNodeEmbedding(nn.Module):
    """Deep multi-modal embedding module that fuses visual morphology, spatial location, and confidence.
    
    Transforms heterogeneous cell node properties:
        [Visual RoI features (32D or 96D) || 4D Fourier PE (64D) || Detection Score (1D)]
    into unified, fixed-dimensional embedding vectors for Transformer tracking.
    """

    def __init__(
        self,
        in_visual_dim: int = 32,
        num_frequency_bands: int = 8,
        embedding_dim: int = 128,
        scale: Tuple[float, float, float] = DEFAULT_SCALE,
        sample_neighborhood: bool = True,
        dropout: float = 0.1,
    ):
        """
        Args:
            in_visual_dim: Channel dimension of backbone feature volume (default: 32).
            num_frequency_bands: Frequency octaves for spatio-temporal Fourier encoding.
            embedding_dim: Final unified cell embedding dimension (default: 128).
            scale: Physical scale (s_z, s_y, s_x) in micrometers per voxel.
            sample_neighborhood: Whether to use 7-point morphological neighborhood pooling.
            dropout: Dropout probability.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.scale = scale

        # Trilinear RoI Pooler
        # If neighborhood sampling is True, pooled dimension is 3 * in_visual_dim
        self.roi_pooler = TrilinearRoIPooler(
            scale=scale,
            sample_neighborhood=sample_neighborhood,
            in_channels=in_visual_dim,
            out_channels=64,  # Project pooled visual features to 64D
        )

        # Spatio-Temporal Fourier Positional Encoder
        self.pos_encoder = SpatioTemporalFourierEncoding(
            num_frequency_bands=num_frequency_bands,
            scale=scale,
            include_raw=True,
            proj_dim=64,  # Project 4D coordinates to 64D
        )

        # Input dimension to fusion MLP:
        # 64 (visual) + 64 (pos) + 1 (score) + 4 (raw intensity stats) = 133
        fusion_in_dim = 64 + 64 + 1 + 4

        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        coords_tzyx: torch.Tensor,
        scores: torch.Tensor,
        intensity_stats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse visual, positional, and detection confidence into unified cell node representations.

        Args:
            visual_features: (N, C_roi) tensor from RoI pooling (projected or raw).
            coords_tzyx: (N, 4) tensor containing [t, z, y, x] in voxel coordinates.
            scores: (N, 1) or (N,) tensor of detection confidence scores in [0, 1].
            intensity_stats: Optional (N, 4) tensor containing [mean, max, min, std] intensity.

        Returns:
            embeddings: (N, embedding_dim) tensor of unified node embeddings.
        """
        N = coords_tzyx.shape[0]
        if N == 0:
            return torch.empty((0, self.embedding_dim), device=coords_tzyx.device, dtype=torch.float32)

        if scores.ndim == 1:
            scores = scores.unsqueeze(-1)

        # Encode coordinates: (N, 64)
        pos_emb = self.pos_encoder(coords_tzyx)

        # If visual_features need projection via roi_pooler
        if visual_features.shape[-1] != 64 and hasattr(self.roi_pooler, "proj"):
            vis_emb = self.roi_pooler.proj(visual_features)
        else:
            vis_emb = visual_features

        # Default intensity stats if not provided
        if intensity_stats is None:
            intensity_stats = torch.zeros((N, 4), device=coords_tzyx.device, dtype=coords_tzyx.dtype)

        # Concatenate: (N, 64 + 64 + 1 + 4 = 133)
        cat_features = torch.cat([vis_emb, pos_emb, scores, intensity_stats], dim=-1)

        return self.fusion_mlp(cat_features)


def extract_node_embeddings_for_dataset(
    model: nn.Module,
    embedder: CellNodeEmbedding,
    dataset: DatasetVolume,
    nodes_df: pl.DataFrame,
    device: torch.device,
) -> Tuple[pl.DataFrame, torch.Tensor]:
    """Run full-dataset feature extraction and embedding for a DataFrame of detected cell nodes.

    Args:
        model: Trained TemporalUNet3D model producing (pred_heatmap, feat_map).
        embedder: CellNodeEmbedding instance.
        dataset: DatasetVolume instance.
        nodes_df: Polars DataFrame with columns: ['t', 'node_id', 'z', 'y', 'x', 'score'].
        device: Device to execute computation on.

    Returns:
        nodes_df: Original DataFrame with sorted ordering aligned with embeddings.
        embeddings: (N, embedding_dim) PyTorch FloatTensor of extracted node representations.
    """
    model.eval()
    embedder.eval()

    if nodes_df.height == 0:
        return nodes_df, torch.empty((0, embedder.embedding_dim), dtype=torch.float32)

    # Sort DataFrame by timepoint t for sequential volume reading
    sorted_df = nodes_df.sort("t")
    timepoints = sorted(sorted_df["t"].unique().to_list())

    all_embeddings_list: List[torch.Tensor] = []

    with torch.no_grad():
        for t in timepoints:
            t_nodes = sorted_df.filter(pl.col("t") == t)
            n_cells = t_nodes.height

            if n_cells == 0:
                continue

            # Read raw volume crop at time t
            vol_crop = dataset.read_spatial_crop(
                slice(t, t + 1),
                slice(None),
                slice(None),
                slice(None),
            )  # (1, Z, Y, X)

            vol_norm = normalize_intensity(vol_crop, quantiles=dataset.quantiles)
            inp = torch.from_numpy(vol_norm).unsqueeze(0).to(device)  # (1, 1, Z, Y, X)

            # Forward pass through detector backbone to get dense feature map
            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                _, feat_map = model(inp)  # feat_map: (1, C, Z, Y, X)

            # Extract coordinates and scores
            coords_zyx_np = t_nodes.select(["z", "y", "x"]).to_numpy().astype(np.float32)
            scores_np = t_nodes.select(["score"]).to_numpy().astype(np.float32)

            coords_tzyx_np = np.hstack([np.full((n_cells, 1), float(t), dtype=np.float32), coords_zyx_np])
            intensity_stats_np = extract_patch_intensity_stats(vol_norm[0], coords_zyx_np)

            coords_zyx_torch = torch.from_numpy(coords_zyx_np).to(device)
            coords_tzyx_torch = torch.from_numpy(coords_tzyx_np).to(device)
            scores_torch = torch.from_numpy(scores_np).to(device)
            intensity_stats_torch = torch.from_numpy(intensity_stats_np).to(device)

            # Pool RoI visual features from feat_map
            pooled_vis = embedder.roi_pooler(feat_map, coords_zyx_torch)

            # Fuse into unified embeddings
            node_embs = embedder(
                visual_features=pooled_vis,
                coords_tzyx=coords_tzyx_torch,
                scores=scores_torch,
                intensity_stats=intensity_stats_torch,
            )

            all_embeddings_list.append(node_embs.cpu())

    all_embeddings = torch.cat(all_embeddings_list, dim=0) if all_embeddings_list else torch.empty((0, embedder.embedding_dim))
    return sorted_df, all_embeddings

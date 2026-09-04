"""Spatio-temporal Transformer tracker for cell association and mitosis prediction."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.zarr_reader import DEFAULT_SCALE
from src.representation.positional_encoding import RelativeSpatialEncoding
from src.tracking.candidate_edges import CandidateEdgeBatch


@dataclass
class TrackerPrediction:
    """Outputs from the Spatio-Temporal Tracker."""

    edge_logits: torch.Tensor  # (E,) raw transition logits
    edge_probs: torch.Tensor  # (E,) transition probabilities in [0, 1]
    division_logits: torch.Tensor  # (N_src,) raw mitosis logits
    division_probs: torch.Tensor  # (N_src,) mitosis probabilities in [0, 1]
    edge_features: torch.Tensor  # (E, hidden_dim) contextualized edge embeddings


class CompetingEdgeAttentionLayer(nn.Module):
    """Contextualizes candidate edges through bi-directional competition (source-outgoing & target-incoming)."""

    def __init__(self, edge_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.edge_dim = edge_dim
        self.num_heads = num_heads

        # Projections for competitive context
        self.out_proj = nn.Linear(edge_dim, edge_dim)
        self.in_proj = nn.Linear(edge_dim, edge_dim)

        self.update_mlp = nn.Sequential(
            nn.Linear(edge_dim * 3, edge_dim * 2),
            nn.LayerNorm(edge_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(edge_dim * 2, edge_dim),
        )
        self.norm = nn.LayerNorm(edge_dim)

    def forward(
        self,
        edge_feats: torch.Tensor,
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
        n_src: int,
        n_dst: int,
    ) -> torch.Tensor:
        """
        Args:
            edge_feats: (E, edge_dim) current edge embeddings.
            src_indices: (E,) indices of source nodes.
            dst_indices: (E,) indices of destination nodes.
            n_src: Number of source nodes.
            n_dst: Number of target nodes.

        Returns:
            updated_edge_feats: (E, edge_dim).
        """
        E = edge_feats.shape[0]
        if E == 0:
            return edge_feats

        device = edge_feats.device

        # Outgoing competitive pooling (grouped by src node)
        # We aggregate competing edge vectors from the same source cell u
        proj_out = self.out_proj(edge_feats)
        src_agg = torch.zeros((n_src, self.edge_dim), device=device, dtype=edge_feats.dtype)
        src_counts = torch.zeros((n_src, 1), device=device, dtype=edge_feats.dtype)
        src_agg.index_add_(0, src_indices, proj_out)
        src_counts.index_add_(0, src_indices, torch.ones((E, 1), device=device, dtype=edge_feats.dtype))
        src_mean = src_agg / src_counts.clamp(min=1.0)
        # Broadcast back to edges
        competing_out = src_mean[src_indices]  # (E, edge_dim)

        # Incoming competitive pooling (grouped by dst node)
        # We aggregate competing edge vectors targeting the same destination cell v
        proj_in = self.in_proj(edge_feats)
        dst_agg = torch.zeros((n_dst, self.edge_dim), device=device, dtype=edge_feats.dtype)
        dst_counts = torch.zeros((n_dst, 1), device=device, dtype=edge_feats.dtype)
        dst_agg.index_add_(0, dst_indices, proj_in)
        dst_counts.index_add_(0, dst_indices, torch.ones((E, 1), device=device, dtype=edge_feats.dtype))
        dst_mean = dst_agg / dst_counts.clamp(min=1.0)
        # Broadcast back to edges
        competing_in = dst_mean[dst_indices]  # (E, edge_dim)

        # Update edge representation with competing context
        cat_context = torch.cat([edge_feats, competing_out, competing_in], dim=-1)
        residual = self.update_mlp(cat_context)
        return self.norm(edge_feats + residual)


class SpatioTemporalTracker(nn.Module):
    """End-to-end Deep Learning Spatio-Temporal Cell Tracker.
    
    Associates cell node representations across consecutive timepoints via competitive graph attention,
    predicting continuation probabilities and mitosis events simultaneously.
    """

    def __init__(
        self,
        node_dim: int = 128,
        rel_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        scale: Tuple[float, float, float] = DEFAULT_SCALE,
        dropout: float = 0.1,
    ):
        """
        Args:
            node_dim: Dimensionality of cell node embeddings.
            rel_dim: Dimensionality of relative displacement Fourier embeddings.
            hidden_dim: Hidden dimension for tracker layers.
            num_layers: Number of competitive attention layers.
            num_heads: Number of attention heads.
            scale: Physical scale (s_z, s_y, s_x) in micrometers per voxel.
            dropout: Dropout rate.
        """
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.scale = scale

        # Relative spatial Fourier encoder
        self.rel_encoder = RelativeSpatialEncoding(
            num_frequency_bands=6,
            scale=scale,
            output_dim=rel_dim,
        )

        # Initial edge projection: [e_u (node_dim) || e_v (node_dim) || r_uv (rel_dim) || dist_um (1)]
        in_edge_dim = node_dim * 2 + rel_dim + 1
        self.edge_init = nn.Sequential(
            nn.Linear(in_edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Competing edge attention layers
        self.layers = nn.ModuleList(
            [CompetingEdgeAttentionLayer(edge_dim=hidden_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )

        # Prediction Heads:
        # 1. Edge Transition Classifier
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # 2. Source Cell Division (Mitosis) Classifier
        self.division_head = nn.Sequential(
            nn.Linear(node_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        src_embeddings: torch.Tensor,
        dst_embeddings: torch.Tensor,
        candidate_batch: CandidateEdgeBatch,
    ) -> TrackerPrediction:
        """Forward pass predicting edge transition and mitosis probabilities.

        Args:
            src_embeddings: (N_src, node_dim) embeddings of cells at frame t.
            dst_embeddings: (N_dst, node_dim) embeddings of cells at frame t+1.
            candidate_batch: CandidateEdgeBatch with src/dst indices, displacements, and distances.

        Returns:
            TrackerPrediction dataclass with edge and division probabilities.
        """
        n_src = src_embeddings.shape[0]
        n_dst = dst_embeddings.shape[0]
        E = candidate_batch.src_indices.shape[0]

        device = src_embeddings.device

        # Mitosis prediction for all source cells: (N_src,)
        if n_src > 0:
            div_logits = self.division_head(src_embeddings).squeeze(-1)  # (N_src,)
            div_probs = torch.sigmoid(div_logits)
        else:
            div_logits = torch.empty((0,), device=device, dtype=torch.float32)
            div_probs = torch.empty((0,), device=device, dtype=torch.float32)

        if E == 0:
            return TrackerPrediction(
                edge_logits=torch.empty((0,), device=device, dtype=torch.float32),
                edge_probs=torch.empty((0,), device=device, dtype=torch.float32),
                division_logits=div_logits,
                division_probs=div_probs,
                edge_features=torch.empty((0, self.hidden_dim), device=device, dtype=torch.float32),
            )

        src_idx = candidate_batch.src_indices.to(device)
        dst_idx = candidate_batch.dst_indices.to(device)
        deltas = candidate_batch.delta_zyx.to(device)
        dists = candidate_batch.distances_um.to(device)

        # Gather node embeddings for edges
        e_u = src_embeddings[src_idx]  # (E, node_dim)
        e_v = dst_embeddings[dst_idx]  # (E, node_dim)

        # Encode relative spatial displacement
        r_uv = self.rel_encoder(deltas)  # (E, rel_dim)

        # Initial edge features
        edge_input = torch.cat([e_u, e_v, r_uv, dists], dim=-1)  # (E, 2*node_dim + rel_dim + 1)
        h_edge = self.edge_init(edge_input)  # (E, hidden_dim)

        # Contextualize edges through competing attention layers
        for layer in self.layers:
            h_edge = layer(h_edge, src_idx, dst_idx, n_src=n_src, n_dst=n_dst)

        # Transition edge logits: (E,)
        edge_logits = self.edge_head(h_edge).squeeze(-1)  # (E,)
        edge_probs = torch.sigmoid(edge_logits)

        return TrackerPrediction(
            edge_logits=edge_logits,
            edge_probs=edge_probs,
            division_logits=div_logits,
            division_probs=div_probs,
            edge_features=h_edge,
        )

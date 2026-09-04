"""Temporal self-attention module across volumetric time frames."""

import torch
import torch.nn as nn


class TemporalCrossAttention(nn.Module):
    """Applies multi-head self-attention across the temporal dimension of a 3D feature volume.

    Allows bottleneck features to exchange temporal context across consecutive frames,
    improving detection of faint nuclei and dividing cells.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        """
        Args:
            channels: Feature dimension C at the bottleneck.
            num_heads: Number of attention heads (must divide channels).
            mlp_ratio: Expansion factor for MLP feedforward layer.
            dropout: Dropout probability.
        """
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(channels)
        mlp_hidden_dim = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, B: int, W: int) -> torch.Tensor:
        """
        Args:
            x: Flattened spatio-temporal tensor of shape (B * W, C, Z, Y, X)
               or 6D tensor of shape (B, W, C, Z, Y, X).
            B: Batch size.
            W: Temporal window size.

        Returns:
            Tensor of the same shape as input with temporal attention applied.
        """
        is_6d = x.dim() == 6
        if is_6d:
            # (B, W, C, Z, Y, X)
            _, _, C, Z, Y, X = x.shape
            x_6d = x
        else:
            # (B * W, C, Z, Y, X)
            BW, C, Z, Y, X = x.shape
            assert BW == B * W, f"Expected B*W={B*W}, got {BW}"
            x_6d = x.view(B, W, C, Z, Y, X)

        if W <= 1:
            # No temporal interaction possible with a single frame
            return x

        # Reshape to (B * Z * Y * X, W, C) for parallel attention across time
        # Permute: (B, Z, Y, X, W, C) -> flatten spatial batch
        feat = x_6d.permute(0, 3, 4, 5, 1, 2).contiguous()
        feat_flat = feat.view(B * Z * Y * X, W, C)

        # 1. Multihead Attention block
        norm_feat = self.norm1(feat_flat)
        attn_out, _ = self.attn(norm_feat, norm_feat, norm_feat)
        feat_flat = feat_flat + attn_out

        # 2. Feedforward MLP block
        norm_feat2 = self.norm2(feat_flat)
        mlp_out = self.mlp(norm_feat2)
        feat_flat = feat_flat + mlp_out

        # Reshape back to original representation
        out_6d = feat_flat.view(B, Z, Y, X, W, C).permute(0, 4, 5, 1, 2, 3).contiguous()

        if is_6d:
            return out_6d
        else:
            return out_6d.view(B * W, C, Z, Y, X)

"""3D Temporal U-Net for volumetric cell detection and feature extraction."""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.detection.temporal_attention import TemporalCrossAttention


class ConvBlock3D(nn.Module):
    """Residual 3D Convolutional Block with InstanceNorm3d and LeakyReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_ch, affine=True)
        self.act1 = nn.LeakyReLU(0.01, inplace=True)

        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_ch, affine=True)
        self.act2 = nn.LeakyReLU(0.01, inplace=True)

        self.res = (
            nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.res(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act2(out + res)


class UpBlock3D(nn.Module):
    """Trilinear upsampling block with skip-connection concatenation."""

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        scale_factor: Tuple[int, int, int] = (1, 2, 2),
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.reduce = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
        self.conv = ConvBlock3D(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x, scale_factor=self.scale_factor, mode="trilinear", align_corners=False
        )
        x = self.reduce(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="trilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class TemporalUNet3D(nn.Module):
    """Anisotropic 3D U-Net with bottleneck cross-frame temporal self-attention.

    Predicts:
        1. Heatmap: continuous cell presence probability (B, W, 1, Z, Y, X) in [0, 1].
        2. Features: dense visual embeddings (B, W, C_feat, Z, Y, X) for spatial-temporal tracking.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        feature_dim: int = 32,
        use_temporal_attention: bool = True,
        num_attention_heads: int = 4,
        out_activation: str = "sigmoid",
    ):
        """
        Args:
            in_channels: Input channel dimension (1 for single-channel fluorescence).
            base_channels: Number of filters in first stage (16 or 24 recommended for 4GB VRAM).
            feature_dim: Channel dimension for tracking visual embeddings.
            use_temporal_attention: Whether to use cross-time attention at the bottleneck.
            num_attention_heads: Multi-head attention heads at bottleneck.
            out_activation: 'sigmoid' or 'none' (logits).
        """
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.feature_dim = feature_dim
        self.use_temporal_attention = use_temporal_attention
        self.out_activation = out_activation

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        # Spatial Encoder (anisotropic downsampling: (1, 2, 2) in early stages)
        self.inc = ConvBlock3D(in_channels, c1)

        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.down1 = ConvBlock3D(c1, c2)

        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.down2 = ConvBlock3D(c2, c3)

        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.down3 = ConvBlock3D(c3, c4)

        # Bottleneck Temporal Attention
        if self.use_temporal_attention:
            self.temporal_attn = TemporalCrossAttention(
                channels=c4,
                num_heads=num_attention_heads,
            )
        else:
            self.temporal_attn = None

        # Spatial Decoder
        self.up1 = UpBlock3D(c4, c3, c3, scale_factor=(2, 2, 2))
        self.up2 = UpBlock3D(c3, c2, c2, scale_factor=(1, 2, 2))
        self.up3 = UpBlock3D(c2, c1, c1, scale_factor=(1, 2, 2))

        # Prediction Heads
        self.heatmap_head = nn.Conv3d(c1, 1, kernel_size=1)
        self.feature_head = nn.Conv3d(c1, feature_dim, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor. Supported shapes:
               - (B, 1, W, Z, Y, X) [standard DataLoader layout from PatchSampler]
               - (B, W, 1, Z, Y, X)
               - (B, 1, Z, Y, X) [single frame]

        Returns:
            heatmap: (B, W, 1, Z, Y, X) or (B, 1, Z, Y, X) probability volume.
            features: (B, W, C_feat, Z, Y, X) or (B, C_feat, Z, Y, X) dense embeddings.
        """
        is_5d = x.dim() == 5
        if is_5d:
            # Single frame: (B, C, Z, Y, X)
            B, C, Z, Y, X = x.shape
            W = 1
            x_in = x
        elif x.dim() == 6:
            # Check if shape is (B, 1, W, Z, Y, X) or (B, W, 1, Z, Y, X)
            if x.shape[1] == self.in_channels:
                # (B, C, W, Z, Y, X) -> permute to (B, W, C, Z, Y, X)
                B, C, W, Z, Y, X = x.shape
                x = x.permute(0, 2, 1, 3, 4, 5).contiguous()
            else:
                B, W, C, Z, Y, X = x.shape

            # Flatten (B, W) into batch for 3D CNN: (B * W, C, Z, Y, X)
            x_in = x.view(B * W, C, Z, Y, X)
        else:
            raise ValueError(f"Expected 5D or 6D tensor, got shape {x.shape}")

        # Encoder forward pass
        x1 = self.inc(x_in)
        x2 = self.down1(self.pool1(x1))
        x3 = self.down2(self.pool2(x2))
        x4 = self.down3(self.pool3(x3))

        # Bottleneck temporal attention
        if self.temporal_attn is not None and W > 1:
            x4 = self.temporal_attn(x4, B=B, W=W)

        # Decoder forward pass
        u1 = self.up1(x4, x3)
        u2 = self.up2(u1, x2)
        u3 = self.up3(u2, x1)

        # Heads
        heatmap = self.heatmap_head(u3)
        if self.out_activation == "sigmoid":
            heatmap = torch.sigmoid(heatmap)

        features = self.feature_head(u3)

        if is_5d:
            return heatmap, features
        else:
            # Reshape back to (B, W, 1, Z, Y, X) and (B, W, C_feat, Z, Y, X)
            hm_out = heatmap.view(B, W, 1, Z, Y, X)
            feat_out = features.view(B, W, self.feature_dim, Z, Y, X)
            return hm_out, feat_out

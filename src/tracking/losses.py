"""Loss functions for Spatio-Temporal Transformer tracking and mitosis prediction."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrackerLoss(nn.Module):
    """Composite loss function for edge association and division classification with class balancing."""

    def __init__(
        self,
        edge_gamma: float = 2.0,
        edge_alpha: float = 0.75,  # Higher weight for positive edge transitions
        div_gamma: float = 2.0,
        div_alpha: float = 0.85,   # Higher weight for rare mitosis events
        div_weight: float = 1.0,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.edge_gamma = edge_gamma
        self.edge_alpha = edge_alpha
        self.div_gamma = div_gamma
        self.div_alpha = div_alpha
        self.div_weight = div_weight
        self.eps = eps

    def binary_focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        alpha: float,
        gamma: float,
    ) -> torch.Tensor:
        """Numerically stable focal loss in float32."""
        if logits.numel() == 0:
            return torch.tensor(0.0, device=logits.device, dtype=torch.float32)

        # Cast to float32 for FP16 AMP stability
        logits = logits.float()
        targets = targets.float()

        probs = torch.sigmoid(logits)
        probs_clamped = torch.clamp(probs, min=self.eps, max=1.0 - self.eps)

        pt = targets * probs_clamped + (1.0 - targets) * (1.0 - probs_clamped)
        alpha_t = targets * alpha + (1.0 - targets) * (1.0 - alpha)

        focal_weight = alpha_t * torch.pow(1.0 - pt, gamma)
        bce = F.binary_cross_entropy(probs_clamped, targets, reduction="none")

        loss = focal_weight * bce
        return loss.mean()

    def forward(
        self,
        edge_logits: torch.Tensor,
        edge_targets: Optional[torch.Tensor],
        division_logits: torch.Tensor,
        division_targets: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute combined tracking losses.

        Args:
            edge_logits: (E,) edge continuation logits.
            edge_targets: (E,) binary ground-truth labels.
            division_logits: (N_src,) mitosis logits.
            division_targets: (N_src,) binary ground-truth division labels.

        Returns:
            Dict containing 'loss', 'edge_loss', 'div_loss'.
        """
        device = edge_logits.device
        zero_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

        edge_loss = zero_loss
        if edge_targets is not None and edge_logits.numel() > 0:
            edge_loss = self.binary_focal_loss(
                edge_logits,
                edge_targets,
                alpha=self.edge_alpha,
                gamma=self.edge_gamma,
            )

        div_loss = zero_loss
        if division_targets is not None and division_logits.numel() > 0:
            div_loss = self.binary_focal_loss(
                division_logits,
                division_targets,
                alpha=self.div_alpha,
                gamma=self.div_gamma,
            )

        total_loss = edge_loss + self.div_weight * div_loss

        return {
            "loss": total_loss,
            "edge_loss": edge_loss,
            "div_loss": div_loss,
        }

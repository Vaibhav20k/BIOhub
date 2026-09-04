"""Loss functions for 3D continuous heatmap detection."""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModifiedFocalLoss(nn.Module):
    """Modified Focal Loss for continuous Gaussian heatmap targets (CenterNet / CornerNet).

    Penalizes false negatives heavily at peak locations (y == 1), while reducing penalty
    for false positives that are spatially close to true centroids ((1 - y)^beta factor).
    """

    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 4.0,
        eps: float = 1e-4,
    ):
        """
        Args:
            alpha: Focusing parameter for hard examples.
            beta: Decay factor for distance penalty near ground-truth centers.
            eps: Numerical stability constant (>= 1e-4 for FP16 compatibility).
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = max(eps, 1e-4)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted probabilities in (0, 1) of shape (B, W, 1, Z, Y, X) or (B, 1, Z, Y, X).
            target: Ground truth Gaussian heatmaps in [0, 1] of matching shape.
            mask: Optional boolean or binary mask of valid frames/voxels.

        Returns:
            Scalar loss tensor.
        """
        pred = torch.clamp(pred.float(), self.eps, 1.0 - self.eps)
        target = target.float()

        pos_mask = target.ge(0.5).float()
        neg_mask = target.lt(0.5).float()

        if mask is not None:
            mask = mask.float()
            pos_mask = pos_mask * mask
            neg_mask = neg_mask * mask

        pos_weights = torch.pow(1.0 - pred, self.alpha)
        neg_weights = torch.pow(1.0 - target, self.beta) * torch.pow(pred, self.alpha)

        pos_loss = -pos_weights * torch.log(pred) * pos_mask
        neg_loss = -neg_weights * torch.log(1.0 - pred) * neg_mask

        num_pos = max(1.0, float(pos_mask.sum().item()))
        num_neg = max(1.0, float(neg_mask.sum().item()))

        loss = (pos_loss.sum() / num_pos) + (neg_loss.sum() / num_neg)
        return loss


class SoftDiceLoss(nn.Module):
    """Continuous Soft Dice Loss for volumetric foreground segmentation / heatmaps."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted probabilities in (0, 1).
            target: Ground truth heatmap in [0, 1].
            mask: Optional valid voxel mask.
        """
        pred = pred.float()
        target = target.float()
        if mask is not None:
            mask = mask.float()
            pred = pred * mask
            target = target * mask

        intersection = 2.0 * (pred * target).sum() + self.eps
        union = (pred.pow(2) + target.pow(2)).sum() + self.eps
        return 1.0 - (intersection / union)


class DetectionLoss(nn.Module):
    """Composite loss for 3D cell center heatmap prediction."""

    def __init__(
        self,
        focal_weight: float = 1.0,
        dice_weight: float = 1.0,
        alpha: float = 2.0,
        beta: float = 4.0,
    ):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal_loss = ModifiedFocalLoss(alpha=alpha, beta=beta)
        self.dice_loss = SoftDiceLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred: Predicted heatmap (B, W, 1, Z, Y, X).
            target: Target heatmap (B, W, 1, Z, Y, X).
            mask: Optional frame/voxel mask.

        Returns:
            total_loss: Differentiable scalar loss.
            metrics: Dictionary of component loss values.
        """
        loss_focal = self.focal_loss(pred, target, mask=mask)
        loss_dice = self.dice_loss(pred, target, mask=mask)

        total_loss = (self.focal_weight * loss_focal) + (self.dice_weight * loss_dice)

        metrics = {
            "loss_total": float(total_loss.detach().item()),
            "loss_focal": float(loss_focal.detach().item()),
            "loss_dice": float(loss_dice.detach().item()),
        }

        return total_loss, metrics

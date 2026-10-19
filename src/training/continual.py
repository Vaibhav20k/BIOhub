"""Milestone 2: safe continual fine-tuning of the detector.

- A low learning rate with a one-epoch linear warm-up and cosine decay to a floor (`make_warmup_cosine_scheduler`).
- Feature-manifold preservation: the student's intermediate activations are pulled towards those of a frozen copy of
  the promoted baseline (`FeatureTap`, `FeatureDistillationLoss`, `frozen_copy`), so new data cannot overwrite what
  the baseline already detects:

      L_total = L_focal + L_dice + lambda_distill * || F_student - F_frozen ||^2   (mean over elements)
"""

import copy
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def warmup_cosine_factor(
    step: int,
    warmup_steps: int,
    total_steps: int,
    min_ratio: float,
    start_factor: float = 0.1,
) -> float:
    """LR multiplier: linear from ``start_factor`` to 1 over ``warmup_steps``, then cosine from 1 to ``min_ratio``
    at ``total_steps`` (held at ``min_ratio`` afterwards)."""
    if warmup_steps > 0 and step < warmup_steps:
        return start_factor + (1.0 - start_factor) * step / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / decay_steps)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def make_warmup_cosine_scheduler(
    optimizer: Optimizer,
    base_lr: float,
    min_lr: float,
    steps_per_epoch: int,
    warmup_epochs: int = 1,
    total_epochs: int = 8,
    start_factor: float = 0.1,
) -> LambdaLR:
    """Per-step scheduler: warm up for ``warmup_epochs``, reach ``min_lr`` at the end of ``total_epochs``.
    The optimizer's lr must be set to ``base_lr``; call ``scheduler.step()`` after every optimizer step."""
    if base_lr <= 0 or not 0 <= min_lr <= base_lr:
        raise ValueError(f"Need 0 <= min_lr <= base_lr and base_lr > 0 (got {min_lr}, {base_lr})")
    warmup = warmup_epochs * steps_per_epoch
    total = total_epochs * steps_per_epoch
    ratio = min_lr / base_lr
    return LambdaLR(optimizer, lambda s: warmup_cosine_factor(s, warmup, total, ratio, start_factor))


def frozen_copy(model: nn.Module) -> nn.Module:
    """A detached, eval-mode copy of ``model`` that never receives gradients (the distillation teacher)."""
    teacher = copy.deepcopy(model).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


class FeatureTap:
    """Captures the output of one named submodule (default: the last decoder block, which feeds both the heatmap
    head and the feature head) on every forward pass."""

    def __init__(self, model: nn.Module, layer: str = "up3"):
        modules = dict(model.named_modules())
        if layer not in modules:
            raise ValueError(f"Layer '{layer}' not found in {type(model).__name__}")
        self.output: Optional[torch.Tensor] = None
        self._handle = modules[layer].register_forward_hook(self._hook)

    def _hook(self, module: nn.Module, inputs, output) -> None:
        self.output = output

    def take(self) -> torch.Tensor:
        if self.output is None:
            raise RuntimeError("No activation captured yet; run a forward pass first")
        out, self.output = self.output, None
        return out

    def close(self) -> None:
        self._handle.remove()


class FeatureDistillationLoss(nn.Module):
    """``weight * mean((student - teacher)^2)``; the teacher side is always detached."""

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        if student.shape != teacher.shape:
            raise ValueError(f"Shape mismatch: student {tuple(student.shape)} vs teacher {tuple(teacher.shape)}")
        return self.weight * F.mse_loss(student, teacher.detach())

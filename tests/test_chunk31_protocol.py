"""Milestones 1 and 2: density-normalised patch quotas, warm-up + cosine LR, and feature distillation."""

import pytest
import torch

from src.data.patch_sampler import density_normalized_quotas, domain_totals
from src.detection.unet3d import TemporalUNet3D
from src.training.continual import (
    FeatureDistillationLoss,
    FeatureTap,
    frozen_copy,
    make_warmup_cosine_scheduler,
    warmup_cosine_factor,
)

# Chunk 3 composition: 6bba 3,990 cells over 5 sequences, 44b6 1,080 over 3 (44b6_0c582fdc has 71)
CHUNK3_CELLS = {"6bba_61ecbe65": 850, "6bba_cf35214c": 790, "6bba_b204cac7": 820, "6bba_d0fc38b5": 760,
                "6bba_ebdf3b34": 770, "44b6_0c582fdc": 71, "44b6_c96cfa10": 520, "44b6_996155de": 489}


def test_quotas_sum_and_follow_cell_density():
    quotas = density_normalized_quotas(CHUNK3_CELLS, 800)
    assert sum(quotas.values()) == 800
    assert domain_totals(quotas) == {"6bba": 630, "44b6": 170}  # README Milestone 1: 78.7% / 21.3%
    per_cell = [quotas[s] / CHUNK3_CELLS[s] for s in CHUNK3_CELLS]
    assert max(per_cell) - min(per_cell) < 0.01  # every cell gets about the same exposure


def test_old_50_50_split_is_gone():
    quotas = density_normalized_quotas(CHUNK3_CELLS, 800)
    # under 50/50, the 71-cell sequence got 134 patches (1.89 per cell); now it is proportional
    assert quotas["44b6_0c582fdc"] / 71 < 0.2


def test_quota_floor_and_edge_cases():
    assert density_normalized_quotas({"a_1": 0, "b_1": 1000}, 10, min_per_sequence=2) == {"a_1": 2, "b_1": 8}
    assert sum(density_normalized_quotas({"a_1": 0, "b_1": 0, "c_1": 0}, 10).values()) == 10
    assert density_normalized_quotas({}, 100) == {}
    with pytest.raises(ValueError):
        density_normalized_quotas({"a_1": 1, "b_1": 1}, 1)
    with pytest.raises(ValueError):
        density_normalized_quotas({"a_1": -1}, 10)


def test_quotas_are_deterministic():
    assert density_normalized_quotas(CHUNK3_CELLS, 800) == density_normalized_quotas(dict(reversed(CHUNK3_CELLS.items())), 800)


def test_warmup_cosine_factor_shape():
    warm, total, floor = 100, 800, 1e-5 / 2e-4
    assert warmup_cosine_factor(0, warm, total, floor) == pytest.approx(0.1)
    assert warmup_cosine_factor(warm, warm, total, floor) == pytest.approx(1.0)
    assert warmup_cosine_factor(total, warm, total, floor) == pytest.approx(floor)
    assert warmup_cosine_factor(total + 500, warm, total, floor) == pytest.approx(floor)
    decay = [warmup_cosine_factor(s, warm, total, floor) for s in range(warm, total + 1, 50)]
    assert all(a >= b for a, b in zip(decay, decay[1:]))


def test_scheduler_reaches_base_then_floor():
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([param], lr=2e-4)
    sched = make_warmup_cosine_scheduler(opt, base_lr=2e-4, min_lr=1e-5, steps_per_epoch=10, warmup_epochs=1, total_epochs=8)
    assert opt.param_groups[0]["lr"] == pytest.approx(2e-5)
    lrs = []
    for _ in range(80):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert lrs[9] == pytest.approx(2e-4)  # end of warm-up epoch
    assert lrs[-1] == pytest.approx(1e-5)  # end of epoch 8
    with pytest.raises(ValueError):
        make_warmup_cosine_scheduler(opt, base_lr=1e-4, min_lr=2e-4, steps_per_epoch=10)


def _tiny_detector():
    torch.manual_seed(0)
    return TemporalUNet3D(in_channels=1, base_channels=4, feature_dim=8, use_temporal_attention=False)


def test_distillation_anchors_student_to_frozen_teacher():
    student = _tiny_detector()
    teacher = frozen_copy(student)
    assert not any(p.requires_grad for p in teacher.parameters())
    assert not teacher.training

    tap_s, tap_t = FeatureTap(student, "up3"), FeatureTap(teacher, "up3")
    x = torch.rand(2, 1, 4, 16, 16)
    student.eval()
    with torch.no_grad():
        teacher(x)
        student(x)
    loss = FeatureDistillationLoss(weight=0.1)(tap_s.take(), tap_t.take())
    assert loss.item() == pytest.approx(0.0, abs=1e-10)  # identical weights -> no penalty

    with torch.no_grad():
        for p in student.parameters():
            p.add_(0.05 * torch.randn_like(p))
    student.train()
    with torch.no_grad():
        teacher(x)
    student(x)
    loss = FeatureDistillationLoss(weight=0.1)(tap_s.take(), tap_t.take())
    assert loss.item() > 0
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters())
    assert all(p.grad is None for p in teacher.parameters())
    tap_s.close()
    tap_t.close()


def test_feature_tap_errors():
    model = _tiny_detector()
    with pytest.raises(ValueError):
        FeatureTap(model, "no_such_layer")
    tap = FeatureTap(model, "up3")
    with pytest.raises(RuntimeError):
        tap.take()
    tap.close()
    with pytest.raises(ValueError):
        FeatureDistillationLoss()(torch.zeros(1, 2), torch.zeros(1, 3))

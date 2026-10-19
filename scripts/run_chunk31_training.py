#!/usr/bin/env python3
"""Milestone 5 — Chunk 3.1 controlled retraining under the corrected continual-learning protocol.

Same Chunk 3 sequences and permanent anchors as the failed Chunk 3 run; what changes is how we train:
- Milestone 1: density-normalised quotas (patches ∝ annotated cells, ~630 6bba / ~170 44b6 of 800) and fully
  cell-centred crops, so every annotated cell has the same exposure.
- Milestone 2: AdamW LR 2e-4 (10x lower), wd 1e-4, 1-epoch warm-up then cosine to 1e-5 at epoch 8, plus feature
  distillation (lambda 0.1) on the last decoder block against a frozen copy of the promoted Chunk 2 detector.
- 8 detector epochs, best epoch picked on the anchors, then 3 tracker epochs on top of the chosen detector.

Promotion gate (README Milestone 5): macro >= 0.045, 6bba anchor >= 0.075, 44b6 anchor > 0.015.
Promoted weights are written to the output folder only; the active baseline is never overwritten here.

Runs on Kaggle (paths auto-detected under /kaggle) or locally with --train-dir / --work-dir / --out-dir.
"""

import argparse
import hashlib
import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader

from src.data.patch_sampler import density_normalized_quotas, domain_totals
from src.data.zarr_reader import DatasetVolume, open_dataset
from src.detection.dataset import DetectionDataset
from src.detection.unet3d import TemporalUNet3D
from src.pipeline.anchor_monitor import evaluate_anchor_detection
from src.pipeline.chunk_manager import ChunkStorageManager
from src.pipeline.continual_trainer import ContinualTrainer
from src.pipeline.held_out_evaluator import HeldOutEvaluator
from src.representation.node_embedding import CellNodeEmbedding
from src.tracking.transformer import SpatioTemporalTracker
from src.training.continual import FeatureDistillationLoss, FeatureTap, frozen_copy, make_warmup_cosine_scheduler
from src.training.losses import DetectionLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("chunk31")

CHUNK3_STEMS = ["6bba_61ecbe65", "44b6_0c582fdc", "44b6_c96cfa10", "6bba_cf35214c",
                "6bba_b204cac7", "44b6_996155de", "6bba_d0fc38b5", "6bba_ebdf3b34"]
ANCHOR_44B6, ANCHOR_6BBA = "44b6_66f9292d", "6bba_d82a4fc6"
BASELINE = {"macro_final_score": 0.03865588, "44b6_final_score": 0.0, "6bba_final_score": 0.07731176}
GATE = {"macro_min": 0.045, "6bba_min": 0.075, "44b6_above": 0.015}
DETECTOR_SHA256 = "6e9a64c754551402d45c30b7a6949d04f2d2d7e40a3f076243581037a130eaab"
TRACKER_SHA256 = "ed3a4498461da21dbd7b39943eece75f06b6b0fa8faaf570c3aa93403994baf5"
def _json_default(o):
    return o.item() if hasattr(o, "item") else str(o)


EVAL_KW = dict(detection_threshold=0.25, max_peaks_per_frame=150, max_distance_um=7.0, window_size=1)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def find_train_dir(explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit)
    default = Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/train")
    if default.exists():
        return default
    for root, dirs, _ in os.walk("/kaggle/input"):
        if f"{ANCHOR_44B6}.zarr" in dirs:
            return Path(root)
    raise FileNotFoundError("Competition train folder not found; pass --train-dir")


def gate_verdict(summary: Dict) -> Dict:
    s44 = summary["sequences"][ANCHOR_44B6]["final_score"]
    s6b = summary["sequences"][ANCHOR_6BBA]["final_score"]
    macro = summary["macro_final_score"]
    checks = {
        "macro_ge_0.045": macro >= GATE["macro_min"],
        "6bba_ge_0.075": s6b >= GATE["6bba_min"],
        "44b6_gt_0.015": s44 > GATE["44b6_above"],
        "6bba_drop_le_15pct": (BASELINE["6bba_final_score"] - s6b) / BASELINE["6bba_final_score"] <= 0.15,
    }
    return {"promoted": all(checks.values()), "checks": checks, "macro": macro, "44b6": s44, "6bba": s6b}


def load_models(det_path: Path, trk_path: Path, device: torch.device):
    det_ckpt = torch.load(det_path, map_location=device, weights_only=False)
    cfg = det_ckpt.get("config", {})
    detector = TemporalUNet3D(in_channels=1, base_channels=cfg.get("base_channels", 16),
                              feature_dim=cfg.get("feature_dim", 32), prior_prob=0.05,
                              use_temporal_attention=cfg.get("use_temporal_attention", False)).to(device)
    detector.load_state_dict(det_ckpt["model_state_dict"])
    trk_ckpt = torch.load(trk_path, map_location=device, weights_only=False)
    embedder = CellNodeEmbedding(in_visual_dim=32, embedding_dim=128).to(device)
    if "embedder_state_dict" in trk_ckpt:
        embedder.load_state_dict(trk_ckpt["embedder_state_dict"])
    tracker = SpatioTemporalTracker(node_dim=128, rel_dim=32, hidden_dim=128, num_layers=2).to(device)
    tracker.load_state_dict(trk_ckpt["tracker_state_dict"])
    return detector, embedder, tracker


def full_eval(detector, embedder, tracker, val_datasets, device) -> Dict:
    ev = HeldOutEvaluator(detector=detector, embedder=embedder, tracker=tracker, diagnostic_mode=False,
                          device=device, **EVAL_KW)
    return ev.evaluate_cohort(val_datasets)


def brief(summary: Dict) -> str:
    seqs = summary["sequences"]
    return (f"macro={summary['macro_final_score']:.6f} 44b6={seqs[ANCHOR_44B6]['final_score']:.6f} "
            f"6bba={seqs[ANCHOR_6BBA]['final_score']:.6f} TP/FP/FN={summary['total_node_tp']}/"
            f"{summary['total_node_fp']}/{summary['total_node_fn']}")


def train_detector(detector, train_datasets, val_datasets, quotas, args, device, out_ckpt: Path) -> List[Dict]:
    patch = (32, 128, 128)
    subsets = [DetectionDataset(dataset=ds, patch_size=(min(patch[0], ds.shape[1]), min(patch[1], ds.shape[2]),
                                                        min(patch[2], ds.shape[3])),
                                window_size=1, samples_per_epoch=quotas[ds.name], center_on_cell_prob=1.0,
                                augment=True, seed=args.seed + i)
               for i, ds in enumerate(train_datasets)]
    loader = DataLoader(ConcatDataset(subsets), batch_size=args.batch_size, shuffle=True, num_workers=0,
                        pin_memory=device.type == "cuda")

    teacher = frozen_copy(detector)
    tap_student, tap_teacher = FeatureTap(detector, args.distill_layer), FeatureTap(teacher, args.distill_layer)
    criterion = DetectionLoss(focal_weight=1.0, dice_weight=1.0)
    distill = FeatureDistillationLoss(weight=args.distill_weight)
    optimizer = AdamW(detector.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_warmup_cosine_scheduler(optimizer, base_lr=args.lr, min_lr=args.min_lr,
                                             steps_per_epoch=len(loader), warmup_epochs=1, total_epochs=args.epochs)

    logs = []
    for epoch in range(1, args.epochs + 1):
        detector.train()
        t0, sums, steps = time.time(), {"det": 0.0, "distill": 0.0}, 0
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.no_grad():
                teacher(images)
            teacher_feat = tap_teacher.take()
            preds, _ = detector(images)
            det_loss, _ = criterion(preds, batch["heatmap"].to(device), mask=batch["frame_mask"].to(device))
            d_loss = distill(tap_student.take(), teacher_feat)
            optimizer.zero_grad(set_to_none=True)
            (det_loss + d_loss).backward()
            torch.nn.utils.clip_grad_norm_(detector.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            sums["det"] += float(det_loss.item())
            sums["distill"] += float(d_loss.item())
            steps += 1

        val, macro_j = evaluate_anchor_detection(detector, val_datasets, device, detection_threshold=0.25,
                                                 max_peaks_per_frame=150)
        entry = {"epoch": epoch, "det_loss": sums["det"] / max(1, steps), "distill_loss": sums["distill"] / max(1, steps),
                 "lr_end": scheduler.get_last_lr()[0], "seconds": time.time() - t0, "macro_j_adj": macro_j,
                 ANCHOR_44B6: val[ANCHOR_44B6], ANCHOR_6BBA: val[ANCHOR_6BBA]}
        logs.append(entry)
        torch.save({"epoch": epoch, "model_state_dict": detector.state_dict(), "macro_j_adj": macro_j,
                    "config": {"base_channels": 16, "feature_dim": 32, "use_temporal_attention": False}},
                   out_ckpt / f"c31_epoch_{epoch:02d}_detector.pt")
        a, b = val[ANCHOR_44B6], val[ANCHOR_6BBA]
        log.info(f"Ep {epoch}/{args.epochs} det={entry['det_loss']:.4f} distill={entry['distill_loss']:.5f} "
                 f"lr={entry['lr_end']:.2e} | 44b6 rec={a['recall']:.3f} J={a['j_adj']:.4f} gt_mean={a['gt_mean']:.3f} | "
                 f"6bba rec={b['recall']:.3f} J={b['j_adj']:.4f} gt_mean={b['gt_mean']:.3f} | macro J={macro_j:.4f} "
                 f"({entry['seconds']:.0f}s)")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    tap_student.close()
    tap_teacher.close()
    return logs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--work-dir", default=".", help="repo root holding checkpoints/best_*.pt")
    p.add_argument("--train-dir", default=None, help="competition train folder with <stem>.zarr / <stem>.geff")
    p.add_argument("--out-dir", default="/kaggle/working" if Path("/kaggle/working").exists() else "outputs/chunk31")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--tracker-epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--tracker-lr", type=float, default=3e-4)
    p.add_argument("--distill-weight", type=float, default=0.1)
    p.add_argument("--distill-layer", default="up3")
    p.add_argument("--patches", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    work, out = Path(args.work_dir).resolve(), Path(args.out_dir).resolve()
    out_ckpt = out / "checkpoints"
    out_ckpt.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # 1. Pre-flight: baseline hashes, data integrity
    det_path, trk_path = work / "checkpoints/best_detector.pt", work / "checkpoints/best_tracker.pt"
    hashes = {"detector": sha256(det_path), "tracker": sha256(trk_path)}
    assert hashes["detector"] == DETECTOR_SHA256, f"Detector is not the promoted Chunk 2 baseline: {hashes['detector']}"
    assert hashes["tracker"] == TRACKER_SHA256, f"Tracker is not the promoted Chunk 2 baseline: {hashes['tracker']}"
    train_dir = find_train_dir(args.train_dir)
    storage = ChunkStorageManager(staging_dir=out / "staging", min_free_disk_gb=2.0)

    def open_checked(stem: str) -> DatasetVolume:
        ok, err = storage.verify_sequence_integrity(train_dir / f"{stem}.zarr", require_tracks=True)
        assert ok, f"{stem} failed integrity check: {err}"
        return open_dataset(train_dir / f"{stem}.zarr", load_tracks=True, require_tracks=True)

    val_datasets = [open_checked(ANCHOR_44B6), open_checked(ANCHOR_6BBA)]
    train_datasets = [open_checked(s) for s in CHUNK3_STEMS]
    node_counts = {ds.name: int(ds.tracks.num_nodes()) for ds in train_datasets}
    log.info(f"Train dir {train_dir}; cells per sequence: {node_counts}")

    # 2. Milestone 1 quotas
    quotas = density_normalized_quotas(node_counts, args.patches)
    assert sum(quotas.values()) == args.patches
    log.info(f"Density-normalised quotas: {quotas} -> domains {domain_totals(quotas)}; patches per cell "
             f"{ {k: round(quotas[k] / max(1, node_counts[k]), 3) for k in quotas} }")

    # 3. Reproduce the baseline in this environment
    detector, embedder, tracker = load_models(det_path, trk_path, device)
    baseline_summary = full_eval(detector, embedder, tracker, val_datasets, device)
    log.info(f"Baseline reproduction: {brief(baseline_summary)} (expected macro {BASELINE['macro_final_score']})")

    # 4. Detector: 8 epochs, Milestone 2 schedule + distillation
    epoch_logs = train_detector(detector, train_datasets, val_datasets, quotas, args, device, out_ckpt)

    # 5. Pick the detector: best anchor epoch and the last epoch, each through the full lineage with the C2 tracker
    best_ep = max(epoch_logs, key=lambda e: e["macro_j_adj"])["epoch"]
    candidates = sorted({best_ep, args.epochs})
    det_eval: Dict[int, Dict] = {}
    for ep in candidates:
        state = torch.load(out_ckpt / f"c31_epoch_{ep:02d}_detector.pt", map_location=device, weights_only=False)
        detector.load_state_dict(state["model_state_dict"])
        det_eval[ep] = full_eval(detector, embedder, tracker, val_datasets, device)
        log.info(f"Epoch {ep} detector + C2 tracker: {brief(det_eval[ep])}")
    chosen_ep = max(candidates, key=lambda ep: det_eval[ep]["macro_final_score"])
    state = torch.load(out_ckpt / f"c31_epoch_{chosen_ep:02d}_detector.pt", map_location=device, weights_only=False)
    detector.load_state_dict(state["model_state_dict"])
    log.info(f"Chosen detector: epoch {chosen_ep}")

    # 6. Tracker: 3 epochs on top of the chosen detector, kept only if it helps
    c2_tracker_state = {k: v.clone() for k, v in tracker.state_dict().items()}
    trainer = ContinualTrainer(detector=detector, embedder=embedder, tracker=tracker, device=device,
                               tracker_lr=args.tracker_lr)
    t_trk = time.time()
    trk_loss = trainer.train_tracker_on_chunk(datasets=train_datasets, epochs=args.tracker_epochs, max_distance_um=7.0)
    trk_seconds = time.time() - t_trk
    tuned_summary = full_eval(detector, embedder, tracker, val_datasets, device)
    log.info(f"Chosen detector + tuned tracker: {brief(tuned_summary)} (tracker loss {trk_loss:.5f})")
    use_tuned = tuned_summary["macro_final_score"] >= det_eval[chosen_ep]["macro_final_score"]
    if not use_tuned:
        tracker.load_state_dict(c2_tracker_state)
    final_summary = tuned_summary if use_tuned else det_eval[chosen_ep]

    cand_det, cand_trk = out_ckpt / "chunk31_candidate_detector.pt", out_ckpt / "chunk31_candidate_tracker.pt"
    trainer.save_checkpoints(detector_path=cand_det, tracker_path=cand_trk)

    # 7. Promotion gate
    verdict = gate_verdict(final_summary)
    if verdict["promoted"]:
        trainer.save_checkpoints(detector_path=out_ckpt / "promoted_best_detector.pt",
                                 tracker_path=out_ckpt / "promoted_best_tracker.pt")
    log.info(f"GATE {'PASSED -> PROMOTE' if verdict['promoted'] else 'FAILED -> KEEP CHUNK 2 BASELINE'}: {verdict}")

    # 8. Results and registry entry
    now = datetime.now(timezone.utc).isoformat()
    provenance = {"base_detector_sha256": hashes["detector"], "base_tracker_sha256": hashes["tracker"],
                  "candidate_detector_sha256": sha256(cand_det), "candidate_tracker_sha256": sha256(cand_trk)}
    config = {"detector": {"architecture": "TemporalUNet3D", "base_channels": 16, "feature_dim": 32,
                           "optimizer": "AdamW", "lr": args.lr, "min_lr": args.min_lr, "weight_decay": args.weight_decay,
                           "schedule": "1-epoch linear warm-up (x0.1 start) + cosine", "epochs": args.epochs,
                           "batch_size": args.batch_size, "patch_size": [32, 128, 128], "loss": "Focal + Dice (1.0/1.0)",
                           "distillation": {"weight": args.distill_weight, "layer": args.distill_layer,
                                            "teacher": "frozen Chunk 2 detector"},
                           "sampler": "density_normalized_quotas + center_on_cell_prob=1.0", "quotas": quotas,
                           "domain_split": domain_totals(quotas), "chosen_epoch": chosen_ep},
              "tracker": {"architecture": "SpatioTemporalTracker", "lr": args.tracker_lr, "epochs": args.tracker_epochs,
                          "kept_tuned_tracker": use_tuned},
              "provenance": provenance, "seed": args.seed}
    results = {"timestamp": now, "chunk_id": "chunk_003_1_train", "sequences": CHUNK3_STEMS, "node_counts": node_counts,
               "baseline_reproduction": baseline_summary, "epoch_logs": epoch_logs,
               "detector_candidates": {str(k): v for k, v in det_eval.items()}, "tuned_tracker_summary": tuned_summary,
               "final_summary": final_summary, "gate": GATE, "verdict": verdict, "config": config,
               "tracker_seconds": trk_seconds, "total_seconds": time.time() - started}
    (out / "chunk31_results.json").write_text(json.dumps(results, indent=2, default=_json_default))
    registry = {"chunk_id": "chunk_003_1_train", "sequence_stems": CHUNK3_STEMS, "embryo_group": "mixed",
                "chunk_size_bytes": 3280000000, "role": "train", "base_checkpoint": "checkpoints/best_detector.pt",
                "resulting_checkpoint": "checkpoints/chunk31/chunk31_candidate_detector.pt",
                "epochs_trained": args.epochs + args.tracker_epochs,
                "train_loss_detector": epoch_logs[-1]["det_loss"], "train_loss_tracker": trk_loss,
                "val_sequences_evaluated": [ANCHOR_44B6, ANCHOR_6BBA],
                "val_node_recall": final_summary.get("macro_node_recall"),
                "val_node_precision": final_summary.get("macro_node_precision"), "val_mean_dist_um": None,
                "val_edge_jaccard": final_summary.get("macro_edge_jaccard"),
                "val_div_jaccard": final_summary.get("macro_division_jaccard"),
                "val_adj_jaccard": final_summary.get("macro_adjusted_jaccard"),
                "val_final_score": final_summary["macro_final_score"], "training_config": config,
                "inference_time_sec": final_summary.get("inference_time_sec"),
                "ilp_runtime_sec": final_summary.get("ilp_runtime_sec"), "val_node_tp": final_summary["total_node_tp"],
                "val_node_fp": final_summary["total_node_fp"], "val_node_fn": final_summary["total_node_fn"],
                "promoted_to_best": verdict["promoted"], "status": "PROCESSED", "disk_freed_bytes": 0,
                "cleanup_verified": True, "duration_sec": time.time() - started, "error_message": None,
                "created_at": now, "completed_at": now}
    (out / "chunk31_registry_entry.json").write_text(json.dumps(registry, default=_json_default))
    log.info(f"Done in {(time.time() - started) / 60:.1f} min. Final: {brief(final_summary)}. Results in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

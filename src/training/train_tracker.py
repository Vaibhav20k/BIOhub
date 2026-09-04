#!/usr/bin/env python3
"""CLI training pipeline for Spatio-Temporal Transformer Tracker."""

import argparse
import logging
from pathlib import Path
import sys
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.zarr_reader import open_dataset
from src.detection.unet3d import TemporalUNet3D
from src.representation.node_embedding import CellNodeEmbedding
from src.tracking.dataset import TrackingPairDataset
from src.tracking.losses import TrackerLoss
from src.tracking.transformer import SpatioTemporalTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_epoch(
    tracker: SpatioTemporalTracker,
    dataset: TrackingPairDataset,
    criterion: TrackerLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Execute one training epoch across all consecutive frame pairs."""
    tracker.train()

    total_loss = 0.0
    total_edge_loss = 0.0
    total_div_loss = 0.0
    total_edges = 0
    correct_edges = 0
    total_divs = 0
    correct_divs = 0

    indices = torch.randperm(len(dataset)).tolist()

    for idx in indices:
        src_embs, dst_embs, cand_batch = dataset[idx]

        src_embs = src_embs.to(device)
        dst_embs = dst_embs.to(device)

        optimizer.zero_grad()

        pred = tracker(src_embs, dst_embs, cand_batch)

        edge_targets = cand_batch.edge_labels.to(device) if cand_batch.edge_labels is not None else None
        div_targets = cand_batch.division_labels.to(device) if cand_batch.division_labels is not None else None

        loss_dict = criterion(
            edge_logits=pred.edge_logits,
            edge_targets=edge_targets,
            division_logits=pred.division_logits,
            division_targets=div_targets,
        )

        loss = loss_dict["loss"]

        if loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tracker.parameters(), max_norm=2.0)
            optimizer.step()

        total_loss += float(loss.item())
        total_edge_loss += float(loss_dict["edge_loss"].item())
        total_div_loss += float(loss_dict["div_loss"].item())

        # Track classification accuracies
        if edge_targets is not None and edge_targets.numel() > 0:
            pred_binary = (pred.edge_probs >= 0.5).float()
            correct_edges += int((pred_binary == edge_targets).sum().item())
            total_edges += edge_targets.numel()

        if div_targets is not None and div_targets.numel() > 0:
            pred_div_bin = (pred.division_probs >= 0.5).float()
            correct_divs += int((pred_div_bin == div_targets).sum().item())
            total_divs += div_targets.numel()

    n = max(1, len(dataset))
    return {
        "loss": total_loss / n,
        "edge_loss": total_edge_loss / n,
        "div_loss": total_div_loss / n,
        "edge_acc": (correct_edges / max(1, total_edges)) if total_edges > 0 else 0.0,
        "div_acc": (correct_divs / max(1, total_divs)) if total_divs > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Spatio-Temporal Transformer Tracker.")
    parser.add_argument("--train-dataset", type=str, default="44b6_0b24845f", help="Dataset name or path")
    parser.add_argument("--detector-checkpoint", type=str, default="checkpoints/best_detector.pt", help="Detector checkpoint")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--max-distance", type=float, default=7.0, help="Candidate matching radius in micrometers")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Tracker hidden dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of competitive attention layers")
    parser.add_argument("--output-checkpoint", type=str, default="checkpoints/best_tracker.pt", help="Path to save best tracker")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    # Load dataset
    ds_path = Path(args.train_dataset)
    if not (ds_path.parent / f"{ds_path.name}.zarr").exists():
        if (Path("data/train") / f"{args.train_dataset}.zarr").exists():
            ds_path = Path("data/train") / args.train_dataset
        elif (Path("data/fixtures") / f"{args.train_dataset}.zarr").exists():
            ds_path = Path("data/fixtures") / args.train_dataset

    dataset = open_dataset(ds_path, load_tracks=True, require_tracks=True)
    logger.info(f"Loaded dataset {dataset.name} | Shape: {dataset.shape} | GT Nodes: {dataset.tracks.num_nodes()}")

    # Load trained detector model if checkpoint exists
    detector_model = None
    ckpt_path = Path(args.detector_checkpoint)
    if ckpt_path.exists():
        logger.info(f"Loading detector weights from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        detector_model = TemporalUNet3D(
            in_channels=1,
            base_channels=ckpt.get("config", {}).get("base_channels", 16),
            feature_dim=ckpt.get("config", {}).get("feature_dim", 32),
            use_temporal_attention=True,
        ).to(device)
        detector_model.load_state_dict(ckpt["model_state_dict"])
        detector_model.eval()
    else:
        logger.warning(f"No detector checkpoint at {ckpt_path}. Tracking will use geometric encodings.")

    # Initialize Embedder & Tracker
    embedder = CellNodeEmbedding(
        in_visual_dim=32,
        embedding_dim=128,
        scale=dataset.scale,
        sample_neighborhood=True,
    ).to(device)

    tracker = SpatioTemporalTracker(
        node_dim=128,
        rel_dim=32,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        scale=dataset.scale,
    ).to(device)

    # Build Tracking Dataset (caches embeddings across all timepoints)
    logger.info("Building tracking pair dataset and extracting node embeddings...")
    tracking_dataset = TrackingPairDataset(
        dataset=dataset,
        embedder=embedder,
        detector_model=detector_model,
        device=device,
        max_distance_um=args.max_distance,
        scale=dataset.scale,
    )
    logger.info(f"Built {len(tracking_dataset)} consecutive frame transition pairs.")

    # Criterion, Optimizer, Scheduler
    criterion = TrackerLoss(
        edge_gamma=2.0,
        edge_alpha=0.75,
        div_gamma=2.0,
        div_alpha=0.85,
        div_weight=1.0,
    )
    optimizer = AdamW(tracker.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

    best_loss = float("inf")
    Path(args.output_checkpoint).parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting Tracker training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        metrics = train_epoch(tracker, tracking_dataset, criterion, optimizer, device)
        scheduler.step()

        logger.info(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] | "
            f"Loss: {metrics['loss']:.4f} (Edge: {metrics['edge_loss']:.4f}, Div: {metrics['div_loss']:.4f}) | "
            f"Edge Acc: {metrics['edge_acc']*100:.1f}% | Div Acc: {metrics['div_acc']*100:.1f}%"
        )

        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "tracker_state_dict": tracker.state_dict(),
                    "embedder_state_dict": embedder.state_dict(),
                    "best_loss": best_loss,
                    "metrics": metrics,
                    "config": {
                        "node_dim": 128,
                        "rel_dim": 32,
                        "hidden_dim": args.hidden_dim,
                        "num_layers": args.num_layers,
                        "max_distance_um": args.max_distance,
                        "scale": dataset.scale,
                    },
                },
                args.output_checkpoint,
            )
            logger.info(f"Saved new best tracker checkpoint to {args.output_checkpoint} (loss: {best_loss:.4f})")

    logger.info(f"Tracker training complete. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()

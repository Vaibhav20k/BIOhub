#!/usr/bin/env python3
"""Training script for 3D Temporal U-Net cell nucleus detection baseline."""

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Dict, Optional, Tuple, Union

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from src.data.zarr_reader import open_dataset, DatasetVolume
from src.detection.dataset import DetectionDataset
from src.detection.unet3d import TemporalUNet3D
from src.training.losses import DetectionLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def resolve_dataset_path(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if (p.parent / f"{p.name}.zarr").exists():
        return p
    if p.exists() and p.suffix in (".zarr", ".geff"):
        return p.parent / p.stem
    # Check data/train or data/fixtures
    train_p = Path("data/train") / name_or_path
    if (train_p.parent / f"{train_p.name}.zarr").exists():
        return train_p
    fix_p = Path("data/fixtures") / name_or_path
    if (fix_p.parent / f"{fix_p.name}.zarr").exists():
        return fix_p
    raise FileNotFoundError(f"Dataset '{name_or_path}' not found.")


def run_overfit_verification(
    dataset: DatasetVolume,
    device: torch.device,
    patch_size: Tuple[int, int, int] = (16, 64, 64),
    window_size: int = 2,
    base_channels: int = 16,
    num_iterations: int = 40,
    lr: float = 2e-3,
) -> bool:
    """Run an overfit check on a single fixed patch to verify model capacity and gradient flow."""
    logger.info("=== Starting Overfit Verification Check ===")
    det_dataset = DetectionDataset(
        dataset=dataset,
        patch_size=patch_size,
        window_size=window_size,
        samples_per_epoch=1,
        center_on_cell_prob=1.0,
        augment=False,
        seed=42,
    )

    batch = det_dataset[0]
    image = batch["image"].unsqueeze(0).to(device)  # (1, 1, W, Z, Y, X)
    heatmap_gt = batch["heatmap"].unsqueeze(0).to(device)  # (1, 1, W, Z, Y, X)
    mask = batch["frame_mask"].unsqueeze(0).to(device)

    model = TemporalUNet3D(
        in_channels=1,
        base_channels=base_channels,
        feature_dim=32,
        use_temporal_attention=True,
    ).to(device)

    criterion = DetectionLoss(focal_weight=1.0, dice_weight=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    initial_loss = float("inf")
    final_loss = float("inf")

    model.train()
    for step in range(num_iterations):
        optimizer.zero_grad()
        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            pred_hm, _ = model(image)
            loss, metrics = criterion(pred_hm, heatmap_gt, mask=mask)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_val = metrics["loss_total"]
        if step == 0:
            initial_loss = loss_val
        final_loss = loss_val

        if (step + 1) % 5 == 0 or step == num_iterations - 1:
            logger.info(
                f"Step {step+1:02d}/{num_iterations:02d} | "
                f"Total Loss: {loss_val:.4f} (Focal: {metrics['loss_focal']:.4f}, Dice: {metrics['loss_dice']:.4f})"
            )

    logger.info(f"Overfit check: Initial Loss={initial_loss:.4f} -> Final Loss={final_loss:.4f}")
    loss_reduction = (initial_loss - final_loss) / max(1e-4, initial_loss)
    success = final_loss < 0.35 and loss_reduction > 0.60
    if success:
        logger.info("Overfit verification PASSED! Model converges rapidly.")
    else:
        logger.warning("Overfit check did not achieve target loss threshold.")
    return success


def train_detector(
    train_dataset_name: str,
    val_dataset_name: Optional[str] = None,
    epochs: int = 10,
    samples_per_epoch: int = 40,
    patch_size: Tuple[int, int, int] = (32, 128, 128),
    window_size: int = 2,
    batch_size: int = 1,
    base_channels: int = 16,
    feature_dim: int = 32,
    lr: float = 3e-4,
    checkpoint_dir: Union[str, Path] = "checkpoints",
    reports_dir: Union[str, Path] = "reports",
    device_name: Optional[str] = None,
    seed: int = 42,
) -> Tuple[nn.Module, Dict[str, list]]:
    """Train the 3D Temporal U-Net detector on volumetric patches."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if device_name is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    logger.info(f"Using compute device: {device}")

    # Load datasets
    train_path = resolve_dataset_path(train_dataset_name)
    train_vol = open_dataset(train_path, load_tracks=True)
    logger.info(f"Loaded training volume: {train_vol.name} {train_vol.shape}")

    val_vol = None
    if val_dataset_name:
        val_path = resolve_dataset_path(val_dataset_name)
        val_vol = open_dataset(val_path, load_tracks=True)
        logger.info(f"Loaded validation volume: {val_vol.name} {val_vol.shape}")

    train_data = DetectionDataset(
        dataset=train_vol,
        patch_size=patch_size,
        window_size=window_size,
        samples_per_epoch=samples_per_epoch,
        center_on_cell_prob=0.85,
        augment=True,
        seed=seed,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = None
    if val_vol:
        val_data = DetectionDataset(
            dataset=val_vol,
            patch_size=patch_size,
            window_size=window_size,
            samples_per_epoch=max(10, samples_per_epoch // 4),
            center_on_cell_prob=0.85,
            augment=False,
            seed=seed + 100,
        )
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize model, loss, optimizer
    model = TemporalUNet3D(
        in_channels=1,
        base_channels=base_channels,
        feature_dim=feature_dim,
        use_temporal_attention=True,
    ).to(device)

    criterion = DetectionLoss(focal_weight=1.0, dice_weight=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    history = {"train_loss": [], "train_focal": [], "train_dice": [], "val_loss": []}
    best_loss = float("inf")
    best_ckpt_path = checkpoint_dir / "best_detector.pt"

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        train_focal_losses = []
        train_dice_losses = []

        for batch in train_loader:
            image = batch["image"].to(device)
            heatmap_gt = batch["heatmap"].to(device)
            mask = batch["frame_mask"].to(device)

            optimizer.zero_grad()
            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                pred_hm, _ = model(image)
                loss, metrics = criterion(pred_hm, heatmap_gt, mask=mask)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(metrics["loss_total"])
            train_focal_losses.append(metrics["loss_focal"])
            train_dice_losses.append(metrics["loss_dice"])

        scheduler.step()

        mean_train_loss = float(np.mean(train_losses))
        mean_focal = float(np.mean(train_focal_losses))
        mean_dice = float(np.mean(train_dice_losses))

        history["train_loss"].append(mean_train_loss)
        history["train_focal"].append(mean_focal)
        history["train_dice"].append(mean_dice)

        # Validation step
        mean_val_loss = mean_train_loss
        if val_loader:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    image = batch["image"].to(device)
                    heatmap_gt = batch["heatmap"].to(device)
                    mask = batch["frame_mask"].to(device)

                    with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                        pred_hm, _ = model(image)
                        v_loss, _ = criterion(pred_hm, heatmap_gt, mask=mask)
                    val_losses.append(v_loss.item())
            mean_val_loss = float(np.mean(val_losses))
            history["val_loss"].append(mean_val_loss)

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {mean_train_loss:.4f} (Focal: {mean_focal:.4f}, Dice: {mean_dice:.4f}) | "
            f"Val Loss: {mean_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        # Checkpoint best model
        target_score = mean_val_loss if val_loader else mean_train_loss
        if target_score < best_loss:
            best_loss = target_score
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": best_loss,
                    "config": {
                        "base_channels": base_channels,
                        "feature_dim": feature_dim,
                        "patch_size": patch_size,
                        "scale": train_vol.scale,
                    },
                },
                best_ckpt_path,
            )
            logger.info(f"  -> Saved best checkpoint (loss: {best_loss:.4f}) to {best_ckpt_path}")

    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.1f}s. Best Loss: {best_loss:.4f}")

    # Plot loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), history["train_loss"], label="Train Total Loss", color="#1f77b4", lw=2)
    plt.plot(range(1, epochs + 1), history["train_focal"], label="Train Focal Loss", color="#ff7f0e", linestyle="--")
    plt.plot(range(1, epochs + 1), history["train_dice"], label="Train Dice Loss", color="#2ca02c", linestyle=":")
    if val_loader:
        plt.plot(range(1, epochs + 1), history["val_loss"], label="Val Loss", color="#d62728", lw=2)
    plt.title("3D Temporal U-Net Detection Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    curve_path = reports_dir / "detection_training_loss.png"
    plt.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved loss curve plot to {curve_path}")

    return model, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 3D Temporal U-Net Cell Detector.")
    parser.add_argument("--train-dataset", type=str, default="44b6_0b24845f")
    parser.add_argument("--val-dataset", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--samples-per-epoch", type=int, default=20)
    parser.add_argument("--patch-z", type=int, default=32)
    parser.add_argument("--patch-y", type=int, default=128)
    parser.add_argument("--patch-x", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--reports-dir", type=str, default="reports")
    parser.add_argument("--overfit-check", action="store_true", help="Run rapid overfit verification check.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.overfit_check:
        train_path = resolve_dataset_path(args.train_dataset)
        vol = open_dataset(train_path, load_tracks=True)
        # Use a compact patch for rapid overfit verification
        pz = min(vol.shape[1], args.patch_z)
        py = min(vol.shape[2], args.patch_y)
        px = min(vol.shape[3], args.patch_x)
        success = run_overfit_verification(
            dataset=vol,
            device=device,
            patch_size=(pz, py, px),
            window_size=args.window_size,
            base_channels=args.base_channels,
            num_iterations=50,
        )
        if not success:
            sys.exit(1)
        sys.exit(0)

    train_detector(
        train_dataset_name=args.train_dataset,
        val_dataset_name=args.val_dataset,
        epochs=args.epochs,
        samples_per_epoch=args.samples_per_epoch,
        patch_size=(args.patch_z, args.patch_y, args.patch_x),
        window_size=args.window_size,
        batch_size=args.batch_size,
        base_channels=args.base_channels,
        feature_dim=args.feature_dim,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        reports_dir=args.reports_dir,
    )


if __name__ == "__main__":
    main()

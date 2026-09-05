"""Continual fine-tuning governor across sequential dataset chunks."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader

from src.data.zarr_reader import DatasetVolume, open_dataset
from src.detection.dataset import DetectionDataset
from src.detection.unet3d import TemporalUNet3D
from src.representation.node_embedding import CellNodeEmbedding
from src.tracking.dataset import TrackingPairDataset
from src.tracking.losses import TrackerLoss
from src.training.losses import DetectionLoss
from src.tracking.transformer import SpatioTemporalTracker

logger = logging.getLogger(__name__)


class ContinualTrainer:
    """Manages sequential fine-tuning of 3D Temporal U-Net and Transformer Tracker on chunked sequences."""

    def __init__(
        self,
        detector: TemporalUNet3D,
        embedder: CellNodeEmbedding,
        tracker: SpatioTemporalTracker,
        device: Optional[torch.device] = None,
        detector_lr: float = 0.0005,
        tracker_lr: float = 0.0003,
        weight_decay: float = 1e-4,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = detector.to(self.device)
        self.embedder = embedder.to(self.device)
        self.tracker = tracker.to(self.device)

        self.detector_lr = detector_lr
        self.tracker_lr = tracker_lr
        self.weight_decay = weight_decay

    @classmethod
    def from_checkpoints(
        cls,
        detector_checkpoint: Union[str, Path] = "checkpoints/best_detector.pt",
        tracker_checkpoint: Union[str, Path] = "checkpoints/best_tracker.pt",
        device: Optional[torch.device] = None,
        detector_lr: float = 0.0005,
        tracker_lr: float = 0.0003,
    ) -> "ContinualTrainer":
        """Initialize trainer by loading existing model checkpoints."""
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load detector
        det_ckpt = torch.load(detector_checkpoint, map_location=dev)
        detector = TemporalUNet3D(
            in_channels=1,
            base_channels=det_ckpt.get("config", {}).get("base_channels", 16),
            feature_dim=det_ckpt.get("config", {}).get("feature_dim", 32),
            use_temporal_attention=True,
        )
        detector.load_state_dict(det_ckpt["model_state_dict"])

        # Load embedder & tracker
        trk_ckpt = torch.load(tracker_checkpoint, map_location=dev)
        embedder = CellNodeEmbedding(in_visual_dim=32, embedding_dim=128)
        if "embedder_state_dict" in trk_ckpt:
            embedder.load_state_dict(trk_ckpt["embedder_state_dict"])

        tracker = SpatioTemporalTracker(
            node_dim=128,
            rel_dim=32,
            hidden_dim=trk_ckpt.get("config", {}).get("hidden_dim", 128),
            num_layers=trk_ckpt.get("config", {}).get("num_layers", 2),
        )
        tracker.load_state_dict(trk_ckpt["tracker_state_dict"])

        return cls(
            detector=detector,
            embedder=embedder,
            tracker=tracker,
            device=dev,
            detector_lr=detector_lr,
            tracker_lr=tracker_lr,
        )

    def train_detector_on_chunk(
        self,
        datasets: List[DatasetVolume],
        epochs: int = 5,
        batch_size: int = 2,
        samples_per_seq: int = 50,
        patch_size: Tuple[int, int, int] = (32, 128, 128),
        lr_multiplier: float = 1.0,
    ) -> float:
        """Fine-tune 3D Temporal U-Net detector across chunk sequences."""
        self.detector.train()

        # Build concatenated dataset
        sub_datasets = [
            DetectionDataset(
                dataset=ds,
                patch_size=(
                    min(patch_size[0], ds.shape[1]),
                    min(patch_size[1], ds.shape[2]),
                    min(patch_size[2], ds.shape[3]),
                ),
                window_size=2,
                samples_per_epoch=samples_per_seq,
                center_on_cell_prob=0.8,
                augment=True,
            )
            for ds in datasets
            if ds.tracks is not None and ds.tracks.num_nodes() > 0
        ]

        if not sub_datasets:
            logger.warning("No datasets with tracks available in chunk for detector training.")
            return 0.0

        concat_ds = ConcatDataset(sub_datasets)
        loader = DataLoader(concat_ds, batch_size=batch_size, shuffle=True, drop_last=False)

        current_lr = self.detector_lr * lr_multiplier
        optimizer = AdamW(self.detector.parameters(), lr=current_lr, weight_decay=self.weight_decay)
        criterion = DetectionLoss(focal_weight=1.0, dice_weight=1.0)
        scaler = GradScaler("cuda", enabled=self.device.type == "cuda")

        logger.info(f"Fine-tuning detector on {len(concat_ds)} samples ({len(datasets)} seqs) for {epochs} epochs at lr={current_lr:.6f}...")

        avg_loss = 0.0
        for ep in range(epochs):
            total_loss = 0.0
            steps = 0
            for batch in loader:
                images = batch["image"].to(self.device)  # (B, 1, W, Z, Y, X)
                heatmaps = batch["heatmap"].to(self.device)
                masks = batch["frame_mask"].to(self.device)

                optimizer.zero_grad()
                with autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                    preds, _ = self.detector(images)
                    loss, metrics = criterion(preds, heatmaps, mask=masks)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.detector.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                total_loss += float(metrics["loss_total"])
                steps += 1

            avg_loss = total_loss / max(1, steps)
            logger.info(f" [Detector Epoch {ep+1}/{epochs}] Loss: {avg_loss:.5f}")

        return avg_loss

    def train_tracker_on_chunk(
        self,
        datasets: List[DatasetVolume],
        epochs: int = 5,
        max_distance_um: float = 7.0,
        lr_multiplier: float = 1.0,
    ) -> float:
        """Fine-tune Spatio-Temporal Transformer tracker across chunk sequences."""
        self.tracker.train()

        sub_datasets = [
            TrackingPairDataset(
                dataset=ds,
                embedder=self.embedder,
                detector_model=self.detector,
                device=self.device,
                max_distance_um=max_distance_um,
                scale=ds.scale,
            )
            for ds in datasets
            if ds.tracks is not None and ds.tracks.num_edges() > 0
        ]

        if not sub_datasets:
            logger.warning("No datasets with tracks/edges in chunk for tracker training.")
            return 0.0

        current_lr = self.tracker_lr * lr_multiplier
        optimizer = AdamW(self.tracker.parameters(), lr=current_lr, weight_decay=self.weight_decay)
        criterion = TrackerLoss(div_weight=2.0)

        logger.info(f"Fine-tuning tracker across {len(sub_datasets)} sequences for {epochs} epochs at lr={current_lr:.6f}...")

        avg_loss = 0.0
        for ep in range(epochs):
            total_loss = 0.0
            steps = 0
            for ds in sub_datasets:
                indices = torch.randperm(len(ds)).tolist()
                for idx in indices:
                    src_embs, dst_embs, cand_batch = ds[idx]
                    src_embs = src_embs.to(self.device)
                    dst_embs = dst_embs.to(self.device)

                    optimizer.zero_grad()
                    pred = self.tracker(src_embs, dst_embs, cand_batch)

                    edge_targets = cand_batch.edge_labels.to(self.device) if cand_batch.edge_labels is not None else None
                    div_targets = cand_batch.division_labels.to(self.device) if cand_batch.division_labels is not None else None

                    loss_dict = criterion(
                        edge_logits=pred.edge_logits,
                        edge_targets=edge_targets,
                        division_logits=pred.division_logits,
                        division_targets=div_targets,
                    )
                    loss = loss_dict["loss"]

                    if loss.requires_grad:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.tracker.parameters(), max_norm=2.0)
                        optimizer.step()

                    total_loss += float(loss.item())
                    steps += 1

            avg_loss = total_loss / max(1, steps)
            logger.info(f" [Tracker Epoch {ep+1}/{epochs}] Loss: {avg_loss:.5f}")

        return avg_loss

    def save_checkpoints(
        self,
        detector_path: Union[str, Path] = "checkpoints/latest_detector.pt",
        tracker_path: Union[str, Path] = "checkpoints/latest_tracker.pt",
    ) -> None:
        """Save model weights to specified checkpoint paths."""
        Path(detector_path).parent.mkdir(parents=True, exist_ok=True)
        Path(tracker_path).parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state_dict": self.detector.state_dict(),
                "config": {"base_channels": 16, "feature_dim": 32},
            },
            detector_path,
        )

        torch.save(
            {
                "tracker_state_dict": self.tracker.state_dict(),
                "embedder_state_dict": self.embedder.state_dict(),
                "config": {"hidden_dim": 128, "num_layers": 2},
            },
            tracker_path,
        )
        logger.info(f"Checkpoints saved: {detector_path} and {tracker_path}")

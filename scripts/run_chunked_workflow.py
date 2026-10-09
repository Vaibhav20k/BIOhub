#!/usr/bin/env python3
"""Storage-Efficient Chunked Dataset Workflow Orchestrator.

Progressively downloads, verifies, fine-tunes, evaluates, persists,
and safely cleans up 3-5 GB competition dataset chunks.
"""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from src.data.zarr_reader import DatasetVolume, open_dataset
from src.pipeline.chunk_manager import ChunkStorageManager
from src.pipeline.continual_trainer import ContinualTrainer
from src.pipeline.held_out_evaluator import HeldOutEvaluator
from src.pipeline.registry import ChunkRecord, ChunkRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/chunked_workflow.yaml") -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_stratified_manifest(
    catalog_path: str = "dataset_catalog.json",
    val_ratio: float = 0.15,
    chunk_size: int = 8,
    seed: int = 42,
) -> Tuple[List[str], List[ChunkRecord]]:
    """Partition 199 sequences into a permanent held-out validation set and sequential training chunks."""
    with open(catalog_path, "r") as f:
        catalog = json.load(f)

    train_catalog = catalog["train"]

    # Stratify by embryo prefix (44b6 vs 6bba)
    by_embryo: Dict[str, List[str]] = {}
    for stem in sorted(train_catalog.keys()):
        prefix = stem.split("_")[0]
        by_embryo.setdefault(prefix, []).append(stem)

    rng = random.Random(seed)
    val_sequences: List[str] = []
    train_pool: List[Tuple[str, str, int]] = []  # (stem, embryo, bytes)

    for embryo, stems in sorted(by_embryo.items()):
        shuffled = list(stems)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_ratio))
        val_subset = shuffled[:n_val]
        train_subset = shuffled[n_val:]

        val_sequences.extend(val_subset)
        for s in train_subset:
            sz = train_catalog[s]["zarr_bytes"] + train_catalog[s]["geff_bytes"]
            train_pool.append((s, embryo, sz))

    # Shuffle training pool deterministically to interleave embryos
    rng.shuffle(train_pool)

    # Group into chunks
    chunk_records: List[ChunkRecord] = []
    chunk_idx = 1

    for i in range(0, len(train_pool), chunk_size):
        chunk_items = train_pool[i : i + chunk_size]
        stems = [item[0] for item in chunk_items]
        total_sz = sum(item[2] for item in chunk_items)
        embryos = set(item[1] for item in chunk_items)
        embryo_str = embryos.pop() if len(embryos) == 1 else "mixed"

        rec = ChunkRecord(
            chunk_id=f"chunk_{chunk_idx:03d}_train",
            sequence_stems=stems,
            embryo_group=embryo_str,
            chunk_size_bytes=total_sz,
            role="train",
            status="PENDING",
        )
        chunk_records.append(rec)
        chunk_idx += 1

    return val_sequences, chunk_records


def init_workflow(config: Dict) -> None:
    """Initialize SQLite/JSONL registry and build chunk manifest."""
    catalog_file = config["competition"]["catalog_file"]
    val_ratio = config["validation"]["val_ratio"]
    chunk_size = config["storage"]["target_sequences_per_chunk"]
    seed = config["validation"]["seed"]

    logger.info(f"Generating stratified chunks from {catalog_file} (val_ratio={val_ratio}, chunk_size={chunk_size})...")
    val_seqs, chunk_records = build_stratified_manifest(
        catalog_path=catalog_file,
        val_ratio=val_ratio,
        chunk_size=chunk_size,
        seed=seed,
    )

    # Save validation sequences manifest
    val_file = Path("registry/val_sequences.json")
    val_file.parent.mkdir(parents=True, exist_ok=True)
    with open(val_file, "w") as f:
        json.dump(val_seqs, f, indent=2)
    logger.info(f"Saved {len(val_seqs)} held-out validation sequences to {val_file}")

    # Register into DB
    registry = ChunkRegistry(
        sqlite_path=config["registry"]["sqlite_path"],
        jsonl_path=config["registry"]["jsonl_path"],
    )
    new_chunks = registry.register_manifest(chunk_records)
    logger.info(f"Manifest initialized: {len(chunk_records)} total chunks ({new_chunks} newly registered).")


def setup_validation_anchor(
    config: Dict,
    storage_mgr: Optional[ChunkStorageManager] = None,
    dry_run: bool = False,
) -> List[DatasetVolume]:
    """Ensure dedicated permanent validation anchor sequences are downloaded in data/val/ and loaded."""
    if dry_run:
        fixture_p = Path("data/fixtures/synthetic_clip")
        return [open_dataset(fixture_p, load_tracks=True, require_tracks=True)]

    val_dir = Path(config["validation"].get("val_dir", "data/val"))
    val_dir.mkdir(parents=True, exist_ok=True)
    anchor_stems = config["validation"].get("val_anchor_sequences", ["44b6_66f9292d", "6bba_d82a4fc6"])

    if storage_mgr is None:
        storage_mgr = ChunkStorageManager(
            staging_dir=config["storage"]["staging_dir"],
            min_free_disk_gb=config["storage"]["min_free_disk_gb"],
            competition_name=config["competition"]["name"],
        )

    loaded = []
    for stem in anchor_stems:
        stem_zarr = val_dir / f"{stem}.zarr"
        stem_geff = val_dir / f"{stem}.geff"

        is_valid = False
        if stem_zarr.exists() and stem_geff.exists():
            valid, _ = storage_mgr.verify_sequence_integrity(val_dir / stem, require_tracks=True)
            is_valid = valid

        if not is_valid:
            logger.info(f"Downloading permanent validation anchor sequence '{stem}' to {val_dir}...")
            storage_mgr.assert_disk_safety(required_extra_gb=1.0)
            ok = storage_mgr.download_sequence(stem=stem, split="train", dest_dir=val_dir)
            if not ok:
                raise RuntimeError(f"Failed to download validation anchor sequence '{stem}'")
            storage_mgr.verify_sequence_integrity(val_dir / stem, require_tracks=True)

        loaded.append(open_dataset(val_dir / stem, load_tracks=True))

    logger.info(f"Permanent validation anchor ready: {[ds.name for ds in loaded]}")
    return loaded


def print_status(config: Dict) -> None:
    """Print comprehensive summary of processed chunks, sizes, and scores."""
    registry = ChunkRegistry(
        sqlite_path=config["registry"]["sqlite_path"],
        jsonl_path=config["registry"]["jsonl_path"],
    )
    summary = registry.get_summary()
    chunks = registry.get_all_chunks()

    print("\n================================================================================")
    print("           BIOHUB STORAGE-EFFICIENT CHUNKED WORKFLOW STATUS")
    print("================================================================================")
    print(f"Total Chunks        : {summary['total_chunks']}")
    print(f"Completed Chunks    : {summary['completed_chunks']}")
    print(f"Failed Chunks       : {summary['failed_chunks']}")
    print(f"Total Dataset Size  : {summary['total_size_gb']:.2f} GB")
    print(f"Processed Size      : {summary['processed_size_gb']:.2f} GB")
    print(f"Disk Space Freed    : {summary['disk_freed_gb']:.2f} GB")
    print(f"Best Val Final Score: {summary['best_validation_score']:.4f}")
    print("--------------------------------------------------------------------------------")
    print(f"{'Chunk ID':<18} | {'Embryo':<6} | {'Seqs':<4} | {'Size (MB)':<9} | {'Status':<10} | {'Score':<6} | {'Cleaned'}")
    print("--------------------------------------------------------------------------------")
    for c in chunks:
        score_str = f"{c.val_final_score:.4f}" if c.val_final_score is not None else "  N/A "
        print(f"{c.chunk_id:<18} | {c.embryo_group:<6} | {len(c.sequence_stems):<4} | {c.chunk_size_bytes/(1024**2):<9.1f} | {c.status:<10} | {score_str:<6} | {c.cleanup_verified}")
    print("================================================================================\n")


def process_chunk(
    chunk_record: ChunkRecord,
    config: Dict,
    dry_run: bool = False,
) -> None:
    """Execute complete download -> verify -> train -> evaluate -> persist -> cleanup cycle."""
    start_time = time.time()
    logger.info(f"\n>>> Starting processing for chunk: {chunk_record.chunk_id} ({len(chunk_record.sequence_stems)} sequences) <<<")

    storage_mgr = ChunkStorageManager(
        staging_dir=config["storage"]["staging_dir"],
        min_free_disk_gb=config["storage"]["min_free_disk_gb"],
        competition_name=config["competition"]["name"],
    )
    registry = ChunkRegistry(
        sqlite_path=config["registry"]["sqlite_path"],
        jsonl_path=config["registry"]["jsonl_path"],
    )

    chunk_dir = Path(config["storage"]["staging_dir"]) / chunk_record.chunk_id

    # 1. Pre-flight disk space check
    storage_mgr.assert_disk_safety(required_extra_gb=chunk_record.chunk_size_bytes / (1024**3))

    try:
        # 2. Download
        if not dry_run:
            logger.info(f"Downloading sequences for {chunk_record.chunk_id}...")
            storage_mgr.download_chunk_sequences(
                sequence_stems=chunk_record.sequence_stems,
                chunk_dir=chunk_dir,
                split="train",
            )
            chunk_record.status = "DOWNLOADED"
            registry.update_chunk(chunk_record)

            # 3. Verify integrity
            storage_mgr.verify_chunk_integrity(chunk_dir, chunk_record.sequence_stems, require_tracks=True)
            loaded_datasets = [
                open_dataset(chunk_dir / stem, load_tracks=True, require_tracks=True)
                for stem in chunk_record.sequence_stems
            ]
        else:
            logger.info("[DRY RUN] Using local synthetic fixture...")
            fixture_p = Path("data/fixtures/synthetic_clip")
            loaded_datasets = [open_dataset(fixture_p, load_tracks=True, require_tracks=True)]
            chunk_dir.mkdir(parents=True, exist_ok=True)
            (chunk_dir / "mock_data.bin").write_bytes(b"0" * 1024 * 1024)

        # 4. Continual Training
        logger.info(f"Fine-tuning models on {chunk_record.chunk_id}...")
        trainer = ContinualTrainer.from_checkpoints(
            detector_checkpoint=config["checkpoints"]["best_detector"],
            tracker_checkpoint=config["checkpoints"]["best_tracker"],
            detector_lr=config["training"]["detector"]["base_lr"],
            tracker_lr=config["training"]["tracker"]["base_lr"],
        )

        det_epochs = 1 if dry_run else config["training"]["detector"]["epochs_per_chunk"]
        trk_epochs = 1 if dry_run else config["training"]["tracker"]["epochs_per_chunk"]

        chunk_num = 1
        try:
            chunk_num = int(chunk_record.chunk_id.split("_")[1])
        except Exception:
            pass

        det_decay = float(config["training"]["detector"].get("lr_decay_per_chunk", 1.0)) ** max(0, chunk_num - 1)
        trk_decay = float(config["training"]["tracker"].get("lr_decay_per_chunk", 1.0)) ** max(0, chunk_num - 1)

        win_size = config["training"]["detector"].get("window_size", 1)
        det_patch_size = tuple(config["training"]["detector"].get("patch_size", [32, 128, 128]))
        samples_per_seq = int(config["training"]["detector"].get("samples_per_seq", 100))
        center_on_cell_prob = float(config["training"]["detector"].get("center_on_cell_prob", 0.8))
        use_amp = bool(config["training"]["detector"].get("use_amp", False))
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        det_loss = trainer.train_detector_on_chunk(
            datasets=loaded_datasets,
            epochs=det_epochs,
            batch_size=config["training"]["detector"]["batch_size"],
            samples_per_seq=samples_per_seq,
            center_on_cell_prob=center_on_cell_prob,
            patch_size=det_patch_size,
            window_size=win_size,
            use_amp=use_amp,
            lr_multiplier=det_decay,
        )
        trk_loss = trainer.train_tracker_on_chunk(
            datasets=loaded_datasets,
            epochs=trk_epochs,
            lr_multiplier=trk_decay,
        )
        train_vram_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        logger.info(f"Training Peak VRAM: {train_vram_gb:.3f} GB")

        # Save latest checkpoints
        trainer.save_checkpoints(
            detector_path=config["checkpoints"]["latest_detector"],
            tracker_path=config["checkpoints"]["latest_tracker"],
        )

        chunk_record.epochs_trained = det_epochs + trk_epochs
        chunk_record.train_loss_detector = det_loss
        chunk_record.train_loss_tracker = trk_loss
        chunk_record.training_config = config.get("training", {})
        chunk_record.status = "PROCESSED"
        registry.update_chunk(chunk_record)

        # 5. Held-Out Evaluation on Dedicated Validation Anchor
        logger.info("Evaluating updated models on held-out validation anchor...")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        det_threshold = float(config.get("evaluation", {}).get("detection_threshold", 0.25))
        evaluator = HeldOutEvaluator(
            detector=trainer.detector,
            embedder=trainer.embedder,
            tracker=trainer.tracker,
            detection_threshold=det_threshold,
            max_peaks_per_frame=150,
            max_distance_um=config["training"]["tracker"]["max_distance_um"],
            window_size=win_size,
            diagnostic_mode=False,
        )

        val_candidates = setup_validation_anchor(config, storage_mgr, dry_run=dry_run)
        val_summary = evaluator.evaluate_cohort(val_candidates)
        eval_vram_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        logger.info(f"Evaluation Peak VRAM: {eval_vram_gb:.3f} GB")
        score = val_summary["macro_final_score"]
        adj_jaccard = val_summary["macro_adjusted_jaccard"]
        div_jaccard = val_summary["macro_division_jaccard"]
        edge_jaccard = val_summary["macro_edge_jaccard"]

        logger.info(
            f"Validation results: Final Score = {score:.4f} "
            f"(J_adj={adj_jaccard:.4f}, J_div={div_jaccard:.4f}, Edge_J={edge_jaccard:.4f})"
        )

        prev_best = registry.get_best_score()
        promoted = False
        rollback_threshold = config["checkpoints"].get("rollback_threshold_drop", 0.15)

        if score >= prev_best:
            logger.info(f"★ NEW BEST MODEL! Score improved from {prev_best:.4f} to {score:.4f}. Promoting checkpoints.")
            shutil.copy(config["checkpoints"]["latest_detector"], config["checkpoints"]["best_detector"])
            shutil.copy(config["checkpoints"]["latest_tracker"], config["checkpoints"]["best_tracker"])
            promoted = True
        elif prev_best > 0 and score < prev_best * (1.0 - rollback_threshold):
            logger.warning(
                f"Validation score {score:.4f} dropped > {rollback_threshold:.0%} below best {prev_best:.4f}. "
                f"Rolling back latest weights to golden best checkpoint."
            )
            shutil.copy(config["checkpoints"]["best_detector"], config["checkpoints"]["latest_detector"])
            shutil.copy(config["checkpoints"]["best_tracker"], config["checkpoints"]["latest_tracker"])

        # Archive checkpoint snapshot per chunk
        archive_dir = Path(config["checkpoints"].get("archive_dir", ""))
        if archive_dir and not dry_run:
            archive_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(config["checkpoints"]["latest_detector"], archive_dir / f"{chunk_record.chunk_id}_detector.pt")
            shutil.copy(config["checkpoints"]["latest_tracker"], archive_dir / f"{chunk_record.chunk_id}_tracker.pt")

        chunk_record.val_sequences_evaluated = [ds.name for ds in val_candidates]
        chunk_record.val_node_tp = val_summary.get("total_node_tp")
        chunk_record.val_node_fp = val_summary.get("total_node_fp")
        chunk_record.val_node_fn = val_summary.get("total_node_fn")
        chunk_record.val_node_precision = val_summary.get("macro_node_precision")
        chunk_record.val_node_recall = val_summary.get("macro_node_recall")
        chunk_record.val_edge_jaccard = edge_jaccard
        chunk_record.val_adj_jaccard = adj_jaccard
        chunk_record.val_div_jaccard = div_jaccard
        chunk_record.val_final_score = score
        chunk_record.inference_time_sec = val_summary.get("inference_time_sec", 0.0)
        chunk_record.ilp_runtime_sec = val_summary.get("ilp_runtime_sec", 0.0)
        chunk_record.promoted_to_best = promoted
        chunk_record.status = "EVALUATED"
        registry.update_chunk(chunk_record)

        # 6. Transactional Cleanup
        logger.info(f"Commencing safe cleanup for {chunk_record.chunk_id}...")
        freed = storage_mgr.cleanup_chunk(
            chunk_dir=chunk_dir,
            registry_verified=True,
            checkpoints_verified=Path(config["checkpoints"]["latest_detector"]).exists(),
        )

        duration = time.time() - start_time
        chunk_record.status = "CLEANED_UP"
        chunk_record.disk_freed_bytes = freed
        chunk_record.cleanup_verified = True
        chunk_record.duration_sec = duration
        chunk_record.completed_at = datetime.now(timezone.utc).isoformat()
        registry.update_chunk(chunk_record)

        logger.info(f"✓ Chunk {chunk_record.chunk_id} successfully completed in {duration:.1f}s ({freed / (1024**2):.1f} MB freed).")

    except Exception as e:
        logger.error(f"Error processing chunk {chunk_record.chunk_id}: {str(e)}", exc_info=True)
        chunk_record.status = "FAILED"
        chunk_record.error_message = str(e)
        registry.update_chunk(chunk_record)
        if chunk_dir.exists():
            logger.info(f"Reclaiming disk space from failed chunk directory {chunk_dir}...")
            try:
                storage_mgr.cleanup_chunk(chunk_dir, registry_verified=True, checkpoints_verified=False)
            except Exception as cleanup_err:
                logger.warning(f"Error cleaning up failed chunk dir: {cleanup_err}")
        raise e


def main() -> None:
    parser = argparse.ArgumentParser(description="Storage-efficient chunked dataset workflow.")
    parser.add_argument("--config", type=str, default="configs/chunked_workflow.yaml", help="Path to YAML config")
    parser.add_argument("--init", action="store_true", help="Initialize manifest and registry from dataset catalog")
    parser.add_argument("--status", action="store_true", help="Display workflow execution and disk status")
    parser.add_argument("--setup-val-anchor", action="store_true", help="Download and verify permanent validation anchor sequences")
    parser.add_argument("--process-next", action="store_true", help="Process the next pending chunk")
    parser.add_argument("--process-all", action="store_true", help="Process all remaining pending chunks sequentially")
    parser.add_argument("--chunk-id", type=str, default=None, help="Process a specific chunk by ID")
    parser.add_argument("--dry-run", action="store_true", help="Simulate chunk cycle with local fixtures")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.init:
        init_workflow(config)
        print_status(config)
        return

    if args.setup_val_anchor:
        setup_validation_anchor(config, dry_run=args.dry_run)
        return

    if args.status:
        print_status(config)
        return

    registry = ChunkRegistry(
        sqlite_path=config["registry"]["sqlite_path"],
        jsonl_path=config["registry"]["jsonl_path"],
    )

    if args.process_all:
        processed_count = 0
        while True:
            target_chunk = registry.get_next_pending_chunk()
            if target_chunk is None:
                logger.info("All chunks in registry have completed processing! Nothing pending.")
                break
            logger.info(f"\n{'='*70}\n[RUN ALL] Processing chunk {target_chunk.chunk_id} (count: {processed_count + 1})\n{'='*70}")
            process_chunk(target_chunk, config, dry_run=args.dry_run)
            processed_count += 1
            print_status(config)
        logger.info(f"Finished processing all pending chunks. Total processed this session: {processed_count}")
        return

    target_chunk = None
    if args.chunk_id:
        target_chunk = registry.get_chunk(args.chunk_id)
        if target_chunk is None:
            logger.error(f"Chunk '{args.chunk_id}' not found in registry.")
            sys.exit(1)
    elif args.process_next or args.dry_run:
        target_chunk = registry.get_next_pending_chunk()
        if target_chunk is None:
            logger.info("All chunks in registry are already completed! Nothing to process.")
            sys.exit(0)

    if target_chunk:
        process_chunk(target_chunk, config, dry_run=args.dry_run)
        print_status(config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

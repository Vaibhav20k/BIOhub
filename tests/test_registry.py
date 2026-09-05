"""Unit tests for ChunkRegistry and ChunkRecord persistence."""

from pathlib import Path
import pytest
from src.pipeline.registry import ChunkRecord, ChunkRegistry


def test_registry_lifecycle(tmp_path: Path):
    sqlite_file = tmp_path / "test_reg.sqlite"
    jsonl_file = tmp_path / "test_reg.jsonl"

    reg = ChunkRegistry(sqlite_path=sqlite_file, jsonl_path=jsonl_file)
    assert sqlite_file.exists()

    rec1 = ChunkRecord(
        chunk_id="chunk_001_train",
        sequence_stems=["44b6_seq1", "44b6_seq2"],
        embryo_group="44b6",
        chunk_size_bytes=800_000_000,
        role="train",
    )
    rec2 = ChunkRecord(
        chunk_id="chunk_002_train",
        sequence_stems=["6bba_seq1", "6bba_seq2"],
        embryo_group="6bba",
        chunk_size_bytes=900_000_000,
        role="train",
    )

    # Register manifest
    new_count = reg.register_manifest([rec1, rec2])
    assert new_count == 2

    # Attempt re-registration: should be idempotent (0 new)
    assert reg.register_manifest([rec1]) == 0

    # Get next pending
    next_chunk = reg.get_next_pending_chunk()
    assert next_chunk is not None
    assert next_chunk.chunk_id == "chunk_001_train"

    # Update chunk
    next_chunk.status = "CLEANED_UP"
    next_chunk.epochs_trained = 5
    next_chunk.train_loss_detector = 0.0025
    next_chunk.val_final_score = 0.85
    next_chunk.promoted_to_best = True
    next_chunk.disk_freed_bytes = 800_000_000
    next_chunk.cleanup_verified = True
    reg.update_chunk(next_chunk)

    # Verify update in DB
    fetched = reg.get_chunk("chunk_001_train")
    assert fetched is not None
    assert fetched.status == "CLEANED_UP"
    assert fetched.val_final_score == 0.85
    assert fetched.promoted_to_best is True
    assert fetched.cleanup_verified is True

    # Next pending should now be chunk 2
    next_chunk_2 = reg.get_next_pending_chunk()
    assert next_chunk_2 is not None
    assert next_chunk_2.chunk_id == "chunk_002_train"

    # Check best score
    assert reg.get_best_score() == 0.85

    # Check completed sequences
    completed_seqs = reg.get_completed_sequences()
    assert "44b6_seq1" in completed_seqs
    assert "44b6_seq2" in completed_seqs
    assert "6bba_seq1" not in completed_seqs

    # Check summary
    summary = reg.get_summary()
    assert summary["total_chunks"] == 2
    assert summary["completed_chunks"] == 1
    assert summary["best_validation_score"] == 0.85

    # Check JSONL log file exists and has content
    assert jsonl_file.exists()
    lines = jsonl_file.read_text().strip().split("\n")
    assert len(lines) == 1

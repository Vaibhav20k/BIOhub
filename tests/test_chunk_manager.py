"""Unit tests for ChunkStorageManager disk safety, integrity checks, and cleanup."""

from pathlib import Path
import pytest
from src.pipeline.chunk_manager import ChunkStorageManager


def test_disk_safety_checks(tmp_path: Path):
    mgr = ChunkStorageManager(staging_dir=tmp_path / "staging", min_free_disk_gb=1.0)
    free_gb = mgr.get_disk_free_gb()
    assert free_gb > 0

    # Should pass under reasonable threshold
    mgr.assert_disk_safety(required_extra_gb=0.1)

    # Should raise when requesting impossible disk space
    with pytest.raises(RuntimeError, match="Disk safety guard tripped"):
        mgr.assert_disk_safety(required_extra_gb=1_000_000.0)


def test_verify_sequence_integrity_synthetic_fixture():
    mgr = ChunkStorageManager(staging_dir="data/chunks")
    fixture_path = Path("data/fixtures/synthetic_clip")

    is_valid, err = mgr.verify_sequence_integrity(fixture_path, require_tracks=True)
    assert is_valid is True
    assert err is None

    # Test non-existent path
    is_valid_fake, err_fake = mgr.verify_sequence_integrity(Path("data/fixtures/non_existent_clip"))
    assert is_valid_fake is False
    assert "missing" in err_fake.lower()


def test_cleanup_chunk_safety_guards(tmp_path: Path):
    staging_dir = tmp_path / "staging"
    mgr = ChunkStorageManager(staging_dir=staging_dir)

    chunk_dir = staging_dir / "chunk_001"
    chunk_dir.mkdir(parents=True)
    test_file = chunk_dir / "test_data.bin"
    test_file.write_bytes(b"0" * 1024 * 1024)  # 1 MB

    # 1. Blocked if registry not verified
    with pytest.raises(RuntimeError, match="registry record has not been verified"):
        mgr.cleanup_chunk(chunk_dir, registry_verified=False, checkpoints_verified=True)

    # 2. Blocked if checkpoints not verified
    with pytest.raises(RuntimeError, match="model checkpoints have not been verified"):
        mgr.cleanup_chunk(chunk_dir, registry_verified=True, checkpoints_verified=False)

    # 3. Blocked if target is outside staging_dir
    outside_dir = tmp_path / "outside_dir"
    outside_dir.mkdir()
    with pytest.raises(RuntimeError, match="CRITICAL SAFETY ABORT"):
        mgr.cleanup_chunk(outside_dir, registry_verified=True, checkpoints_verified=True)

    # 4. Successful cleanup
    freed_bytes = mgr.cleanup_chunk(chunk_dir, registry_verified=True, checkpoints_verified=True)
    assert freed_bytes >= 1024 * 1024
    assert not chunk_dir.exists()

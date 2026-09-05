"""Integration tests for the chunked dataset workflow."""

import json
from pathlib import Path
import pytest

from scripts.run_chunked_workflow import build_stratified_manifest, load_config
from src.pipeline.registry import ChunkRegistry


def test_stratified_manifest_no_leakage():
    """Assert zero leakage between train chunks and permanent held-out validation sequences."""
    val_seqs, chunks = build_stratified_manifest(
        catalog_path="dataset_catalog.json",
        val_ratio=0.15,
        chunk_size=8,
        seed=42,
    )

    assert len(val_seqs) > 0
    val_set = set(val_seqs)

    train_stems_all = []
    for c in chunks:
        assert len(c.sequence_stems) <= 8
        for s in c.sequence_stems:
            train_stems_all.append(s)

    # 1. Zero intersection between train and held-out validation
    train_set = set(train_stems_all)
    overlap = train_set.intersection(val_set)
    assert len(overlap) == 0, f"Data leakage detected! Overlapping stems: {overlap}"

    # 2. No duplicate sequences across training chunks
    assert len(train_stems_all) == len(train_set)

    # 3. Total sequences match catalog train count (199)
    assert len(train_set) + len(val_set) == 199


def test_chunked_workflow_config_integrity():
    """Verify that chunked_workflow.yaml exists and contains all required parameters."""
    cfg = load_config("configs/chunked_workflow.yaml")

    assert "storage" in cfg
    assert cfg["storage"]["min_free_disk_gb"] >= 10.0
    assert cfg["storage"]["target_sequences_per_chunk"] == 8

    assert "registry" in cfg
    assert "validation" in cfg
    assert "training" in cfg
    assert "checkpoints" in cfg

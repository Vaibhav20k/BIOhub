#!/usr/bin/env python3
"""Download and verify Chunk 1 sequences."""
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.chunk_manager import ChunkStorageManager
from kaggle.api.kaggle_api_extended import KaggleApi

CHUNK1_STEMS = [
    "6bba_3fda6b25",
    "6bba_9a41d029",
    "44b6_949adeb1",
    "44b6_5740d24b",
    "6bba_05b6850b",
    "6bba_f4ae811c",
    "6bba_b1ae37b9",
    "6bba_7b5d3b2c",
]

def main():
    chunk_dir = Path("data/chunks/chunk_001_train")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    mgr = ChunkStorageManager()
    
    print(f"Checking disk space... Free: {mgr.get_disk_free_gb():.2f} GB")
    mgr.assert_disk_safety(required_extra_gb=4.0)
    
    api = KaggleApi()
    api.authenticate()
    
    for idx, stem in enumerate(CHUNK1_STEMS, 1):
        print(f"\n[{idx}/{len(CHUNK1_STEMS)}] Processing {stem}...")
        is_valid, _ = mgr.verify_sequence_integrity(chunk_dir / stem, require_tracks=True)
        if is_valid:
            print(f"  -> {stem} already exists and passed integrity check. Skipping.")
            continue
        
        max_seq_attempts = 3
        for seq_attempt in range(max_seq_attempts):
            t0 = time.time()
            try:
                ok = mgr.download_sequence(stem=stem, split="train", dest_dir=chunk_dir, api=api)
                valid, err = mgr.verify_sequence_integrity(chunk_dir / stem, require_tracks=True)
                if valid:
                    elapsed = time.time() - t0
                    print(f"  -> {stem} verified successfully in {elapsed:.1f}s.")
                    break
                else:
                    print(f"  [Attempt {seq_attempt+1}/{max_seq_attempts}] Verification failed for {stem}: {err}")
                    if seq_attempt < max_seq_attempts - 1:
                        print("  Waiting 20s before retry...")
                        time.sleep(20)
                    else:
                        print(f"  FATAL: {stem} failed verification after {max_seq_attempts} attempts.")
                        sys.exit(1)
            except Exception as e:
                print(f"  [Attempt {seq_attempt+1}/{max_seq_attempts}] Download error on {stem}: {e}")
                if seq_attempt < max_seq_attempts - 1:
                    print("  Waiting 25s before retry...")
                    time.sleep(25)
                else:
                    print(f"  FATAL: {stem} failed after {max_seq_attempts} attempts.")
                    sys.exit(1)
        
    print("\nAll 8 sequences for Chunk 1 are downloaded and verified intact!")

if __name__ == "__main__":
    main()

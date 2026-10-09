"""Storage manager for chunked dataset downloads, disk safety checks, integrity validation, and safe cleanup."""

import concurrent.futures
import logging
import os
from pathlib import Path
import random
import shutil
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import tracksdata as td
import zarr

logger = logging.getLogger(__name__)

COMPETITION_NAME = "biohub-cell-tracking-during-development"

GEFF_TEMPLATE_FILES = [
    "edges/ids/c/0/0",
    "edges/ids/zarr.json",
    "edges/props/zarr.json",
    "edges/zarr.json",
    "nodes/ids/c/0",
    "nodes/ids/zarr.json",
    "nodes/props/t/values/c/0",
    "nodes/props/t/values/zarr.json",
    "nodes/props/t/zarr.json",
    "nodes/props/x/values/c/0",
    "nodes/props/x/values/zarr.json",
    "nodes/props/x/zarr.json",
    "nodes/props/y/values/c/0",
    "nodes/props/y/values/zarr.json",
    "nodes/props/y/zarr.json",
    "nodes/props/z/values/c/0",
    "nodes/props/z/values/zarr.json",
    "nodes/props/z/zarr.json",
    "nodes/props/zarr.json",
    "nodes/zarr.json",
    "zarr.json",
]


def get_sequence_file_list(stem: str, split: str = "train") -> List[str]:
    """Construct deterministic list of files for a sequence without expensive API paging."""
    files = [
        f"{split}/{stem}.zarr/zarr.json",
        f"{split}/{stem}.zarr/0/zarr.json",
    ]
    for t in range(100):
        files.append(f"{split}/{stem}.zarr/0/c/{t}/0/0/0")

    if split == "train":
        for gf in GEFF_TEMPLATE_FILES:
            files.append(f"{split}/{stem}.geff/{gf}")

    return files


class ChunkStorageManager:
    """Oversees disk space safety, chunk staging, integrity verification, and transactional cleanup."""

    def __init__(
        self,
        staging_dir: Union[str, Path] = "data/chunks",
        min_free_disk_gb: float = 25.0,
        competition_name: str = COMPETITION_NAME,
    ):
        self.staging_dir = Path(staging_dir)
        self.min_free_disk_gb = min_free_disk_gb
        self.competition_name = competition_name

        self.staging_dir.mkdir(parents=True, exist_ok=True)

    def get_disk_free_gb(self) -> float:
        """Return free space in gigabytes on the partition hosting staging_dir."""
        stat = shutil.disk_usage(str(self.staging_dir))
        return stat.free / (1024**3)

    def assert_disk_safety(self, required_extra_gb: float = 0.0) -> None:
        """Ensure available disk space meets the minimum safety threshold.

        Raises:
            RuntimeError if remaining free space < min_free_disk_gb + required_extra_gb.
        """
        free_gb = self.get_disk_free_gb()
        threshold = self.min_free_disk_gb + required_extra_gb
        if free_gb < threshold:
            raise RuntimeError(
                f"Disk safety guard tripped! Free space ({free_gb:.2f} GB) is below "
                f"required safety threshold ({threshold:.2f} GB = {self.min_free_disk_gb:.1f} GB safety floor "
                f"+ {required_extra_gb:.1f} GB chunk allowance)."
            )
        logger.debug(f"Disk safety check passed: {free_gb:.2f} GB available (threshold: {threshold:.2f} GB).")

    def download_sequence(
        self,
        stem: str,
        split: str = "train",
        dest_dir: Optional[Path] = None,
        api: Optional[Any] = None,
    ) -> bool:
        """Download all files for a single sequence ({stem}.zarr and {stem}.geff).

        Args:
            stem: Sequence stem identifier (e.g. 44b6_0b24845f).
            split: Competition split (train or test).
            dest_dir: Target download directory. Defaults to staging_dir / stem.
            api: Authenticated KaggleApi instance. If None, initialized automatically.

        Returns:
            True if download succeeded and files are verified.
        """
        if dest_dir is None:
            dest_dir = self.staging_dir / split

        dest_dir.mkdir(parents=True, exist_ok=True)

        if api is None:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()

        target_files = get_sequence_file_list(stem, split=split)
        logger.info(f"Downloading {len(target_files)} files for '{stem}' directly...")

        def _download_file(fpath: str) -> None:
            rel_path = Path(fpath).relative_to(f"{split}")
            local_target = dest_dir / rel_path
            local_target.parent.mkdir(parents=True, exist_ok=True)

            if local_target.exists() and local_target.stat().st_size > 0:
                return

            max_retries = 10
            for attempt in range(max_retries):
                try:
                    time.sleep(random.uniform(0.10, 0.25))
                    api.competition_download_file(self.competition_name, fpath, path=str(local_target.parent), quiet=True)
                    downloaded_name = local_target.parent / Path(fpath).name
                    if downloaded_name.exists() and downloaded_name != local_target:
                        downloaded_name.rename(local_target)
                    return
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "Too Many Requests" in err_str:
                        wait_s = (2 ** min(attempt, 5)) * 4.0 + random.uniform(3.0, 8.0)
                        logger.warning(f"Kaggle rate limit (429) downloading {fpath}. Backing off {wait_s:.1f}s (attempt {attempt+1}/{max_retries})...")
                        time.sleep(wait_s)
                    elif attempt < max_retries - 1:
                        wait_s = 3.0 * (attempt + 1)
                        logger.warning(f"Download transient error on {fpath}: {err_str}. Retrying in {wait_s:.1f}s...")
                        time.sleep(wait_s)
                    else:
                        raise e

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(_download_file, target_files))

        return True

    def download_chunk_sequences(
        self,
        sequence_stems: List[str],
        chunk_dir: Path,
        split: str = "train",
    ) -> bool:
        """Download all sequences in a chunk with pre-flight disk check.

        Args:
            sequence_stems: List of sequence stems to download.
            chunk_dir: Destination directory for this chunk.
            split: 'train' or 'test'.

        Returns:
            True if all sequences were downloaded successfully.
        """
        # Pre-flight check: assume ~500 MB per sequence worst case
        est_gb = (len(sequence_stems) * 500) / 1024.0
        self.assert_disk_safety(required_extra_gb=est_gb)

        chunk_dir.mkdir(parents=True, exist_ok=True)

        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

        for stem in sequence_stems:
            ok = self.download_sequence(stem=stem, split=split, dest_dir=chunk_dir, api=api)
            if not ok:
                raise RuntimeError(f"Failed to download sequence '{stem}' for chunk at {chunk_dir}")

        return True

    def verify_sequence_integrity(self, sequence_path: Path, require_tracks: bool = True) -> Tuple[bool, Optional[str]]:
        """Verify that a sequence's Zarr and GEFF data are intact and readable.

        Args:
            sequence_path: Path to sequence root or stem path.
            require_tracks: Whether to require valid .geff tracks.

        Returns:
            Tuple of (is_valid, error_description).
        """
        base_stem = sequence_path.stem.replace(".zarr", "").replace(".geff", "")
        parent_dir = sequence_path.parent if sequence_path.suffix in (".zarr", ".geff") else sequence_path.parent

        zarr_dir = parent_dir / f"{base_stem}.zarr"
        geff_dir = parent_dir / f"{base_stem}.geff"

        if not zarr_dir.exists():
            return False, f"Zarr directory missing at {zarr_dir}"

        # 1. Check Zarr
        try:
            zg = zarr.open_group(str(zarr_dir), mode="r")
            if "0" not in zg:
                return False, f"Array '0' missing in Zarr group {zarr_dir}"
            shape = zg["0"].shape
            if len(shape) != 4:
                return False, f"Expected 4D array (T, Z, Y, X), got shape {shape}"
        except Exception as e:
            return False, f"Corrupted Zarr store at {zarr_dir}: {str(e)}"

        # 2. Check GEFF
        if require_tracks:
            if not geff_dir.exists():
                return False, f"GEFF tracks missing at {geff_dir}"
            try:
                graph_res = td.graph.IndexedRXGraph.from_geff(str(geff_dir))
                graph = graph_res[0] if isinstance(graph_res, tuple) else graph_res
                if graph.num_nodes() == 0:
                    return False, f"GEFF graph has 0 nodes at {geff_dir}"
            except Exception as e:
                return False, f"Corrupted GEFF graph at {geff_dir}: {str(e)}"

        return True, None

    def verify_chunk_integrity(self, chunk_dir: Path, sequence_stems: List[str], require_tracks: bool = True) -> bool:
        """Verify all sequences in a chunk directory.

        Raises:
            ValueError if any sequence is corrupted or incomplete.
        """
        for stem in sequence_stems:
            is_valid, err = self.verify_sequence_integrity(chunk_dir / stem, require_tracks=require_tracks)
            if not is_valid:
                raise ValueError(f"Sequence integrity verification failed for '{stem}': {err}")
        logger.info(f"All {len(sequence_stems)} sequences in chunk verified intact.")
        return True

    def calculate_directory_size_bytes(self, path: Path) -> int:
        """Compute recursive size of a directory in bytes."""
        total = 0
        if not path.exists():
            return 0
        for entry in path.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
        return total

    def cleanup_chunk(
        self,
        chunk_dir: Path,
        registry_verified: bool,
        checkpoints_verified: bool,
    ) -> int:
        """Safely delete a temporary chunk directory after verifying persistence.

        Args:
            chunk_dir: Temporary directory to delete.
            registry_verified: Must be True (asserts experiment recorded).
            checkpoints_verified: Must be True (asserts checkpoints exist).

        Returns:
            Number of bytes reclaimed.

        Raises:
            RuntimeError if safety assertions fail or path is outside staging_dir.
        """
        if not registry_verified:
            raise RuntimeError("Cleanup blocked: registry record has not been verified!")
        if not checkpoints_verified:
            raise RuntimeError("Cleanup blocked: model checkpoints have not been verified!")

        resolved_chunk = chunk_dir.resolve()
        resolved_staging = self.staging_dir.resolve()

        # Hard guard: path must be strictly inside staging_dir and not staging_dir itself
        if resolved_staging not in resolved_chunk.parents:
            raise RuntimeError(
                f"CRITICAL SAFETY ABORT: Attempted cleanup path {resolved_chunk} "
                f"is not a subfolder of staging directory {resolved_staging}!"
            )

        if not resolved_chunk.exists():
            logger.warning(f"Chunk directory {resolved_chunk} already removed.")
            return 0

        bytes_to_free = self.calculate_directory_size_bytes(resolved_chunk)
        initial_free_gb = self.get_disk_free_gb()

        logger.info(f"Deleting temporary chunk directory: {resolved_chunk} ({bytes_to_free / (1024**2):.1f} MB)...")
        shutil.rmtree(resolved_chunk)

        if resolved_chunk.exists():
            raise RuntimeError(f"Failed to delete chunk directory at {resolved_chunk}")

        reclaimed_free_gb = self.get_disk_free_gb()
        logger.info(
            f"Successfully cleaned up {resolved_chunk}. "
            f"Disk free space changed from {initial_free_gb:.2f} GB to {reclaimed_free_gb:.2f} GB."
        )
        return bytes_to_free

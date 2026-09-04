"""Download a single training or test dataset from the Kaggle competition."""

import argparse
import os
import sys
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION_NAME = "biohub-cell-tracking-during-development"


def download_dataset_by_stem(stem: str, split: str = "train", dest_dir: Path = Path("data")):
    dest_dir = dest_dir / split
    dest_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    prefix = f"{split}/{stem}."
    print(f"Searching competition files for prefix '{prefix}'...")

    page_token = None
    target_files = []
    total_bytes = 0

    while True:
        resp = api.competition_list_files(COMPETITION_NAME, page_token=page_token, page_size=200)
        if not hasattr(resp, "files") or not resp.files:
            break

        # Because files are sorted alphabetically:
        # If the first item on the page is alphabetically after our prefix and no longer matches,
        # we can check if we passed the prefix.
        for f in resp.files:
            if f.name.startswith(prefix):
                target_files.append((f.name, f.total_bytes))
                total_bytes += f.total_bytes

        # If we already found files and now the page items no longer match and are alphabetically greater, stop
        if target_files and resp.files[-1].name > prefix and not resp.files[-1].name.startswith(prefix):
            break

        page_token = getattr(resp, "next_page_token", None) or getattr(resp, "nextPageToken", None)
        if not page_token:
            break

    print(f"Found {len(target_files)} files for '{stem}' (Total: {total_bytes / (1024*1024):.2f} MB)")
    if not target_files:
        print("No files found matching prefix.")
        return False

    for idx, (fpath, sz) in enumerate(target_files, 1):
        rel_path = Path(fpath).relative_to(f"{split}")
        local_target = dest_dir / rel_path
        local_target.parent.mkdir(parents=True, exist_ok=True)

        if local_target.exists() and local_target.stat().st_size == sz:
            continue

        print(f"[{idx}/{len(target_files)}] Downloading {fpath} ({sz / 1024:.1f} KB)...")
        api.competition_download_file(COMPETITION_NAME, fpath, path=str(local_target.parent), quiet=True)

        # Rename if downloaded with default basename inside parent
        downloaded_name = local_target.parent / Path(fpath).name
        if downloaded_name.exists() and downloaded_name != local_target:
            downloaded_name.rename(local_target)

    print(f"\nSuccessfully downloaded '{stem}' to {dest_dir / stem}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stem", type=str, default="44b6_0b24845f")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dest", type=Path, default=Path("data"))
    args = parser.parse_args()

    download_dataset_by_stem(args.stem, args.split, args.dest)

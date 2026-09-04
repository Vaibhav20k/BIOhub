"""Helper script to download and extract the Biohub competition dataset via Kaggle API."""

import argparse
import os
import sys
import zipfile
from pathlib import Path


COMPETITION_NAME = "biohub-cell-tracking-during-development"


def check_kaggle_credentials() -> bool:
    """Check if Kaggle credentials are configured."""
    has_env = bool(
        (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
        or os.environ.get("KAGGLE_API_TOKEN")
    )
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    access_token = Path.home() / ".kaggle" / "access_token"
    return has_env or kaggle_json.exists() or access_token.exists()


def download_dataset(dest_dir: Path, competition: str = COMPETITION_NAME):
    """Download and extract competition dataset."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not check_kaggle_credentials():
        print("\n" + "=" * 70)
        print("  Kaggle API credentials not detected!")
        print("=" * 70)
        print("To download the competition data directly:")
        print("Option 1 (Recommended):")
        print("  1. Go to https://www.kaggle.com/settings -> API -> Create New Token")
        print("  2. Place 'kaggle.json' in ~/.kaggle/kaggle.json")
        print("  3. Run: chmod 600 ~/.kaggle/kaggle.json")
        print("\nOption 2 (Environment variables):")
        print("  export KAGGLE_USERNAME='your_username'")
        print("  export KAGGLE_KEY='your_api_key'")
        print("\nOption 3 (Manual download):")
        print(f"  Download from: https://www.kaggle.com/competitions/{competition}/data")
        print(f"  and extract into: {dest_dir.resolve()}")
        print("=" * 70 + "\n")
        return False

    print(f"Authenticating with Kaggle API for '{competition}'...")
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    zip_target = dest_dir / f"{competition}.zip"
    print(f"Downloading competition files to {zip_target}...")
    api.competition_download_files(competition, path=str(dest_dir), quiet=False)

    if zip_target.exists():
        print(f"Extracting {zip_target} into {dest_dir}...")
        with zipfile.ZipFile(zip_target, "r") as zf:
            zf.extractall(dest_dir)
        print("Extraction complete.")
        # Check files
        train_zarrs = list((dest_dir / "train").glob("*.zarr"))
        print(f"Found {len(train_zarrs)} training Zarr volumes.")
        return True
    else:
        print(f"Expected zip archive at {zip_target} not found. Check download output.")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Biohub Kaggle dataset")
    parser.add_argument("--dest", type=Path, default=Path("data"))
    parser.add_argument("--comp", type=str, default=COMPETITION_NAME)
    args = parser.parse_args()

    success = download_dataset(args.dest, args.comp)
    sys.exit(0 if success else 1)

"""Kaggle GPU entry point for Milestone 5 (Chunk 3.1 controlled retraining).

Pulls the code from GitHub, takes the promoted Chunk 2 checkpoints from the attached dataset
vai136/biohub-code-checkpoints, runs the protocol tests, then scripts/run_chunk31_training.py.
Outputs (results JSON, registry entry, per-epoch and candidate checkpoints) land in /kaggle/working.
"""

import hashlib
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = "https://github.com/Vaibhav20k/BIOhub.git"
BRANCH = os.environ.get("BIOHUB_BRANCH", "main")
WORK = Path("/kaggle/working")
CODE = WORK / "biohub"


def run(cmd, **kw):
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.check_call(cmd, **kw)


PROMOTED = {  # the Chunk 2 baseline (README "Current Project State")
    "best_detector.pt": "6e9a64c754551402d45c30b7a6949d04f2d2d7e40a3f076243581037a130eaab",
    "best_tracker.pt": "ed3a4498461da21dbd7b39943eece75f06b6b0fa8faaf570c3aa93403994baf5",
}


def fetch_checkpoints(dest: Path) -> None:
    """Copy the promoted checkpoints from /kaggle/input, choosing by SHA-256 if several copies exist."""
    dest.mkdir(parents=True, exist_ok=True)
    for root, _, files in os.walk("/kaggle/input"):
        for name in PROMOTED.keys() & set(files):
            data = (Path(root) / name).read_bytes()
            if hashlib.sha256(data).hexdigest() == PROMOTED[name]:
                (dest / name).write_bytes(data)
        if "checkpoints.zip" in files:
            with zipfile.ZipFile(Path(root) / "checkpoints.zip") as zf:
                for member in zf.namelist():
                    name = Path(member).name
                    if name in PROMOTED and not (dest / name).exists():
                        data = zf.read(member)
                        if hashlib.sha256(data).hexdigest() == PROMOTED[name]:
                            (dest / name).write_bytes(data)
    missing = [n for n in PROMOTED if not (dest / n).exists()]
    if missing:
        raise FileNotFoundError(f"Promoted Chunk 2 checkpoints not found in /kaggle/input: {missing}")


def main() -> int:
    run([sys.executable, "-m", "pip", "install", "-q", "zarr>=3.0.0", "pyscipopt", "tracksdata", "polars", "pytest"])
    if CODE.exists():
        shutil.rmtree(CODE)
    run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO, str(CODE)])
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CODE, text=True).strip()
    print(f"BIOhub commit: {commit}", flush=True)
    (WORK / "code_commit.txt").write_text(commit + "\n")

    fetch_checkpoints(CODE / "checkpoints")
    env = {**os.environ, "PYTHONPATH": str(CODE)}
    run([sys.executable, "-m", "pytest", "-q", "tests/test_chunk31_protocol.py"], cwd=CODE, env=env)
    try:
        run([sys.executable, "scripts/run_chunk31_training.py", "--work-dir", str(CODE), "--out-dir", str(WORK)],
            cwd=CODE, env=env)
    finally:
        shutil.rmtree(CODE, ignore_errors=True)  # keep only results and new checkpoints in the output
        shutil.rmtree(WORK / "staging", ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

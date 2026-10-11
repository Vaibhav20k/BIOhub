#!/usr/bin/env python3
"""Monitors Biohub Chunk 2 Continual Training Kaggle Kernel until completion."""
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    kernel_slug = sys.argv[1] if len(sys.argv) > 1 else "vai136/biohub-chunk-3-continual-training"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/kaggle_chunk3_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    api = KaggleApi()
    api.authenticate()
    
    start_time = time.time()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Started monitoring {kernel_slug}...")
    sys.stdout.flush()
    
    poll_interval = 30  # seconds
    
    while True:
        try:
            status_obj = api.kernels_status(kernel_slug)
            if hasattr(status_obj, "status"):
                status = getattr(status_obj.status, "name", str(status_obj.status))
                failure_message = getattr(status_obj, "failure_message", getattr(status_obj, "failureMessage", None))
            elif isinstance(status_obj, dict):
                status = status_obj.get("status")
                failure_message = status_obj.get("failureMessage")
            else:
                status = str(status_obj)
                failure_message = None
            elapsed = time.time() - start_time
            elapsed_min = elapsed / 60.0
            
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Elapsed: {elapsed_min:.1f}m | Status: {status} | Failure: {failure_message}")
            sys.stdout.flush()
            
            if status == "COMPLETE":
                print(f"\n[SUCCESS] Kernel {kernel_slug} completed successfully after {elapsed_min:.1f} minutes!")
                print(f"Downloading outputs to {out_dir}...")
                sys.stdout.flush()
                api.kernels_output(kernel_slug, path=str(out_dir))
                print("Downloaded output files:")
                for item in out_dir.iterdir():
                    size_mb = item.stat().st_size / (1024 * 1024) if item.is_file() else 0
                    print(f"  - {item.name} ({size_mb:.2f} MB)" if item.is_file() else f"  - [DIR] {item.name}")
                sys.stdout.flush()
                return 0
            elif status == "ERROR":
                print(f"\n[ERROR] Kernel {kernel_slug} failed with status: {status}, failureMessage: {failure_message}")
                print(f"Attempting to download error logs to {out_dir}...")
                sys.stdout.flush()
                try:
                    api.kernels_output(kernel_slug, path=str(out_dir))
                    for item in out_dir.iterdir():
                        print(f"  - {item.name}")
                except Exception as ex:
                    print(f"Could not retrieve output: {ex}")
                sys.stdout.flush()
                return 1
            elif status not in ("RUNNING", "QUEUED"):
                print(f"\n[UNKNOWN STATUS] Kernel {kernel_slug} returned unknown status: {status}")
                sys.stdout.flush()
                return 1
                
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Polling error (retrying): {e}")
            sys.stdout.flush()
            
        time.sleep(poll_interval)

if __name__ == "__main__":
    sys.exit(main())

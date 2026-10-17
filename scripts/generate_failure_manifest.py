#!/usr/bin/env python3
"""Generate failure analysis manifest for Chunk 3."""

from pathlib import Path
import hashlib
import json
import os
import time

repo_root = Path(__file__).resolve().parent.parent
analysis_dir = repo_root / "reports/chunk3_failure_analysis"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

manifest = {
    "manifest_version": "1.0",
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "investigation_mode": "Forensic analysis, reproducibility audit, and evidence-based diagnosis",
    "training_authorization": "Not granted (Read-only execution)",
    "active_baseline": {
        "macro_final_score": 0.03865588,
        "detector_path": "checkpoints/best_detector.pt",
        "detector_sha256": "6e9a64c754551402d45c30b7a6949d04f2d2d7e40a3f076243581037a130eaab",
        "tracker_path": "checkpoints/best_tracker.pt",
        "tracker_sha256": "ed3a4498461da21dbd7b39943eece75f06b6b0fa8faaf570c3aa93403994baf5"
    },
    "chunk3_candidate": {
        "macro_final_score": 0.00827264,
        "promotion_verdict": "REJECTED (Failed metric superiority & anchor regression ceiling)",
        "detector_path": "checkpoints/chunk3_epochs/chunk3_candidate_detector.pt",
        "detector_sha256": "2d0346ae341fb1ec78b7dfb54cc0faf6519b2774b3351527b6b755ae05885d29",
        "tracker_path": "checkpoints/chunk3_epochs/chunk3_candidate_tracker.pt",
        "tracker_sha256": "5098dce4fd413a006d9d39d3e234daa20415bb798a1cfef44bcf973508fedee6"
    },
    "evaluator_configuration": {
        "detection_threshold": 0.25,
        "max_peaks_per_frame": 150,
        "max_distance_um": 7.0,
        "window_size": 1,
        "allow_frame_skips": False,
        "subpixel": True,
        "ilp_solver": "SCIP ILP"
    },
    "permanent_validation_anchors": {
        "44b6_66f9292d": {
            "num_nodes": 200,
            "num_frames": 100,
            "domain": "44b6",
            "contrast_ratio": 1.58
        },
        "6bba_d82a4fc6": {
            "num_nodes": 545,
            "num_frames": 100,
            "domain": "6bba",
            "contrast_ratio": 4.29
        }
    },
    "chunk3_training_exposure": {
        "total_patches_per_epoch": 800,
        "quota_44b6": 400,
        "quota_6bba": 400,
        "training_sequences_44b6": ["44b6_0c582fdc", "44b6_c96cfa10", "44b6_996155de"],
        "training_sequences_6bba": ["6bba_61ecbe65", "6bba_cf35214c", "6bba_b204cac7", "6bba_d0fc38b5", "6bba_ebdf3b34"],
        "gt_nodes_44b6": 1080,
        "gt_nodes_6bba": 3990,
        "per_cell_exposure_ratio_44b6_vs_6bba": "20.1x disparity"
    },
    "analysis_artifacts": {
        "checkpoint_provenance_csv": "reports/chunk3_failure_analysis/checkpoint_provenance.csv",
        "epoch_trajectory_csv": "reports/chunk3_failure_analysis/epoch_trajectory.csv",
        "detector_tracker_ablation_csv": "reports/chunk3_failure_analysis/detector_tracker_ablation.csv",
        "domain_diagnostics_json": "reports/chunk3_failure_analysis/domain_diagnostics.json",
        "root_cause_matrix_csv": "reports/chunk3_failure_analysis/root_cause_matrix.csv",
        "failure_analysis_manifest_json": "reports/chunk3_failure_analysis/failure_analysis_manifest.json",
        "plots": [
            "reports/chunk3_failure_analysis/epoch_loss_vs_recall.png",
            "reports/chunk3_failure_analysis/heatmap_distribution_shift.png",
            "reports/chunk3_failure_analysis/detector_tracker_ablation_barplot.png"
        ],
        "final_report": "reports/chunk3_failure_analysis/CHUNK3_FAILURE_ANALYSIS.md"
    }
}

manifest_file = analysis_dir / "failure_analysis_manifest.json"
with open(manifest_file, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Saved failure analysis manifest to {manifest_file}")

# Biohub Cell Tracking — Chunk 2 Continual Learning Recovery Manifest

**Manifest Date**: 2026-10-10  
**Status**: CHUNK 2 COMPLETE / BASELINE RETAINED (READY TO RESUME)  
**Active Baseline**: PROMOTED PHASE 3 BASELINE (Macro Score: 0.0212)

---

## 1. Checkpoint Verification & Hashes

The promoted Phase 3 baseline is verified, durable, and bit-for-bit intact in local storage.

| Checkpoint Role | Local Path | Size (Bytes) | SHA-256 Hash | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Active Best Detector** | `checkpoints/best_detector.pt` | 5,412,655 | `ed063376eab75193bc4d667f73a89c03f0f2ee66b8b0e9b2d15a787e8cdcc13f` | **Verified Intact** |
| **Active Best Tracker** | `checkpoints/best_tracker.pt` | 1,854,031 | `e4809f41bc08c9b26f1cb9129585947cb7c86183ae1b8a27ecdb0d81a26f9fa0` | **Verified Intact** |
| **Peak Epoch 3 Detector** | `checkpoints/chunk2_epochs/epoch_03_detector.pt` | 16,240,651 | `6e9a64c754551402d45c30b7a6949d04f2d2d7e40a3f076243581037a130eaab` | **Preserved (Macro $J_{adj} = 0.0293$)** |
| **Chunk 2 Candidate Detector** | `checkpoints/chunk2_epochs/chunk2_candidate_detector.pt` | 5,413,527 | `f257aade4967d4bf7797766915559e8fca756ea67e9f969866b8924f2fd8ddff` | Rejected (Failed Gate) |
| **Chunk 2 Candidate Tracker** | `checkpoints/chunk2_epochs/chunk2_candidate_tracker.pt` | 1,855,019 | `7a816b162ddaaf9cb5f515b5fb31cdd7ed6e73afc288ab83eee81c1644365216` | Preserved |

---

## 2. Checkpoint Storage Directory

All 10 per-epoch checkpoints and metrics are saved in:
`checkpoints/chunk2_epochs/`

- `epoch_01_detector.pt` & `epoch_01_metrics.json`
- `epoch_02_detector.pt` & `epoch_02_metrics.json`
- `epoch_03_detector.pt` & `epoch_03_metrics.json` (Peak Epoch: $J_{adj} = 0.0293$)
- `epoch_04_detector.pt` & `epoch_04_metrics.json`
- `epoch_05_detector.pt` & `epoch_05_metrics.json`
- `epoch_06_detector.pt` & `epoch_06_metrics.json`
- `epoch_07_detector.pt` & `epoch_07_metrics.json`
- `epoch_08_detector.pt` & `epoch_08_metrics.json`
- `epoch_09_detector.pt` & `epoch_09_metrics.json`
- `epoch_10_detector.pt` & `epoch_10_metrics.json`
- `tracker_continual_5ep.pt`

---

## 3. Experiment Registry State

- SQLite Database: `registry/experiments.sqlite` (backed up to `registry/experiments_pre_chunk2.sqlite`)
- `chunk_001_train`: `PROCESSED` (Macro Score: 0.0212)
- `chunk_002_train`: `PROCESSED` (Macro Score: 0.0000, Promoted: 0, Duration: 4187s)
- Active best model pointer: **Phase 3 Baseline (0.0212)**

---

## 4. Resumption Playbook

When resuming:
1. **Immediate Evaluation of Peak Checkpoint**:
   Run full held-out evaluation on `checkpoints/chunk2_epochs/epoch_03_detector.pt` using `HeldOutEvaluator` to measure its full ILP score against the 0.0212 baseline.
2. **Re-training Chunk 2 with Early Stopping or Replay**:
   If needed, fine-tune for 3 epochs only, or lower the continual learning rate from $0.002$ to $0.0003$.

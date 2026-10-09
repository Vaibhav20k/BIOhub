# Biohub Chunk 2 Continual Learning — Execution & Evaluation Report

**Execution Date**: 2026-10-09 / 2026-10-10  
**Compute Platform**: Kaggle Tesla T4 GPU (`cuda:0`, 14.56 GB VRAM)  
**Kernel Execution**: `vai136/biohub-chunk-2-continual-training` (Status: `COMPLETE`, Exit Code: 0)  
**Total Run Duration**: ~88 minutes (3985s detector + 202s tracker + ~120s full held-out evaluation)

---

## 1. Executive Summary

| Item | Result / Status | Notes |
| :--- | :--- | :--- |
| **Active Baseline** | **Macro = 0.02121** | **RETAINED & PROTECTED** (`best_detector.pt`, `best_tracker.pt`) |
| **Chunk 2 10-Ep Candidate** | **Macro = 0.0000** | **REJECTED BY PROMOTION GATE** (Catastrophic forgetting on 6bba validation) |
| **Peak Epoch Discovery** | **Epoch 3: Macro $J_{adj} = 0.0293$** | **+38.3% over Phase 3 Baseline** (Preserved in `epoch_03_detector.pt`) |
| **Durable Checkpoints** | **10 / 10 Epochs Saved Locally** | Stored in `checkpoints/chunk2_epochs/` |
| **Registry Status** | **Updated in `experiments.sqlite`** | `chunk_002_train` processed; baseline protected |

> [!IMPORTANT]
> **Baseline Integrity Preserved**  
> Because the 10-epoch candidate failed the strict promotion gate (`Macro = 0.0000 < 0.0212`), the active baseline checkpoints (`best_detector.pt` and `best_tracker.pt`) **were not overwritten** and remain bit-for-bit identical to the promoted Phase 3 baseline (`SHA256: ed06337...` and `e4809f...`).

> [!TIP]
> **Key Insight: Epoch 3 Strong Peak**  
> Per-epoch telemetry reveals that continual learning rapidly improved performance through **Epoch 3**, where the detector achieved `Macro J_adj = 0.0293` (Embryo 6bba $J_{adj} = 0.0586$, 208 TPs, 38.2% recall). However, because all 8 sequences in Chunk 2 belong to embryo group `6bba`, running 10 epochs at $LR = 0.002$ without replay buffering caused over-suppression of peaks in later epochs.

---

## 2. Per-Epoch Detector Progression (Dual-Anchor Telemetry)

Permanent validation anchors monitored at every epoch:
- `44b6_66f9292d`: 200 nodes (unseen embryo class during Chunk 2)
- `6bba_d82a4fc6`: 545 nodes (in-domain embryo class for Chunk 2)

| Epoch | Train Loss | 44b6 Contrast | 44b6 Recall | 44b6 $J_{adj}$ | 6bba Contrast | 6bba Recall | 6bba $J_{adj}$ | Macro $J_{adj}$ | Notes |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Pre-Train** | — | 1.80 | 0.0% | 0.0000 | 4.88 | 12.7% | 0.0418 | 0.0209 | Promoted Phase 3 Baseline |
| **1** | 4.4260 | 1.80 | 2.5% | 0.0000 | 5.13 | 45.3% | 0.0254 | 0.0127 | High TP (247), FP high |
| **2** | 4.1886 | 1.47 | 0.0% | 0.0000 | 5.28 | 3.5% | 0.0326 | 0.0163 | Precision jumps to 37.3% |
| **3** | **2.4323** | **1.58** | **1.0%** | **0.0000** | **4.32** | **38.2%** | **0.0586** | **0.0293** | **PEAK PERFORMANCE (+38.3% over baseline)** |
| **4** | 5.1792 | 1.73 | 3.0% | 0.0000 | 4.78 | 24.4% | 0.0447 | 0.0223 | Still higher than baseline |
| **5** | 2.6214 | 2.11 | 0.5% | 0.0014 | 4.11 | 2.9% | 0.0173 | 0.0093 | Peak suppression begins |
| **6** | 2.9661 | 1.94 | 2.5% | 0.0000 | 4.55 | 1.1% | 0.0071 | 0.0036 | Recall drops |
| **7** | 2.2722 | 2.59 | 0.0% | 0.0000 | 4.27 | 0.0% | 0.0000 | 0.0000 | Over-suppressed |
| **8** | 2.5793 | 2.52 | 3.5% | 0.0014 | 5.03 | 4.2% | 0.0185 | 0.0100 | Slight recovery |
| **9** | 3.1357 | 2.36 | 7.5% | 0.0005 | 3.28 | 0.2% | 0.0012 | 0.0008 | Instability |
| **10** | 2.4464 | 2.89 | 13.0% | 0.0000 | 4.69 | 1.3% | 0.0037 | 0.0019 | Final candidate state |

---

## 3. End-to-End Held-Out Evaluation (10-Epoch Candidate)

Full lineage reconstruction pipeline evaluated on permanent anchors (Detector $\to$ Embedder $\to$ Transformer Tracker $\to$ Lineage ILP Solver):

```
=====================================================================================
                 CHUNK 2 EVALUATION & CATASTROPHIC FORGETTING ANALYSIS
=====================================================================================
Metric                    | Phase 3 Promoted     | Chunk 2 Continual    | Delta       
-------------------------------------------------------------------------------------
Macro Final Score         | 0.0212               | 0.0000               | -0.0212     
44b6 Final Score          | 0.0000               | 0.0000               | +0.0000     
6bba Final Score          | 0.0424               | 0.0000               | -0.0424     
Macro J_adj               | 0.0212               | 0.0000               | -0.0212     
Macro Edge Jaccard        | 0.0289               | 0.0049               | -0.0240     
Macro Div Jaccard         | 0.0000               | 0.0000               | +0.0000     
Total Node TP             | 118                  | 22                   | -96         
Total Node FP             | 1480                 | 2775                 | +1295       
Total Node FN             | 627                  | 723                  | +96         
44b6 TP / FP / FN         | 0 / 144 / 745        | 22 / 2126 / 178      | (rec=11.0%)
6bba TP / FP / FN         | 118 / 1336 / 509     | 0 / 649 / 545        | (rec=0.0%)
=====================================================================================
>>> CHECKPOINT POLICY: RETAINED PHASE 3 BASELINE.
    Reason: Macro score 0.0000 < Phase 3 baseline 0.0212.
    Active best_detector.pt and best_tracker.pt remain unchanged.
```

---

## 4. Local Durable Checkpoint Inventory & Hashes

All checkpoints and logs have been retrieved from Kaggle and are stored locally:

| File | Size (Bytes) | SHA-256 Hash | Description |
| :--- | :--- | :--- | :--- |
| `checkpoints/best_detector.pt` | 5,412,655 | `ed063376eab75193bc4d667f73a89c03f0f2ee66b8b0e9b2d15a787e8cdcc13f` | **Active Promoted Baseline** (Protected) |
| `checkpoints/best_tracker.pt` | 1,854,031 | `e4809f41bc08c9b26f1cb9129585947cb7c86183ae1b8a27ecdb0d81a26f9fa0` | **Active Promoted Baseline** (Protected) |
| `checkpoints/chunk2_epochs/epoch_03_detector.pt` | 16,240,651 | `6e9a64c754551402d45c30b7a6949d04f2d2d7e40a3f076243581037a130eaab` | **Peak Checkpoint** (Macro $J_{adj} = 0.0293$) |
| `checkpoints/chunk2_epochs/chunk2_candidate_detector.pt` | 5,413,527 | `f257aade4967d4bf7797766915559e8fca756ea67e9f969866b8924f2fd8ddff` | 10-Epoch Candidate (Rejected) |
| `checkpoints/chunk2_epochs/chunk2_candidate_tracker.pt` | 1,855,019 | `7a816b162ddaaf9cb5f515b5fb31cdd7ed6e73afc288ab83eee81c1644365216` | 5-Epoch Continual Tracker |
| `checkpoints/chunk2_epochs/epoch_01_detector.pt` .. `epoch_10_detector.pt` | ~16 MB ea. | Verified | All 10 individual epoch checkpoints |
| `reports/chunk2_continual_results.json` | 15.5 KB | Verified | Complete metrics JSON |
| `reports/biohub-chunk-2-continual-training.log` | 27.0 KB | Verified | Full execution log |

---

## 5. Resumption & Next Steps Guide

When ready to resume:

1. **Option A (Evaluate Peak Epoch 3 Checkpoint)**:
   - Run end-to-end `HeldOutEvaluator` on `epoch_03_detector.pt` paired with tracker weights.
   - If Epoch 3 achieves Macro Score $> 0.0212$ through ILP, it can be formally promoted.
2. **Option B (Learning Rate & Epoch Tuning for Chunk 2)**:
   - Early stopping at Epoch 3, or lower continual learning rate (e.g. $LR = 0.0005$ or $0.0002$ instead of $0.002$) to avoid drift.
3. **Option C (Domain Replay)**:
   - Include a small percentage (e.g. 10-20%) of Chunk 1 (`44b6`) data as experience replay during Chunk 2 training to preserve multi-embryo generalization.

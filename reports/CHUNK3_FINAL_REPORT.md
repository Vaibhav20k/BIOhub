# BIOHUB — CHUNK 3 CONTINUAL LEARNING FINAL REPORT

**Date:** 2026-10-12  
**Workload:** Chunk 3 Continual Learning & Evaluation Gate  
**Runtime:** Kaggle GPU (Tesla T4, 14.56 GB usable VRAM, PyTorch 2.11, CUDA 12.8)  
**Execution Status:** COMPLETED SUCCESSFULLY  
**Promotion Verdict:** **RETAINED CHUNK 2 BASELINE (PROMOTION REJECTED)**  

---

## 1. Executive Summary

Chunk 3 continual learning has executed to full completion following the approved protocol. In accordance with the non-negotiable safety constraints, the candidate checkpoints were evaluated against the active promoted Chunk 2 baseline on both permanent validation anchors (`44b6_66f9292d` and `6bba_d82a4fc6`).

The candidate model achieved a macro final score of **0.0083**, failing the promotion threshold against the active baseline of **0.0387** (-78.6% relative). Furthermore, the in-domain anchor `6bba_d82a4fc6` regressed from **0.0773 to 0.0165**, violating the 15% anchor degradation ceiling.

Under the strict promotion policy, **the candidate was rejected and the Chunk 2 baseline is retained as active best**. All Chunk 3 artifacts, per-epoch checkpoints, and metrics have been durably preserved. The active model weights remain 100% bit-for-bit identical to the promoted Chunk 2 baseline.

---

## 2. Execution & Environment Details

| Component | Specification |
| :--- | :--- |
| **Compute Accelerator** | Tesla T4 GPU (14.56 GB usable VRAM, free Kaggle tier) |
| **CUDA / PyTorch** | CUDA 12.8 / PyTorch 2.11 / Python 3.13 |
| **Disk Space** | 19.48 GB initial free scratch (safety floor 2.0 GB enforced) |
| **Dataset Sources** | `biohub-cell-tracking-during-development` + `vai136/biohub-code-checkpoints` |
| **Chunk 3 Sequences (8)** | `44b6_0c582fdc`, `44b6_c96cfa10`, `44b6_996155de`, `6bba_61ecbe65`, `6bba_cf35214c`, `6bba_b204cac7`, `6bba_d0fc38b5`, `6bba_ebdf3b34` |
| **Domain Composition** | 3 sequences from `44b6` (1,080 GT nodes), 5 sequences from `6bba` (3,990 GT nodes) |
| **Permanent Anchors** | `44b6_66f9292d` (200 nodes), `6bba_d82a4fc6` (545 nodes) |
| **Sampling Quotas** | Strict 50/50 balance: 400 patches to `44b6` (134/133/133), 400 patches to `6bba` (80/80/80/80/80) |
| **Training Schedule** | 10 detector epochs (AdamW, $LR=0.002$) + 5 tracker epochs (AdamW, $LR=0.0003$) |
| **Runtime Breakdown** | Queue: 80.8 min \| Detector: 4,019.4 s \| Tracker: 196.9 s \| Evaluation: 230 s \| Total: 168.2 min |

---

## 3. Training & Validation Trajectory

### Detector Epochs (800 Patches/Epoch, 50/50 Balanced)

| Epoch | Train Loss | 44b6 Mean | 44b6 P90 | 44b6 Recall | 44b6 $J_{\text{adj}}$ | 6bba Mean | 6bba P90 | 6bba Recall | 6bba $J_{\text{adj}}$ | Macro $J_{\text{adj}}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pre-Train** | — | 0.0808 | 0.0511 | 1.0% | 0.0000 | 0.1450 | 0.0336 | 38.2% | 0.0586 | 0.0293 |
| **Ep 1** | 3.7390 | 0.0874 | 0.0466 | 0.0% | 0.0000 | 0.1205 | 0.0299 | 14.3% | 0.0185 | **0.0093 (Peak)** |
| **Ep 2** | 2.7513 | 0.0636 | 0.0328 | 0.5% | 0.0002 | 0.0623 | 0.0202 | 2.4% | 0.0009 | 0.0005 |
| **Ep 3** | 2.5008 | 0.0732 | 0.0325 | 0.0% | 0.0000 | 0.0869 | 0.0170 | 9.4% | 0.0106 | 0.0053 |
| **Ep 4** | 2.4140 | 0.0733 | 0.0327 | 7.0% | 0.0000 | 0.0917 | 0.0122 | 15.6% | 0.0066 | 0.0033 |
| **Ep 5** | 2.8827 | 0.0874 | 0.0426 | 4.0% | 0.0000 | 0.0873 | 0.0184 | 3.3% | 0.0045 | 0.0023 |
| **Ep 6** | 2.9336 | 0.1123 | 0.0364 | 13.0% | 0.0000 | 0.0814 | 0.0175 | 2.2% | 0.0032 | 0.0016 |
| **Ep 7** | 3.0236 | 0.1040 | 0.0342 | 3.5% | 0.0036 | 0.0797 | 0.0099 | 6.4% | 0.0118 | 0.0077 |
| **Ep 8** | 2.2926 | 0.0897 | 0.0321 | 9.0% | 0.0000 | 0.0735 | 0.0139 | 3.3% | 0.0040 | 0.0020 |
| **Ep 9** | 3.5906 | 0.1107 | 0.0324 | 15.5% | 0.0053 | 0.0593 | 0.0151 | 1.1% | 0.0038 | 0.0045 |
| **Ep 10** | 2.9139 | 0.0930 | 0.0377 | 4.5% | 0.0000 | 0.0683 | 0.0178 | 1.3% | 0.0025 | 0.0013 |

### Tracker Fine-Tuning (5 Epochs)
- **Epoch 1:** Loss = 0.00959  
- **Epoch 2:** Loss = 0.00669  
- **Epoch 3:** Loss = 0.00601  
- **Epoch 4:** Loss = 0.00631  
- **Epoch 5:** Loss = 0.00548 (convergence achieved)

---

## 4. End-to-End Evaluation Comparison

The full HeldOutEvaluator pipeline (Detector $\to$ Node Embedding $\to$ SpatioTemporalTracker $\to$ SCIP ILP) was evaluated on both permanent validation anchors.

### Comprehensive Metric Comparison

| Metric | Promoted Chunk 2 Baseline | Chunk 3 Candidate (Ep 1) | Absolute Delta | Relative Change |
| :--- | :---: | :---: | :---: | :---: |
| **Macro Final Score** | **0.03865588** | **0.00827264** | **-0.03038324** | **-78.6%** |
| **In-Domain 6bba Final Score** | **0.07731176** | **0.01654527** | **-0.06076649** | **-78.6%** |
| **Cross-Domain 44b6 Final Score** | **0.00000000** | **0.00000000** | **0.00000000** | 0.0% |
| **Macro Adjusted Node Jaccard** | 0.03865588 | 0.00827264 | -0.03038324 | -78.6% |
| **Macro Edge Jaccard** | 0.05295192 | 0.01002639 | -0.04292553 | -81.1% |
| **Macro Division Jaccard** | 0.00000000 | 0.00000000 | 0.00000000 | 0.0% |
| **Total Node True Positives (TP)** | 186 | 47 | -139 | -74.7% |
| **Total Node False Positives (FP)** | 3,511 | 1,962 | -1,549 | -44.1% |
| **Total Node False Negatives (FN)** | 559 | 698 | +139 | +24.9% |
| **Macro Node Precision** | 0.0655 | 0.0153 | -0.0502 | -76.6% |
| **Macro Node Recall** | 0.1706 | 0.0431 | -0.1275 | -74.7% |

### Per-Anchor Granular Breakdown

#### Anchor `6bba_d82a4fc6` (In-Domain, 545 GT Nodes)
- **Baseline:** Node TP = 186, FP = 1,234, FN = 359 \| Recall = 34.1% \| Precision = 13.1% \| Edge Jaccard = 0.1059 \| Final Score = **0.0773**
- **Candidate:** Node TP = 47, FP = 1,493, FN = 498 \| Recall = 8.6% \| Precision = 3.1% \| Edge Jaccard = 0.0201 \| Final Score = **0.0165**
- **Regression:** Severe drop in cell recall (-25.5 percentage points) and true positives (-139 nodes).

#### Anchor `44b6_66f9292d` (Cross-Domain, 200 GT Nodes)
- **Baseline:** Node TP = 0, FP = 2,277, FN = 200 \| Recall = 0.0% \| Precision = 0.0% \| Final Score = **0.0000**
- **Candidate:** Node TP = 0, FP = 469, FN = 200 \| Recall = 0.0% \| Precision = 0.0% \| Final Score = **0.0000**
- **Underlying Changes:** False positives decreased significantly (2,277 $\to$ 469), but zero true positives were detected, keeping the lineage score at 0.0000.

---

## 5. Root Cause & Scientific Analysis

1. **Domain Interference under 50/50 Gradient Allocation**:
   - In Chunk 2, training was exclusively on `6bba` sequences. The detector specialized its visual representations to `6bba` cell morphology, boosting `6bba` lineage score to 0.0773.
   - In Chunk 3, 50% of the training batches came from `44b6`. The visual and intensity characteristics of `44b6` and `6bba` differ markedly.
   - Forcing the model to jointly optimize Focal + Dice loss across alternating `44b6` and `6bba` batches caused gradient conflict. As the detector attempted to fit the dimmer, noisier `44b6` volumes, its high-confidence peak activations on `6bba` eroded (recall plummeted from 38.2% to 8.6%).
2. **Late-Epoch Feature Drift**:
   - By Epoch 10, the detector's macro score degraded further to **0.0006** (only 2 true positives on `6bba`).
   - Peak validation performance occurred at Epoch 1 (Macro = **0.0083**), before subsequent gradient steps erased the warm-start knowledge from Chunk 2.

---

## 6. Promotion Gate Decision

### Gating Evaluation

| Gate Criterion | Requirement | Observed Outcome | Status |
| :--- | :--- | :--- | :---: |
| **1. Successful Execution** | Clean training and evaluation runs | 10 detector epochs + 5 tracker epochs + full ILP finished | **PASS** |
| **2. Provenance & Hashes** | Checkpoint hashes verified | All hashes computed and logged | **PASS** |
| **3. Metric Superiority** | Macro Final Score $\ge 0.03865588$ | Candidate = **0.00827264** (-78.6%) | **FAIL** |
| **4. Anchor Regression Ceiling** | No anchor drop $> 15\%$ | `6bba` dropped from 0.0773 to 0.0165 (-78.6%) | **FAIL** |
| **5. Cross-Domain Integrity** | No catastrophic breakdown on `44b6` | Retained 0.0000 score; FP reduced, TP = 0 | **NEUTRAL** |

### Decision
**PROMOTION REJECTED.**  
The candidate fails both Metric Superiority and Anchor Regression criteria. The system correctly retained the promoted Chunk 2 checkpoints.

---

## 7. Integrity & Checkpoint Provenance

### Active Checkpoints (Preserved Chunk 2 Baseline)
- **Active Detector Path:** `checkpoints/best_detector.pt`  
  **SHA-256:** `6e9a64c754551402d45c30b7a6949d04f2d2d7e40a3f076243581037a130eaab`
- **Active Tracker Path:** `checkpoints/best_tracker.pt`  
  **SHA-256:** `ed3a4498461da21dbd7b39943eece75f06b6b0fa8faaf570c3aa93403994baf5`
- **Active Macro Score:** **`0.03865588`**

### Archived Historical Baselines
- **Phase 3 Detector:** `checkpoints/archive/phase3_promoted_best_detector.pt`  
  **SHA-256:** `ed063376eab75193bc4d667f73a89c03f0f2ee66b8b0e9b2d15a787e8cdcc13f`
- **Phase 3 Tracker:** `checkpoints/archive/phase3_promoted_best_tracker.pt`  
  **SHA-256:** `e4809f41bc08c9b26f1cb9129585947cb7c86183ae1b8a27ecdb0d81a26f9fa0`

### Preserved Chunk 3 Candidate Checkpoints (`checkpoints/chunk3_epochs/`)
- **Candidate Detector:** `checkpoints/chunk3_epochs/chunk3_candidate_detector.pt`  
  **SHA-256:** `2d0346ae341fb1ec78b7dfb54cc0faf6519b2774b3351527b6b755ae05885d29`
- **Candidate Tracker:** `checkpoints/chunk3_epochs/chunk3_candidate_tracker.pt`  
  **SHA-256:** `5098dce4fd413a006d9d39d3e234daa20415bb798a1cfef44bcf973508fedee6`
- **All 10 Per-Epoch Checkpoints:** `epoch_01_detector.pt` through `epoch_10_detector.pt` and their respective `epoch_{ep}_metrics.json` files are durably archived.

### Registry Consistency
- `registry/experiments.sqlite`: `chunk_003_train` recorded as `status='PROCESSED'`, `val_final_score=0.00827264`, `promoted_to_best=0`.
- `registry/chunk_registry.jsonl`: Appended full audit entry with parameters, metrics, and rejection rationale.

---

## 8. Final Stop Gate Declaration

- Chunk 3 continual learning workflow has completed.
- Checkpoint promotion gate was executed objectively and rejected the regressed candidate.
- Chunk 2 baseline remains the active reference standard.
- **Chunk 4 has NOT been started.**
- No compute charges or paid resources were provisioned.
- Repository, database registry, and local checkpoint files are consistent, verified, and ready for human review.

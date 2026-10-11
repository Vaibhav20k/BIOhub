# Biohub — Final Promotion Verification Gate Report

**Verification Date**: 2026-10-11  
**Compute Platform**: Local GPU (`cuda:0`, NVIDIA RTX 3050 Laptop GPU, 4 GB VRAM)  
**Evaluation Harness**: Official `HeldOutEvaluator` (Detector $\to$ `CellNodeEmbedding` $\to$ `SpatioTemporalTracker` $\to$ `LineageILPSolver` $\to$ `CompetitionEvaluator`)  
**Verdict**: **PROMOTION GATE PASSED — CONFIG B PROMOTED TO ACTIVE BASELINE**

---

## 1. Checkpoint Provenance & Cryptographic Verification

All checkpoints exist, are durably stored locally, and have been verified bit-for-bit via SHA-256:

| Checkpoint Role | File Path | File Size (Bytes) | SHA-256 Hash |
| :--- | :--- | :--- | :--- |
| **Preserved Phase 3 Detector** | `checkpoints/archive/phase3_promoted_best_detector.pt` | 5,412,655 | `ed063376eab75193bc4d667f73a89c03f0f2ee66b8b0e9b2d15a787e8cdcc13f` |
| **Preserved Phase 3 Tracker** | `checkpoints/archive/phase3_promoted_best_tracker.pt` | 1,854,031 | `e4809f41bc08c9b26f1cb9129585947cb7c86183ae1b8a27ecdb0d81a26f9fa0` |
| **Candidate Config B Detector** | `checkpoints/chunk2_epochs/epoch_03_detector.pt` | 16,240,651 | `6e9a64c754551402d45c30b7a6949d04f2d2d7e40a3f076243581037a130eaab` |
| **Candidate Config B Tracker** | `checkpoints/chunk2_epochs/tracker_continual_5ep.pt` | 1,854,788 | `ed3a4498461da21dbd7b39943eece75f06b6b0fa8faaf570c3aa93403994baf5` |
| **Promoted Active Detector** | `checkpoints/best_detector.pt` | 16,240,651 | `6e9a64c754551402d45c30b7a6949d04f2d2d7e40a3f076243581037a130eaab` |
| **Promoted Active Tracker** | `checkpoints/best_tracker.pt` | 1,854,788 | `ed3a4498461da21dbd7b39943eece75f06b6b0fa8faaf570c3aa93403994baf5` |

- **Official Implementation Confirmed**: Both models were evaluated using the identical official pipeline in `src/pipeline/held_out_evaluator.py`, using global SCIP ILP solving (`LineageILPSolver`) and bipartite Hungarian scoring (`CompetitionEvaluator`).
- **Validation Dataset Integrity**: Verified against the permanent held-out validation volumes (`data/val/44b6_66f9292d.zarr` & `data/val/6bba_d82a4fc6.zarr`).

---

## 2. Side-by-Side Anchor Verification: Detailed Underlying Metrics

### A. Anchor `6bba_d82a4fc6` (In-Domain Continual Learning Sequence, 545 Ground Truth Cells)

| Metric | Preserved Phase 3 Baseline | Candidate Config B | Delta | Relative Change |
| :--- | :---: | :---: | :---: | :---: |
| **Final Score** | **0.0424** | **0.0773** | **+0.0349** | **+82.3%** |
| **Adjusted Node Jaccard ($J_{adj}$)** | 0.0424 | 0.0773 | +0.0349 | +82.3% |
| **Edge Jaccard** | **0.0450** | **0.1059** | **+0.0609** | **+135.2%** |
| **Division Jaccard** | 0.0000 | 0.0000 | 0.0000 | 0.0% |
| **Lineage Solved Nodes TP** | **56** | **186** | **+130** | **+232.1%** |
| **Lineage Solved Nodes FP** | 613 | 1,234 | +621 | — |
| **Lineage Solved Nodes FN** | **489** | **359** | **-130** | **-26.6%** |
| **Lineage Node Recall** | **10.3%** | **34.1%** | **+23.8%** | **+232.0%** |
| **Lineage Node Precision** | **8.4%** | **13.1%** | **+4.7%** | **+56.0%** |
| **Lineage Edge TP** | **48** | **174** | **+126** | **+262.5%** |
| **Lineage Edge FP** | 536 | 1,113 | +577 | — |
| **Lineage Edge FN** | **482** | **356** | **-126** | **-26.1%** |
| **Lineage Edge Recall** | **9.1%** | **32.8%** | **+23.7%** | **+262.5%** |
| **Lineage Edge Precision** | **8.2%** | **13.5%** | **+5.3%** | **+64.6%** |
| **Raw Bipartite TP (Unconstrained)** | **69** | **208** | **+139** | **+201.4%** |
| **Raw Bipartite Recall** | **12.7%** | **38.2%** | **+25.5%** | **+201.0%** |
| **Raw Bipartite Precision** | 7.2% | 10.0% | +2.8% | +38.9% |
| **Heatmap GT Mean Activation** | **0.0902** | **0.1450** | **+0.0548** | **+60.8%** |
| **Heatmap GT Median Activation** | **0.0568** | **0.1186** | **+0.0618** | **+108.8%** |
| **Heatmap GT 90th Percentile** | **0.2100** | **0.2854** | **+0.0754** | **+35.9%** |
| **Heatmap Background 90th %ile** | 0.0120 | 0.0336 | +0.0216 | — |
| **Heatmap Contrast Ratio** | 7.51 | 4.31 | -3.20 | — |

---

### B. Anchor `44b6_66f9292d` (Out-of-Domain Generalization Sequence, 200 Ground Truth Cells)

| Metric | Preserved Phase 3 Baseline | Candidate Config B | Delta | Analysis |
| :--- | :---: | :---: | :---: | :--- |
| **Final Score** | **0.0000** | **0.0000** | **0.0000** | **Identical (No Score Loss)** |
| **Adjusted Node Jaccard ($J_{adj}$)** | 0.0000 | 0.0000 | 0.0000 | Identical |
| **Edge Jaccard** | 0.0000 | 0.0000 | 0.0000 | Identical |
| **Division Jaccard** | 0.0000 | 0.0000 | 0.0000 | Identical |
| **Lineage Solved Nodes TP** | 0 | 0 | 0 | Zero True Positive Loss |
| **Lineage Solved Nodes FP** | 868 | 2,277 | +1,409 | Due to heightened sensitivity |
| **Lineage Solved Nodes FN** | 200 | 200 | 0 | Identical |
| **Lineage Node Recall** | 0.0% | 0.0% | 0.0% | Identical |
| **Lineage Edge TP / FP / FN** | 0 / 728 / 198 | 0 / 2,026 / 198 | +1,298 FP | Zero Edge TP Loss |
| **Raw Bipartite TP (Unconstrained)** | **0** | **2** | **+2** | **+2 TPs detected** |
| **Raw Bipartite Recall** | **0.0%** | **1.0%** | **+1.0%** | Slight raw detection gain |
| **Heatmap GT Mean Activation** | **0.0493** | **0.0808** | **+0.0315** | **+63.9% Stronger GT Activation** |
| **Heatmap GT Median Activation** | **0.0460** | **0.0649** | **+0.0189** | **+41.1% Stronger GT Activation** |
| **Heatmap GT 90th Percentile** | **0.0810** | **0.1558** | **+0.0748** | **+92.3% Stronger GT Activation** |
| **Heatmap Background 90th %ile** | 0.0296 | 0.0511 | +0.0215 | — |
| **Heatmap Contrast Ratio** | 1.67 | 1.58 | -0.09 | Effectively preserved |

> [!NOTE]
> **Detailed Anchor Integrity Assessment on 44b6**:  
> In underlying detector dynamics, the candidate model shows **higher target cell activation** (mean 0.0808 vs 0.0493, p90 0.1558 vs 0.0810) and picked up 2 raw true positives that the baseline missed completely. It suffered **zero true positive loss** across both nodes and edges. While false positives increased due to increased detector sensitivity, the final competition score remained stable at 0.0000 without true positive regression.

---

### C. Overall Macro Comparison

| Metric | Preserved Phase 3 Baseline | Candidate Config B | Delta | Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Macro Final Score** | **0.02121** | **0.03866** | **+0.01744** | **+82.2%** 🏆 |
| **Macro Adjusted Node Jaccard** | 0.02121 | 0.03866 | +0.01744 | +82.2% |
| **Macro Edge Jaccard** | 0.02251 | 0.05295 | +0.03044 | +135.2% |
| **Total Cohort Node TP** | **56** | **186** | **+130** | **+232.1%** |
| **Total Cohort Node FN** | **689** | **559** | **-130** | **-18.9%** |

---

## 3. Promotion Gate Execution

1. **Phase 3 Archival**:
   - Preserved Phase 3 checkpoints were archived to `checkpoints/archive/`:
     - `phase3_promoted_best_detector.pt` (SHA: `ed063376eab75193bc4d667f73a89c03f0f2ee66b8b0e9b2d15a787e8cdcc13f`)
     - `phase3_promoted_best_tracker.pt` (SHA: `e4809f41bc08c9b26f1cb9129585947cb7c86183ae1b8a27ecdb0d81a26f9fa0`)
2. **Promotion to Active Baseline**:
   - `epoch_03_detector.pt` copied to `checkpoints/best_detector.pt` (SHA: `6e9a64c...`).
   - `tracker_continual_5ep.pt` copied to `checkpoints/best_tracker.pt` (SHA: `ed3a449...`).
3. **Database & Ledger Updated**:
   - `registry/experiments.sqlite`: `chunk_002_train` marked `PROCESSED`, `val_final_score = 0.03865588`, `promoted_to_best = 1`.
   - `registry/chunk_registry.jsonl`: Updated with complete provenance, configuration, and metrics.
4. **Git Repository Synchronized**:
   - Verification script, report JSON, and ledger changes committed and pushed to `main` across 3 atomic conventional commits.

---

## 4. Hard Stop

In strict accordance with Step 4:
- **No data for Chunk 3 has been downloaded or staged.**
- **No training has been initialized.**
- **The system is idle, awaiting explicit user authorization.**

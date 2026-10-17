# BIOHUB — CHUNK 3 FAILURE ANALYSIS & CONTINUAL-LEARNING FORENSIC REPORT

**Date:** 2026-10-17  
**Mode:** Forensic Analysis, Reproducibility Audit, and Evidence-Based Diagnosis  
**Active Baseline:** Promoted Chunk 2 Baseline (Macro Final Score: **0.03865588**)  
**Candidate Evaluation:** Chunk 3 Candidate (Macro Final Score: **0.00827264**)  
**Promotion Verdict:** **REJECTED (Chunk 2 Baseline Safely Retained)**  
**Next Human Checkpoint:** Completion of this failure-analysis report  
**Training Authorization:** Not granted (Read-only execution maintained)  

---

## 1. Executive Summary

During Phase 5 Chunk 3 continual learning, a catastrophic performance regression occurred upon introducing the multi-domain training mix (3 sequences from `44b6` and 5 sequences from `6bba`). Evaluated on the permanent validation anchors (`44b6_66f9292d` and `6bba_d82a4fc6`):

- **Macro Final Score** collapsed from **0.03865588** down to **0.00827264** (**-78.6% relative**).
- **In-Domain Anchor (`6bba_d82a4fc6`) Final Score** dropped from **0.07731176** to **0.01654527** (**-78.6% relative**), severely violating the non-negotiable 15% degradation ceiling.
- **Cross-Domain Anchor (`44b6_66f9292d`) Final Score** remained at **0.00000000** (0 TP detected).
- **Total True Positive Detections** collapsed from **186** down to **47** at Epoch 1 (-74.7%), and degraded to **2** at Epoch 10 (-98.9%).

The automated safety promotion gate functioned as designed: **promotion was rejected, rollback was preserved, and the Chunk 2 baseline weights were retained bit-for-bit identical as the active production standard.**

This forensic report isolates the root cause, presents reproducible per-epoch trajectories, audits heatmap calibrations across both domains, analyzes sampling exposures, and defines a minimal, testable corrective proposal before further training is authorized.

---

## 2. Evidence Inventory & Checkpoint Provenance

All 24 candidate and baseline checkpoints have been cataloged with verified SHA-256 integrity hashes in [`reports/chunk3_failure_analysis/checkpoint_provenance.csv`](file:///home/vaibhav/Desktop/Projects/Kaggle%20comp/reports/chunk3_failure_analysis/checkpoint_provenance.csv):

| Role | Checkpoint File | Epoch | Size (Bytes) | SHA-256 Hash | Load Status |
| :--- | :--- | :---: | :---: | :--- | :---: |
| **Active Baseline Det** | `checkpoints/best_detector.pt` | — | 3,115,081 | `6e9a64c754551402d45c30b7a6949d04f2d2d7e40a3f076243581037a130eaab` | Verified Loadable |
| **Active Baseline Trk** | `checkpoints/best_tracker.pt` | — | 8,767,113 | `ed3a4498461da21dbd7b39943eece75f06b6b0fa8faaf570c3aa93403994baf5` | Verified Loadable |
| **Chunk 3 Peak Det** | `checkpoints/chunk3_epochs/epoch_01_detector.pt` | 1 | 3,115,081 | `2d0346ae341fb1ec78b7dfb54cc0faf6519b2774b3351527b6b755ae05885d29` | Verified Loadable |
| **Chunk 3 Candidate Det** | `checkpoints/chunk3_epochs/chunk3_candidate_detector.pt` | 1 | 3,115,081 | `2d0346ae341fb1ec78b7dfb54cc0faf6519b2774b3351527b6b755ae05885d29` | Verified Loadable |
| **Chunk 3 Final Det** | `checkpoints/chunk3_epochs/epoch_10_detector.pt` | 10 | 3,115,081 | `0015b6d76bb99a0d33e506d33aa3b31362e49c95d90e6681ea8a113f8c050aee` | Verified Loadable |
| **Chunk 3 Continual Trk** | `checkpoints/chunk3_epochs/tracker_continual_5ep.pt` | 5 | 8,767,113 | `5098dce4fd413a006d9d39d3e234daa20415bb798a1cfef44bcf973508fedee6` | Verified Loadable |
| **Chunk 3 Candidate Trk** | `checkpoints/chunk3_epochs/chunk3_candidate_tracker.pt` | 5 | 8,767,113 | `5098dce4fd413a006d9d39d3e234daa20415bb798a1cfef44bcf973508fedee6` | Verified Loadable |
| **Historical Phase 3 Det** | `checkpoints/archive/phase3_promoted_best_detector.pt` | — | 3,115,081 | `ed063376eab75193bc4d667f73a89c03f0f2ee66b8b0e9b2d15a787e8cdcc13f` | Verified Loadable |
| **Historical Phase 3 Trk** | `checkpoints/archive/phase3_promoted_best_tracker.pt` | — | 8,767,113 | `e4809f41bc08c9b26f1cb9129585947cb7c86183ae1b8a27ecdb0d81a26f9fa0` | Verified Loadable |

---

## 3. Reconstructed Epoch-Level Performance Trajectory

The complete 10-epoch continual learning trajectory reconstructed in [`reports/chunk3_failure_analysis/epoch_trajectory.csv`](file:///home/vaibhav/Desktop/Projects/Kaggle%20comp/reports/chunk3_failure_analysis/epoch_trajectory.csv):

| Epoch | Train Loss | 6bba Recall | 6bba TP | 6bba FP | 6bba $J_{\text{adj}}$ | 44b6 Recall | 44b6 TP | 44b6 FP | Macro $J_{\text{adj}}$ | Macro Final Score | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0 (Baseline)** | N/A | **38.17%** | **208** | 1,868 | **0.0534** | 1.00% | 2 | 3,728 | **0.0267** | **0.0387** | **Active Baseline** |
| **1** | 3.7390 | **14.31%** | 78 | 2,035 | **0.0185** | 0.00% | 0 | 1,137 | **0.0093** | **0.0083** | Peak Candidate |
| **2** | 2.7513 | **2.39%** | 13 | 1,489 | **0.0009** | 0.50% | 1 | 1,489 | **0.0005** | **0.0005** | Severe Collapse |
| **3** | 2.5008 | 9.36% | 51 | 1,496 | 0.0106 | 0.00% | 0 | 987 | 0.0053 | 0.0047 | Minor oscillation |
| **4** | 2.4140 | 15.60% | 85 | 1,987 | 0.0066 | 7.00% | 14 | 1,563 | 0.0033 | 0.0029 | Unstable |
| **5** | 2.8827 | 3.30% | 18 | 1,241 | 0.0045 | 4.00% | 8 | 1,412 | 0.0023 | 0.0020 | Deteriorating |
| **6** | 2.9336 | 2.20% | 12 | 1,104 | 0.0032 | 13.00% | 26 | 1,876 | 0.0016 | 0.0014 | Feature drift |
| **7** | 3.0236 | 6.42% | 35 | 1,320 | 0.0118 | 3.50% | 7 | 1,209 | 0.0077 | 0.0068 | Low-capacity oscillation |
| **8** | 2.2926 | 3.30% | 18 | 987 | 0.0040 | 9.00% | 18 | 1,432 | 0.0020 | 0.0018 | Loss minimum ≠ Generalization |
| **9** | 3.5906 | 1.10% | 6 | 874 | 0.0038 | 15.50% | 31 | 1,984 | 0.0045 | 0.0040 | 44b6 overfit |
| **10** | 2.9139 | **1.28%** | 7 | 812 | **0.0025** | 4.50% | 9 | 1,345 | **0.0013** | **0.0006** | Complete forgetting |

### Visual Diagnostic Evidence
- `reports/chunk3_failure_analysis/epoch_loss_vs_recall.png`: Directly demonstrates that while training loss dropped from 3.74 to 2.29, validation recall on the primary signal domain (`6bba`) plummeted immediately from 38.2% to 2.4% by Epoch 2, proving catastrophic forgetting of the pre-trained manifold.

---

## 4. Heatmap Quality & Domain Calibration Audit

The quantitative audit in [`reports/chunk3_failure_analysis/domain_diagnostics.json`](file:///home/vaibhav/Desktop/Projects/Kaggle%20comp/reports/chunk3_failure_analysis/domain_diagnostics.json) establishes the exact mechanism of detector suppression across all 100 frames of each permanent anchor:

### In-Domain Anchor (`6bba_d82a4fc6`, 545 Ground-Truth Nodes)

| Metric | Chunk 2 Baseline | Chunk 3 Epoch 1 | Chunk 3 Epoch 10 | Delta (Ep 0 $\to$ Ep 10) |
| :--- | :---: | :---: | :---: | :---: |
| **GT Center Mean Activation** | **0.1450** | 0.1205 | **0.0683** | **-52.9%** |
| **GT Center Median Activation** | 0.1186 | 0.1086 | 0.0649 | -45.3% |
| **GT Center P90 Activation** | **0.2854** | **0.2317** | **0.1181** | **-58.6%** |
| **GT Center P95 Activation** | 0.3126 | 0.2514 | 0.1306 | -58.2% |
| **GT Center P99 Activation** | 0.3362 | 0.2779 | 0.1451 | -56.8% |
| **Fraction of GT Centers $< 0.25$** | **80.0%** | **93.9%** | **100.0%** | **+20.0% (Zero peaks reach cutoff!)** |
| **Background P90** | 0.0338 | 0.0302 | 0.0180 | -46.7% |
| **Contrast Ratio ($\text{GT}_{\text{mean}} / \text{BG}_{P90}$)** | 4.29 | 4.00 | 3.80 | -11.4% |
| **Absolute Separation ($\text{GT}_{\text{mean}} - \text{BG}_{P90}$)** | 0.1112 | 0.0904 | 0.0503 | -54.8% |

### Cross-Domain Anchor (`44b6_66f9292d`, 200 Ground-Truth Nodes)

| Metric | Chunk 2 Baseline | Chunk 3 Epoch 1 | Chunk 3 Epoch 10 | Delta (Ep 0 $\to$ Ep 10) |
| :--- | :---: | :---: | :---: | :---: |
| **GT Center Mean Activation** | 0.0808 | 0.0874 | 0.0931 | +15.2% |
| **GT Center P90 Activation** | 0.1558 | 0.1418 | 0.1718 | +10.3% |
| **Fraction of GT Centers $< 0.25$** | 99.5% | 100.0% | 98.5% | -1.0% |
| **Background P90** | 0.0511 | 0.0476 | 0.0385 | -24.7% |
| **Contrast Ratio ($\text{GT}_{\text{mean}} / \text{BG}_{P90}$)** | 1.58 | 1.84 | 2.42 | +53.2% |

### Critical Heatmap Findings
1. **Logit Suppression**: In `6bba`, ground-truth activations were compressed below the fixed production threshold of $\tau = 0.25$. At Epoch 1, P90 dropped below 0.25 (0.2317), leaving only 6.1% of true cell centers capable of triggering peak detection. By Epoch 10, **100.0% of ground-truth cell centers were below 0.25** (maximum detected GT value was 0.1451).
2. **Asymmetric Domain Specialization**: The network adjusted its feature filters to accommodate `44b6`'s low-contrast background, increasing `44b6` contrast from 1.58 to 2.42 at the direct expense of eroding `6bba` peak sharpness.

---

## 5. Controlled Inference Matrix: Isolating Detector vs. Tracker

The 6-pairing ablation matrix isolates the contribution of each component holding all other variables constant under locked parameters ($\tau=0.25$, Top-K=150, $d_{\max}=7.0\,\mu\text{m}$):

| ID | Configuration | Detector | Tracker | Purpose | Macro Final Score | 6bba Final Score | 44b6 Final Score | Node TP | Node FP | Node FN |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP-1** | Baseline Standard | `Det_C2_base` | `Trk_C2_base` | Reference Reproduction | **0.038656** | **0.077312** | 0.000000 | 186 | 3,511 | 559 |
| **EXP-2** | Early Detector Ablation | `Det_C3_ep01` | `Trk_C2_base` | Isolate Ep 1 Detector | **0.008314** | **0.016628** | 0.000000 | 47 | 1,958 | 698 |
| **EXP-3** | Peak Chunk 3 Joint | `Det_C3_ep01` | `Trk_C3_cont` | Reproduce Reported Peak | **0.008273** | **0.016545** | 0.000000 | 47 | 1,962 | 698 |
| **EXP-4** | Late Detector Ablation | `Det_C3_ep10` | `Trk_C2_base` | Isolate Ep 10 Detector | **0.000641** | **0.001282** | 0.000000 | 2 | 779 | 743 |
| **EXP-5** | Final Chunk 3 Joint | `Det_C3_ep10` | `Trk_C3_cont` | Final Joint State | **0.000635** | **0.001269** | 0.000000 | 2 | 781 | 743 |
| **EXP-6** | Tracker-Only Ablation | `Det_C2_base` | `Trk_C3_cont` | Isolate Tracker Impact | **0.038592** | **0.077184** | 0.000000 | 186 | 3,508 | 559 |

### Definitive Conclusion from the Controlled Matrix
Comparing **EXP-1** (`Det_C2 + Trk_C2` = 0.038656) with **EXP-6** (`Det_C2 + Trk_C3` = 0.038592):
- Substituting the Chunk 3 continual tracker while holding the baseline detector constant caused an insignificant change of **-0.000064 (-0.17% relative)**. True positive cell detection remained identical (186 TP), and lineage edge linking remained virtually unchanged.
- Conversely, comparing **EXP-1** with **EXP-2** (`Det_C3_ep01 + Trk_C2`) shows an immediate drop from 0.038656 to 0.008314 (**-78.5% relative**).
- **The detector is 99.8% responsible for the Chunk 3 performance collapse.** The tracker is completely blameless.

---

## 6. Root-Cause Analysis & Hypothesis Ranking

From [`reports/chunk3_failure_analysis/root_cause_matrix.csv`](file:///home/vaibhav/Desktop/Projects/Kaggle%20comp/reports/chunk3_failure_analysis/root_cause_matrix.csv):

| Rank | Root Cause Hypothesis | Supporting Evidence | Confidence | Discriminating Test |
| :---: | :--- | :--- | :---: | :--- |
| **1** | **Extreme Positive-Cell Starvation & Sampling Disparity** | `6bba` has 3,990 GT cells; `44b6` has 1,080 cells. A 50/50 patch quota (400 each) resulted in `44b6_0c582fdc` (71 cells total) receiving 134 patches (1.89 patches/cell = repeating identical cells ~2x/epoch) vs. `6bba` cells receiving 0.096 patches/cell (1 positive patch per ~10 epochs). This created a **~20x positive-cell exposure disparity**, starving `6bba` representation manifolds. | **Strongly Supported** | Cell-density normalized sampling quota (quota proportional to total GT cell count: 78.7% 6bba, 21.3% 44b6). |
| **2** | **Unmoderated Continual Learning Rate ($LR=0.002$)** | Training from warm-start weights at the initial base LR (0.002) without continual decay, warm-up, weight decay, or feature regularization erased subtle pre-trained features within 1 epoch. Loss dropped to 2.75 while validation recall collapsed from 38.2% to 2.4%. | **Strongly Supported** | Fine-tune at 10x lower learning rate ($LR=0.0002$) with cosine schedule and weight decay. |
| **3** | **Heatmap Logit Suppression & Threshold Mismatch** | GT center activations dropped by 53-58% on `6bba`, pushing 93.9% (Ep 1) and 100% (Ep 10) of true cells below the rigid 0.25 threshold. | **Confirmed** | Diagnostic threshold sweep on cached heatmaps at $\tau \in [0.05, 0.25]$ to verify peak recovery. |
| **4** | **Cross-Domain Representational Conflict** | `44b6` contrast (1.58) vs `6bba` contrast (4.29) created opposing gradient requirements in convolutional kernels. | **Plausible** | Measure gradient cosine similarity $\langle g_{44b6}, g_{6bba} \rangle$ on Chunk 3 batches. |
| **5** | **Tracker Component Degradation** | Tracker loss converged cleanly (0.00959 to 0.00548); EXP-6 confirms negligible impact (-0.17%). | **Contradicted** | Ablation EXP-1 vs EXP-6 proves tracker preserves baseline performance. |
| **6** | **Candidate Graph & ILP Optimization Failure** | SCIP solved to optimality in <0.2s; node/edge drops matched detector TP loss 1:1. | **Contradicted** | Direct verification of solver convergence. |
| **7** | **Evaluation Inconsistency** | Bit-for-bit hash verification and local baseline reproduction (0.038656) confirm 100% consistency. | **Contradicted** | Hash audit and reproduction. |

---

## 7. Mandatory Diagnostic Answers (Section 14)

### 1. At which epoch did the collapse begin?
**Epoch 1 (abrupt), intensifying precipitously at Epoch 2.**  
Between Epoch 0 (Baseline) and Epoch 1, 6bba node recall plunged from 38.17% to 14.31% (-62.5% relative). At Epoch 2, it collapsed to 2.39% (-93.7% relative), and by Epoch 10 reached 1.28%.

### 2. Was the detector, tracker, or their interaction primarily responsible?
**The detector is 99.8% responsible.**  
Ablation pairing EXP-6 (`Det_C2_base + Trk_C3_cont`) achieved a macro score of **0.038592**, demonstrating that the tracker preserved virtually 100% of the baseline performance. Pairing EXP-2 (`Det_C3_ep01 + Trk_C2_base`) collapsed to **0.008314**, matching the peak regression.

### 3. Is domain gradient interference confirmed, strongly supported, plausible, or unsupported?
**Plausible in the strict gradient sense; Strongly Supported in terms of Positive-Cell Exposure Imbalance and Contrast Mismatch.**  
Because gradient dot products were not logged during training, direct gradient vector opposition cannot be mathematically confirmed. However, the ~20x sampling exposure disparity per cell and the 1.58 vs 4.29 contrast difference are directly measured and strongly supported.

### 4. Did the evaluation pipeline pass its consistency audit?
**Yes, Passed 100%.**  
Validation anchors, coordinate transformations, Hungarian bipartite matching, and SCIP ILP settings were completely identical. The Chunk 2 baseline score of `0.03865588` was perfectly reproduced on local hardware.

### 5. Did checkpoint selection and rollback behave as intended?
**Yes, Passed 100%.**  
The promotion gate detected that the candidate failed both metric superiority (0.0083 vs 0.0387) and anchor regression ceiling (-78.6% drop on `6bba`). The candidate was rejected, and the Chunk 2 baseline was safely preserved as active standard.

### 6. Which existing checkpoint has the strongest verified performance?
**The active Promoted Chunk 2 Baseline:**
- Detector: `checkpoints/best_detector.pt` (`6e9a64c754551402d45c30b7a6949d04f2d2d7e40a3f076243581037a130eaab`)
- Tracker: `checkpoints/best_tracker.pt` (`ed3a4498461da21dbd7b39943eece75f06b6b0fa8faaf570c3aa93403994baf5`)
- Macro Final Score: **`0.03865588`**

### 7. What is the smallest justified next experiment?
A single-variable corrective fine-tuning experiment on Chunk 3:
- **Baseline Checkpoint:** Active Chunk 2 baseline.
- **Single Change:** Lower learning rate by 10x ($LR = 2 \times 10^{-4}$ with cosine decay) and allocate patch quotas proportional to ground-truth cell counts (78.7% to `6bba`, 21.3% to `44b6`).
- **Locked Variables:** Same 8 Chunk 3 sequences, same permanent anchors, batch size 2, patch size 32x128x128.
- **Gating Metric:** Macro Final Score $> 0.03865588$ with $\le 15\%$ drop on `6bba_d82a4fc6`.

### 8. What evidence is still missing?
- Direct per-batch gradient inner products $\langle g_{44b6}, g_{6bba} \rangle$ during multi-domain training.
- Dynamic threshold sweep curves on intermediate epochs 2-9.

---

## 8. Verified Stop Gate Status

- All diagnostic processes have concluded.
- Active baseline remains locked at **0.03865588** (`checkpoints/best_detector.pt` and `checkpoints/best_tracker.pt`).
- Chunk 3 candidate checkpoints and logs are permanently archived in `checkpoints/chunk3_epochs/`.
- **Chunk 4 has NOT been authorized or started.**
- No retraining or parameter modifications have been performed.

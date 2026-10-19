# Biohub — Cell Tracking During Development

[![Pipeline Status](https://img.shields.io/badge/Pipeline-Verified-green.svg)](#)
[![Active Baseline](https://img.shields.io/badge/Active%20Macro%20Score-0.038656-blue.svg)](#)
[![Promotion Gate](https://img.shields.io/badge/Promotion%20Gate-Enforced%20%26%20Active-brightgreen.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](#)

End-to-end deep learning and combinatorial optimization pipeline for the Kaggle [Biohub — Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development) competition.

Reconstructs complete, four-dimensional spatio-temporal cell lineages across developing organism embryos:

$$\text{4D OME-Zarr }(T, Z, Y, X) \longrightarrow \text{3D Temporal U-Net} \longrightarrow \text{Dense Cell Embeddings} \longrightarrow \text{Spatio-Temporal Transformer} \longrightarrow \text{Candidate DAG} \longrightarrow \text{Global SCIP ILP} \longrightarrow \text{Lineage Graphs}$$

---

## 📑 Table of Contents

1. [Architecture Overview](#-architecture-overview)
2. [What Has Been Accomplished (Completed Milestones)](#-what-has-been-accomplished)
3. [Current Project State & Promoted Baseline Standards](#-current-project-state--promoted-baseline)
4. [Chunk 3 Continual Learning & Forensic Failure Analysis](#-chunk-3-forensic-failure-analysis)
5. [Extreme Next Plan (Actionable Roadmap)](#-extreme-next-plan-master-roadmap)
6. [Codebase Structure](#-codebase-structure)
7. [Quickstart & Reproduction Guide](#-quickstart--reproduction-guide)

---

## 🔬 Architecture Overview

```
                      Raw 4D OME-Zarr Volumes (T, Z, Y, X)
                                     │
                                     ▼
                      [Phase 1] Lazy Loading & Normalization
                         - Quantile intensity scaling (0.5% - 99.5%)
                         - Anisotropic physical spacing (1.0, 0.26, 0.26 µm)
                                     │
                                     ▼
                      [Phase 3] 3D Temporal U-Net Detector
                         - 3D convolution backbone + skip connections
                         - Gaussian heatmap generation (σ_z=1.5, σ_yx=3.0)
                         - Dual output: (1, 1, Z, Y, X) heatmap + (1, 32, Z, Y, X) features
                                     │
                                     ▼
                      [Phase 4] 3D Peak Detection & Sub-Pixel Localization
                         - Local maximum suppression (3D max-pooling)
                         - Sub-voxel quadratic 3D refinement
                         - Hungarian bipartite matching (7.0 µm cutoff)
                                     │
                                     ▼
                      [Phase 5] Deep Cell Node Representation
                         - Trilinear RoI visual pooling from backbone (64D)
                         - Spatio-temporal 4D Fourier positional encoding (64D)
                         - Intensity statistics + detection confidence (5D)
                         - Unified fusion MLP (128D node embeddings)
                                     │
                                     ▼
                      [Phase 6] Spatio-Temporal Transformer Tracker
                         - Multi-head self-attention across temporal windows
                         - Directed edge transition probability head
                         - Binary division probability head
                                     │
                                     ▼
                      [Phase 7] Spatio-Temporal Candidate Graph
                         - Spatial radius pruning (d_max = 7.0 µm)
                         - Forward & backward edge scoring
                                     │
                                     ▼
                      [Phase 8] Global Lineage Optimization (SCIP ILP)
                         - Maximize global likelihood objective
                         - Flow conservation: In-degree ≤ 1, Out-degree ∈ {0, 1, 2}
                         - Division consistency constraints
                                     │
                                     ▼
                      [Phase 9] Submission Generator & Official Metrics
                         - Official competition metric calculation
                         - Macro Final Score: 0.5 * (J_adj + Edge_Jaccard)
```

---

## 🏆 What Has Been Accomplished

### Phase 0 to Phase 9: Core Pipeline Architecture
- [x] **Phase 0 — Verification & Environment**: Fully validated PyTorch, CUDA, Zarr v3, PySCIPOpt, and TrackData dependencies with synthetic end-to-end integration tests.
- [x] **Phase 1 — OME-Zarr Data Engine**: Memory-efficient lazy spatial window reader with robust quantile normalization (`normalize_intensity`), patch extractors, and 3D geometric/intensity augmentations.
- [x] **Phase 2 — Exploratory Profiling & Visualizers**: Headless MIP (Maximum Intensity Projection) rendering and interactive multi-channel Napari 4D inspection utilities.
- [x] **Phase 3 — 3D Temporal U-Net**: Fully functional volumetric detection backbone supporting anisotropic voxel downsampling, focal + dice loss, and AMP (Automatic Mixed Precision).
- [x] **Phase 4 — Sub-Pixel Peak Extraction**: 3D quadratic interpolation for continuous sub-voxel peak recovery with bipartite Hungarian matching against ground truth tracks.
- [x] **Phase 5 — Unified Node Embeddings**: Multi-modal fusion combining trilinear 7-point morphological RoI visual features, 4D continuous Fourier coordinate embeddings, and local intensity statistics.
- [x] **Phase 6 — Spatio-Temporal Transformer**: Edge association and division prediction model operating on dense node representations across temporal windows.
- [x] **Phase 7 — Candidate Graph Construction**: Directed acyclic graph (DAG) builder with strict physical distance bounds ($d_{\max} = 7.0\,\mu\text{m}$) and temporal skip handling.
- [x] **Phase 8 — Global Lineage ILP Solver**: Exact integer linear programming solver using SCIP formulation guaranteeing biological plausibility (flow conservation, single parent, binary mitosis).
- [x] **Phase 9 — Official Metric & Submission Engine**: Bit-for-bit implementation of competition tracking metrics ($J_{\text{adj}}$, Edge Jaccard, Division Jaccard, and Final Score).

### Continual Learning & Promotion Progression
- [x] **Phase 3 Baseline Promotion**: Initial end-to-end lineage benchmark achieved Macro Final Score of **0.021245**.
- [x] **Chunk 2 Continual Training & Promotion**: Continual learning specialized on `6bba` in-domain sequences on Kaggle Tesla T4 GPU. Promoted to active baseline:
  - **Macro Final Score:** **0.03865588** (+82.0% relative improvement over Phase 3).
  - **In-Domain Anchor (`6bba_d82a4fc6`):** **0.07731176** (186 TP, Edge Jaccard = 0.1059).
  - Checkpoints: `best_detector.pt` and `best_tracker.pt` verified and promoted as production standards.
- [x] **Chunk 3 Continual Training Execution**: 10 detector epochs and 5 tracker epochs on multi-domain Chunk 3 (3 sequences from `44b6`, 5 sequences from `6bba`).
- [x] **Automated Promotion Safety Gate**: The candidate achieved Macro Final Score of **0.00827264** (-78.6% regression, violating the 15% degradation ceiling). Promotion was objectively **REJECTED**, and the active baseline was preserved bit-for-bit at Chunk 2 standards.
- [x] **Forensic Failure Analysis & Ablation Audit**: Complete diagnostic audit identifying the exact failure mechanism, verified with 24-checkpoint provenance hashes, 10-epoch trajectory curves, domain calibration scans, and a 6-pairing controlled ablation matrix.

---

## 📍 Current Project State & Promoted Baseline

The active production baseline remains locked and fully verified against regression:

| Component | Checkpoint File | SHA-256 Hash | Macro Final Score | 6bba Score | 44b6 Score |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Active Detector** | `checkpoints/best_detector.pt` | `6e9a64c754551402d45c30b7a6949d04f2d2d7e40a3f076243581037a130eaab` | **0.038656** | **0.077312** | 0.000000 |
| **Active Tracker** | `checkpoints/best_tracker.pt` | `ed3a4498461da21dbd7b39943eece75f06b6b0fa8faaf570c3aa93403994baf5` | **0.038656** | **0.077312** | 0.000000 |

Permanent validation anchors in `data/val/`:
- `6bba_d82a4fc6`: In-domain anchor (545 ground-truth nodes, 100 frames). Baseline: 186 TP, Recall = 34.13%, Precision = 13.10%.
- `44b6_66f9292d`: Cross-domain anchor (200 ground-truth nodes, 100 frames). Baseline: 0 TP, 2,277 FP.

---

## 🔍 Chunk 3 Forensic Failure Analysis

Full forensic analysis report: [`reports/chunk3_failure_analysis/CHUNK3_FAILURE_ANALYSIS.md`](file:///home/vaibhav/Desktop/Projects/Kaggle%20comp/reports/chunk3_failure_analysis/CHUNK3_FAILURE_ANALYSIS.md).

### Controlled Inference Matrix (Isolating Detector vs. Tracker)

| Experiment | Detector | Tracker | Macro Score | 6bba Score | 6bba Recall | Node TP | Key Takeaway |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **EXP-1** | `Det_C2_base` | `Trk_C2_base` | **0.038656** | **0.077312** | **34.13%** | **186** | Promoted Chunk 2 Reference Baseline |
| **EXP-2** | `Det_C3_ep01` | `Trk_C2_base` | **0.008314** | **0.016628** | 8.62% | 47 | Detector alone causes -78.5% collapse |
| **EXP-3** | `Det_C3_ep01` | `Trk_C3_cont` | **0.008273** | **0.016545** | 8.62% | 47 | Candidate joint configuration |
| **EXP-4** | `Det_C3_ep10` | `Trk_C2_base` | **0.000641** | **0.001282** | 0.37% | 2 | Late detector drift erases cell recall |
| **EXP-5** | `Det_C3_ep10` | `Trk_C3_cont` | **0.000635** | **0.001269** | 0.37% | 2 | Final epoch joint configuration |
| **EXP-6** | `Det_C2_base` | `Trk_C3_cont` | **0.038592** | **0.077184** | **34.13%** | **186** | **Tracker preserves 99.8% of baseline!** |

> [!IMPORTANT]
> **Key Finding**: The tracker is completely blameless (holding detector constant, EXP-6 achieves **0.038592**, a negligible -0.17% delta). **99.8% of the regression was caused by detector logit suppression.**

### The 3 Core Root Causes Identified

1. **Extreme Positive-Cell Starvation (~20x Disparity)**:
   - Chunk 3 sequence catalog: `6bba` has 5 sequences with 3,990 ground-truth cells; `44b6` has 3 sequences with 1,080 cells.
   - Sequence `44b6_0c582fdc` had only 71 cells across 100 frames. Under the 50/50 patch split (400 patches per domain), `44b6_0c582fdc` was sampled at **1.89 patches/cell** (forced 2x repetition of identical cells per epoch), while `6bba` cells were sampled at **0.096 patches/cell** (1 sample per ~10 epochs).
   - This starved `6bba` representation manifolds while overfitting the model to sparse `44b6` noise.
2. **Unmoderated High Continual Learning Rate ($LR=0.002$)**:
   - Fine-tuning from warm-start weights at full initial base LR without decay or regularization caused immediate catastrophic forgetting within Epochs 1-2. Training loss decreased from 3.74 to 2.29, but validation recall collapsed from 38.2% to 2.4%.
3. **Heatmap Logit Suppression & Fixed Threshold Mismatch**:
   - `6bba` ground-truth center activation mean dropped from 0.1450 down to 0.0683 (Epoch 10).
   - GT Center P90 dropped from 0.2854 to 0.1181. **100% of ground-truth cell centers fell below the production threshold of 0.25**, choking candidate graph inputs.

---

## 🎯 Extreme Next Plan (Master Roadmap)

To systematically overcome the Chunk 3 failure, prevent cross-domain catastrophic forgetting, and advance through remaining chunks, execute the following staged plan:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        EXTREME NEXT MASTER PLAN                        │
├──────────────────────────┬─────────────────────────────────────────────┤
│ Milestone 1              │ Cell-Density-Normalized Stratified Sampler  │
│ Milestone 2              │ Continual Learning Rate & Regularizer       │
│ Milestone 3              │ Cross-Domain Contrast Equalization Engine   │
│ Milestone 4              │ Dynamic Validation Calibration Audit        │
│ Milestone 5              │ Chunk 3.1 Controlled Retraining Experiment  │
│ Milestone 6              │ Progressive Ingestion of Chunks 4 - 12      │
│ Milestone 7              │ Final Kaggle Competition Submission Run     │
└──────────────────────────┴─────────────────────────────────────────────┘
```

### Milestone 1: Cell-Density-Normalized Stratified Sampler
- **Status**: Implemented: `density_normalized_quotas` in `src/data/patch_sampler.py` (also `ContinualTrainer(quota_mode="cell_density")`); Chunk 3.1 crops are 100% cell-centred.
- **Problem Solved**: Eliminates the 20x per-cell exposure disparity.
- **Implementation**:
  - Replace the naïve 50/50 patch quota with **cell-count proportional quota allocation**:
    $$\text{Quota}(D) = N_{\text{total\_patches}} \times \frac{\sum_{s \in D} \text{Nodes}(s)}{\sum_{s \in \text{All}} \text{Nodes}(s)}$$
  - In Chunk 3: 78.7% of patches (630 patches) to `6bba`, 21.3% of patches (170 patches) to `44b6`.
  - Enforce **positive-cell-centered patch extraction**: sample uniformly over detected/ground-truth cell coordinates rather than drawing blind spatial volumes, guaranteeing that every cell has an equal probability of appearance.

### Milestone 2: Continual Learning Rate & Knowledge Distillation
- **Status**: Implemented in `src/training/continual.py` (warm-up + cosine scheduler, frozen-teacher feature distillation on the last decoder block `up3`).
- **Problem Solved**: Prevents catastrophic forgetting of pre-trained feature manifolds.
- **Implementation**:
  - **10x Learning Rate Reduction**: Drop continual detector fine-tuning LR from $2 \times 10^{-3}$ to **$2 \times 10^{-4}$**.
  - **Cosine Annealing Schedule**: Warmup for 1 epoch, cosine decay to $1 \times 10^{-5}$ at Epoch 8.
  - **Feature-Manifold Preservation Loss (Distillation)**:
    $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Focal}} + \mathcal{L}_{\text{Dice}} + \lambda_{\text{distill}} \|\mathbf{F}_{\text{backbone}} - \mathbf{F}_{\text{frozen\_C2}}\|_2^2$$
    where $\lambda_{\text{distill}} = 0.1$, anchoring intermediate convolutional activations to the golden Chunk 2 baseline representations.

### Milestone 3: Cross-Domain Contrast Equalization Engine
- **Problem Solved**: Resolves the severe contrast gap between `44b6` (contrast = 1.58) and `6bba` (contrast = 4.29).
- **Implementation**:
  - Implement adaptive percentile clipping and local contrast normalization in `src/data/preprocessing.py`:
    $$\tilde{I} = \text{clip}\left(\frac{I - P_2(I)}{P_{98}(I) - P_2(I) + \epsilon}, 0, 1\right)$$
  - Ensure uniform background variance across both domain volumes prior to input into the 3D U-Net.

### Milestone 4: Diagnostic Calibration & Dynamic Threshold Sweeping
- **Problem Solved**: Prevents rigid threshold mismatch ($\tau=0.25$) from choking valid peaks during slight logit scale shifts.
- **Implementation**:
  - Offline threshold sweep tool (`scripts/sweep_detection_threshold.py`) evaluating $\tau \in [0.10, 0.30]$ in steps of 0.02 on cached validation heatmaps.
  - Output precision-recall-F1 curves and simulated lineage scores to identify optimal operating thresholds per domain without modifying inference invariants.

### Milestone 5: Chunk 3.1 Controlled Retraining Experiment
- **Status**: Ready to run: `scripts/run_chunk31_training.py`, launched on Kaggle GPU via `kernels/chunk31/` (`kaggle kernels push -p kernels/chunk31`). Milestones 3 and 4 are not part of this run.
- **Objective**: Execute single-variable controlled retraining on Chunk 3 under the corrected protocol.
- **Configuration**:
  - Base Checkpoint: Promoted Chunk 2 baseline (`best_detector.pt` / `best_tracker.pt`).
  - Optimizer: AdamW, $LR = 2 \times 10^{-4}$, weight decay = $1 \times 10^{-4}$.
  - Sampler: Density-normalized positive-cell sampler (630 patches `6bba`, 170 patches `44b6`).
  - Regularizer: Feature distillation $\lambda = 0.1$.
  - Epochs: 8 detector epochs + 3 tracker epochs.
- **Promotion Gate Target**:
  - Macro Final Score $\ge \mathbf{0.045000}$ (+16.4% over active baseline).
  - Permanent anchor `6bba_d82a4fc6` $\ge 0.0750$ (preserves $>97\%$ of Chunk 2 score).
  - Permanent anchor `44b6_66f9292d` $> 0.0150$ (unlocks cross-domain cell detection).

### Milestone 6: Progressive Ingestion of Chunks 4 through 12
- Systematically ingest remaining ~160 GB of training sequences in stratified 8-sequence chunks using `scripts/run_chunked_workflow.py`:
  - Checkpoint registry logging in `registry/experiments.sqlite` and `registry/chunk_registry.jsonl`.
  - Non-negotiable 15% degradation ceiling enforced on every promotion candidate.
  - Automatic rollback on failure; permanent archival of all per-epoch weights.

### Milestone 7: Final Competition Submission Pipeline
- Automated Kaggle submission script:
  - Generate full 4D lineage predictions for test sequences.
  - Formulate and solve global SCIP ILP to mathematical optimality.
  - Verify submission schema (`lineage_id`, `track_id`, `t`, `z`, `y`, `x`, `parent_id`).
  - Output verified `submission.csv` and checksum report.

---

## 📁 Codebase Structure

```
├── checkpoints/
│   ├── best_detector.pt                 # Golden Promoted Chunk 2 Baseline (Macro = 0.0387)
│   ├── best_tracker.pt                  # Golden Promoted Chunk 2 Baseline
│   ├── chunk3_epochs/                   # Preserved Chunk 3 Candidate Checkpoints (Ep 1 - 10)
│   └── archive/                         # Preserved Phase 3 Historical Checkpoints
├── data/
│   ├── val/                             # Permanent Validation Anchors (44b6 & 6bba)
│   └── raw/                             # Raw 4D OME-Zarr Datasets
├── registry/
│   ├── chunk_registry.jsonl             # Audit trail of chunk training & promotion events
│   └── experiments.sqlite               # Relational database tracking runs & metrics
├── reports/
│   ├── chunk3_failure_analysis/         # Complete forensic diagnostics & ablation data
│   │   ├── CHUNK3_FAILURE_ANALYSIS.md   # Comprehensive failure analysis report
│   │   ├── checkpoint_provenance.csv    # SHA-256 hashes & metadata for 24 checkpoints
│   │   ├── epoch_trajectory.csv         # Canonical per-epoch training & validation metrics
│   │   ├── detector_tracker_ablation.csv# 6-cell controlled ablation matrix
│   │   ├── domain_diagnostics.json      # Heatmap calibration & contrast statistics
│   │   ├── root_cause_matrix.csv        # Ranked hypotheses & confidence classifications
│   │   ├── failure_analysis_manifest.json# Audit metadata & pipeline configuration
│   │   ├── epoch_loss_vs_recall.png     # Visual trajectory plot
│   │   ├── heatmap_distribution_shift.png# Calibration shift bar plot
│   │   └── detector_tracker_ablation_barplot.png # Ablation comparison bar plot
│   ├── CHUNK3_FINAL_REPORT.md           # Chunk 3 training completion report
│   └── chunk3_continual_results.json    # Chunk 3 Kaggle execution results
├── scripts/
│   ├── run_chunk3_training.py           # Kaggle GPU Chunk 3 continual learning script
│   ├── run_chunk31_training.py          # Milestone 5: Chunk 3.1 controlled retraining (M1 + M2)
│   ├── run_ablation_matrix.py           # Controlled inference ablation runner
│   ├── generate_failure_analysis_plots.py# Diagnostic visualization generator
│   ├── generate_failure_manifest.py     # Audit manifest generator
│   └── run_chunked_workflow.py          # Storage-efficient progressive chunk workflow
└── src/
    ├── data/                            # OME-Zarr reader, normalization, augmentations
    ├── detection/                       # 3D Temporal U-Net & sub-pixel peak detector
    ├── evaluation/                      # Competition tracking metrics & Hungarian matching
    ├── graph/                           # Spatio-temporal candidate graph builder
    ├── optimization/                    # Global lineage SCIP ILP solver
    ├── pipeline/                        # End-to-end HeldOutEvaluator & chunk manager
    ├── representation/                  # RoI pooling & 4D Fourier cell node embeddings
    └── tracking/                        # Spatio-temporal Transformer tracker
```

---

## 💻 Quickstart & Reproduction Guide

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/Vaibhav20k/BIOhub.git
cd BIOhub

# Create virtual environment with uv or pip
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

### 2. Verify Promoted Chunk 2 Baseline (Local Reproduction)
Reproduce the exact active baseline score (**0.038656**) on local GPU in ~170 seconds:
```bash
python -c "
from pathlib import Path
from src.data.zarr_reader import open_dataset
from src.pipeline.held_out_evaluator import HeldOutEvaluator
import torch

val_datasets = [
    open_dataset('data/val/44b6_66f9292d', load_tracks=True),
    open_dataset('data/val/6bba_d82a4fc6', load_tracks=True),
]

evaluator = HeldOutEvaluator.from_checkpoints(
    detector_checkpoint='checkpoints/best_detector.pt',
    tracker_checkpoint='checkpoints/best_tracker.pt',
    detection_threshold=0.25,
    max_peaks_per_frame=150,
    max_distance_um=7.0,
    window_size=1,
)
res = evaluator.evaluate_cohort(val_datasets)
print(f'Macro Final Score: {res[\"macro_final_score\"]:.6f}')
print(f'6bba Final Score:  {res[\"sequences\"][\"6bba_d82a4fc6\"][\"final_score\"]:.6f}')
print(f'44b6 Final Score:  {res[\"sequences\"][\"44b6_66f9292d\"][\"final_score\"]:.6f}')
"
```

### 3. Run Controlled Inference Ablation Matrix
```bash
python scripts/run_ablation_matrix.py
```

### 4. Regenerate Failure Analysis Diagnostic Plots
```bash
python scripts/generate_failure_analysis_plots.py
```

### 5. Run Test Suite
```bash
pytest tests/ -v
```

#!/usr/bin/env python3
"""Generate high-resolution forensic diagnostic plots for Chunk 3 Failure Analysis."""

from pathlib import Path
import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

repo_root = Path(__file__).resolve().parent.parent
analysis_dir = repo_root / "reports/chunk3_failure_analysis"
analysis_dir.mkdir(parents=True, exist_ok=True)

# 1. Plot Epoch Trajectory: Loss vs. Recall
def plot_epoch_trajectory():
    csv_path = analysis_dir / "epoch_trajectory.csv"
    if not csv_path.exists():
        print(f"Warning: {csv_path} does not exist yet.")
        return
    
    epochs = []
    losses = []
    rec_6bba = []
    rec_44b6 = []
    j_adj_macro = []
    
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            tl = row["train_loss"]
            try:
                losses.append(float(tl))
            except (ValueError, TypeError):
                losses.append(np.nan)
            rec_6bba.append(float(row["6bba_recall"]) * 100.0)
            rec_44b6.append(float(row["44b6_recall"]) * 100.0)
            j_adj_macro.append(float(row["macro_j_adj"]) * 100.0)
            
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Ax1: Training Loss
    ax1.plot(epochs[1:], losses[1:], "o-", color="#d9534f", linewidth=2.2, label="Detector Train Loss (Focal + Dice)")
    ax1.set_ylabel("Training Loss", fontsize=12, fontweight="bold", color="#d9534f")
    ax1.tick_params(axis="y", labelcolor="#d9534f")
    ax1.set_title("Chunk 3 Continual Training Trajectory: Loss vs. Validation Recall", fontsize=14, fontweight="bold", pad=12)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")
    
    # Ax2: Recall & Macro Score
    ax2.plot(epochs, rec_6bba, "s-", color="#0275d8", linewidth=2.2, label="6bba Node Recall (%)")
    ax2.plot(epochs, rec_44b6, "^-", color="#5cb85c", linewidth=2.2, label="44b6 Node Recall (%)")
    ax2.plot(epochs, j_adj_macro, "d--", color="#6f42c1", linewidth=2.0, label="Macro Adj Jaccard (x100)")
    ax2.axvline(x=0, color="gray", linestyle=":", label="Chunk 2 Promoted Baseline")
    ax2.axvline(x=1, color="orange", linestyle="--", alpha=0.8, label="Abrupt Collapse Point (Ep 1-2)")
    
    ax2.set_xlabel("Continual Learning Epoch", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Validation Metric (%)", fontsize=12, fontweight="bold")
    ax2.set_xticks(range(0, 11))
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")
    
    plt.tight_layout()
    out_file = analysis_dir / "epoch_loss_vs_recall.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved {out_file}")

# 2. Plot Heatmap Calibration Shift
def plot_heatmap_shift():
    json_path = analysis_dir / "domain_diagnostics.json"
    if not json_path.exists():
        print(f"Warning: {json_path} does not exist yet.")
        return
        
    with open(json_path) as f:
        diag = json.load(f)
        
    models = ["Chunk 2 Baseline Detector", "Chunk 3 Epoch 1 Detector", "Chunk 3 Epoch 10 Detector"]
    model_labels = ["Chunk 2 Base\n(Epoch 0)", "Chunk 3 Ep 1\n(Peak Cand.)", "Chunk 3 Ep 10\n(Final Drift)"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 6bba
    gt_means_6b = [diag[m]["6bba_d82a4fc6"]["gt_mean"] for m in models]
    gt_p90s_6b = [diag[m]["6bba_d82a4fc6"]["gt_p90"] for m in models]
    bg_p90s_6b = [diag[m]["6bba_d82a4fc6"]["bg_p90"] for m in models]
    
    x = np.arange(len(models))
    w = 0.25
    ax1.bar(x - w, gt_means_6b, width=w, label="GT Center Mean", color="#0275d8")
    ax1.bar(x, gt_p90s_6b, width=w, label="GT Center P90", color="#5bc0de")
    ax1.bar(x + w, bg_p90s_6b, width=w, label="Background P90", color="#d9534f")
    ax1.axhline(y=0.25, color="black", linestyle="--", linewidth=1.5, label="Detection Cutoff (0.25)")
    ax1.set_title("6bba (High Contrast Domain) Calibration", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels)
    ax1.set_ylabel("Heatmap Activation Value", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 0.35)
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax1.legend(loc="upper right")
    
    # 44b6
    gt_means_44 = [diag[m]["44b6_66f9292d"]["gt_mean"] for m in models]
    gt_p90s_44 = [diag[m]["44b6_66f9292d"]["gt_p90"] for m in models]
    bg_p90s_44 = [diag[m]["44b6_66f9292d"]["bg_p90"] for m in models]
    
    ax2.bar(x - w, gt_means_44, width=w, label="GT Center Mean", color="#0275d8")
    ax2.bar(x, gt_p90s_44, width=w, label="GT Center P90", color="#5bc0de")
    ax2.bar(x + w, bg_p90s_44, width=w, label="Background P90", color="#d9534f")
    ax2.axhline(y=0.25, color="black", linestyle="--", linewidth=1.5, label="Detection Cutoff (0.25)")
    ax2.set_title("44b6 (Low Contrast Domain) Calibration", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_labels)
    ax2.set_ylabel("Heatmap Activation Value", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 0.35)
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax2.legend(loc="upper right")
    
    plt.suptitle("Heatmap Distribution Shift & Separation Audit (Section 6.1)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_file = analysis_dir / "heatmap_distribution_shift.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved {out_file}")

# 3. Plot Ablation Matrix Barplot
def plot_ablation_matrix():
    csv_path = analysis_dir / "detector_tracker_ablation.csv"
    if not csv_path.exists():
        print(f"Warning: {csv_path} does not exist yet.")
        return
        
    ids = []
    labels = []
    macro_scores = []
    s6b_scores = []
    
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(row["exp_id"])
            lbl = f"{row['exp_id']}\n{row['detector'][:10]}\n+\n{row['tracker'][:10]}"
            labels.append(lbl)
            macro_scores.append(float(row["macro_final_score"]))
            s6b_scores.append(float(row["6bba_final_score"]))
            
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(ids))
    w = 0.35
    
    rects1 = ax.bar(x - w/2, macro_scores, width=w, label="Macro Final Score", color="#0275d8")
    rects2 = ax.bar(x + w/2, s6b_scores, width=w, label="6bba Final Score", color="#5cb85c")
    
    ax.set_ylabel("Competition Lineage Score", fontsize=12, fontweight="bold")
    ax.set_title("Controlled Inference Matrix: Isolating Detector vs. Tracker (Section 5.1)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    cfg_names = [
        {"exp_id": "EXP-1", "name": "Det C2 +\nTrk C2"},
        {"exp_id": "EXP-2", "name": "Det Ep1 +\nTrk C2"},
        {"exp_id": "EXP-3", "name": "Det Ep1 +\nTrk C3"},
        {"exp_id": "EXP-4", "name": "Det Ep10 +\nTrk C2"},
        {"exp_id": "EXP-5", "name": "Det Ep10 +\nTrk C3"},
        {"exp_id": "EXP-6", "name": "Det C2 +\nTrk C3"},
    ]
    ax.set_xticklabels([f"{r['exp_id']}:\n{r['name']}" for r in cfg_names[:len(ids)]], fontsize=10)
    ax.axhline(y=0.038656, color="red", linestyle="--", linewidth=1.5, label="Active C2 Baseline (0.0387)")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend(loc="upper right")
    
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.4f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight="bold")
                    
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.4f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight="bold")
                    
    plt.tight_layout()
    out_file = analysis_dir / "detector_tracker_ablation_barplot.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved {out_file}")

if __name__ == "__main__":
    plot_epoch_trajectory()
    plot_heatmap_shift()
    plot_ablation_matrix()

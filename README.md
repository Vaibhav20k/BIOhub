# Biohub — Cell Tracking During Development

End-to-end deep learning pipeline for the Kaggle [Biohub — Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development) competition.

## 🔬 Architecture Overview

The system reconstructs 4D developing cell lineages:

$$\text{4D Volumes }(T, Z, Y, X) \longrightarrow \text{3D U-Net Detection} \longrightarrow \text{Dense Cell Features} \longrightarrow \text{Transformer Tracker} \longrightarrow \text{Global ILP} \longrightarrow \text{submission.csv}$$

## 📊 Pipeline Status

- [x] **Phase 0**: Environment setup, dependency validation, and synthetic test fixture.
- [x] **Phase 1**: Lazy OME-Zarr reader, quantile normalization, patch sampling, and 3D augmentations.
- [x] **Phase 2**: Trajectory velocity/density profiling and interactive Napari / headless MIP visualization.
- [x] **Phase 3**: 3D Temporal U-Net detector, anisotropic downsampling, Gaussian heatmap generation, and mixed-precision training.
- [x] **Phase 4**: 3D sub-pixel NMS peak extraction and bipartite Hungarian matching ($7.0\,\mu\text{m}$ cutoff).
- [x] **Phase 5**: Cell feature extraction and spatio-temporal positional encoding.
- [x] **Phase 6**: Spatio-temporal Transformer edge predictor.
- [ ] **Phase 7**: Candidate graph builder.
- [ ] **Phase 8**: Global ILP lineage solver.
- [ ] **Phase 9**: Competition submission pipeline & validation.

## 🚀 Quickstart

### Setup Environment
```bash
# Using uv or pip
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

### Run Tests
```bash
pytest tests/ -v
```

### Train Baseline Detector
```bash
python src/training/train_detector.py --train-dataset 44b6_0b24845f --epochs 10
```

### Evaluate Detector & Threshold Sweep
```bash
python scripts/evaluate_detector.py --checkpoint checkpoints/best_detector.pt --dataset 44b6_0b24845f --sweep
```

### Train Spatio-Temporal Tracker
```bash
python src/training/train_tracker.py --train-dataset 44b6_0b24845f --detector-checkpoint checkpoints/best_detector.pt --epochs 10
```

### View 4D Dataset & Lineages
```bash
# Interactive Napari viewer
python scripts/view_dataset.py --dataset 44b6_0b24845f

# Headless MIP export
python scripts/view_dataset.py --dataset 44b6_0b24845f --headless
```

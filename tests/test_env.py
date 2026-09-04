"""Environment verification test suite for Biohub Cell Tracking project."""

import sys
import pytest
import torch
import tracksdata as td
import zarr
import polars as pl
import scipy
from scipy.spatial import KDTree
import numpy as np


def test_python_version():
    """Verify python version is 3.11 or 3.12."""
    major, minor = sys.version_info[:2]
    assert major == 3
    assert 11 <= minor <= 13, f"Expected Python 3.11-3.13, found {major}.{minor}"


def test_cuda_acceleration():
    """Verify PyTorch is installed with functional CUDA support."""
    assert torch.cuda.is_available(), "CUDA is not available on this environment"
    assert torch.cuda.device_count() >= 1, "No CUDA devices detected"
    device_name = torch.cuda.get_device_name(0)
    assert len(device_name) > 0, "Failed to query device name"

    # Quick tensor allocation on GPU
    x = torch.ones((10, 10), device="cuda")
    y = x * 2.0
    assert torch.allclose(y, torch.full((10, 10), 2.0, device="cuda"))


def test_tracksdata_graph_construction():
    """Verify tracksdata graph construction and manipulation."""
    g = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        g.add_node_attr_key(key, pl.Float64, 0.0)

    n0 = g.add_node({"t": 0, "z": 10.0, "y": 20.0, "x": 30.0})
    n1 = g.add_node({"t": 1, "z": 12.0, "y": 22.0, "x": 32.0})
    n2 = g.add_node({"t": 1, "z": 14.0, "y": 24.0, "x": 34.0})

    g.add_edge_attr_key("edge_prob", pl.Float64, 0.0)
    e0 = g.add_edge(n0, n1, {"edge_prob": 0.95})
    e1 = g.add_edge(n0, n2, {"edge_prob": 0.90})

    assert g.num_nodes() == 3
    assert g.num_edges() == 2


def test_ilp_solver_initialization():
    """Verify td.solvers.ILPSolver initializes properly."""
    solver = td.solvers.ILPSolver(
        edge_weight=td.EdgeAttr("edge_prob"),
        appearance_weight=0.1,
        disappearance_weight=0.1,
        division_weight=1.0,
    )
    assert solver is not None


def test_zarr_and_polars():
    """Verify Zarr storage and Polars data manipulation."""
    store = zarr.storage.MemoryStore()
    root = zarr.open_group(store=store, mode="w")
    arr = root.create_array("0", shape=(2, 8, 16, 16), dtype="float32")
    arr[0, :, :, :] = np.ones((8, 16, 16), dtype=np.float32)
    assert arr.shape == (2, 8, 16, 16)
    assert float(arr[0, 0, 0, 0]) == 1.0

    df = pl.DataFrame({
        "t": [0, 1],
        "z": [10.0, 12.0],
        "y": [20.0, 22.0],
        "x": [30.0, 32.0],
    })
    assert df.height == 2
    assert "z" in df.columns


def test_scipy_spatial_kd_tree():
    """Verify KDTree Euclidean distance matching in physical space."""
    scale = np.array([1.625, 0.40625, 0.40625])  # (Z, Y, X)
    pts1 = np.array([[10, 20, 30], [15, 25, 35]]) * scale
    pts2 = np.array([[10, 21, 30], [20, 20, 20]]) * scale

    tree = KDTree(pts1)
    dists, indices = tree.query(pts2, distance_upper_bound=7.0)
    # First point should match (dist ~ 0.40625 <= 7.0)
    assert dists[0] <= 7.0
    assert indices[0] == 0


def test_synthetic_fixture_dataset():
    """Verify that the synthetic fixture Zarr and GEFF load cleanly."""
    from pathlib import Path
    zarr_path = Path("data/fixtures/synthetic_clip.zarr")
    geff_path = Path("data/fixtures/synthetic_clip.geff")

    assert zarr_path.exists(), "Synthetic Zarr fixture not found"
    assert geff_path.exists(), "Synthetic GEFF fixture not found"

    # Test Zarr loading
    root = zarr.open_group(str(zarr_path), mode="r")
    assert "0" in root
    img = root["0"]
    assert img.shape == (5, 16, 64, 64)
    assert "image_statistics" in root.attrs
    assert "quantiles" in root.attrs["image_statistics"]

    # Test GEFF loading
    g_res = td.graph.IndexedRXGraph.from_geff(str(geff_path))
    graph = g_res[0] if isinstance(g_res, tuple) else g_res
    assert graph.num_nodes() == 12
    assert graph.num_edges() == 10

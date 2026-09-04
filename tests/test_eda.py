"""Unit tests for Phase 2: Exploratory Data Analysis and Spatial Visualization."""

from pathlib import Path
import numpy as np
import pytest

from src.data.zarr_reader import open_dataset
from src.eda.dataset_stats import (
    compute_displacement_profile,
    compute_density_profile,
    compute_mitosis_profile,
    profile_dataset,
)
from src.visualization.napari_viewer import (
    create_napari_viewer,
    export_mip_gallery,
    export_ortho_projections,
    geff_to_napari_tracks,
    render_mip_overlay,
)


@pytest.fixture
def synthetic_dataset():
    path = Path("data/fixtures/synthetic_clip")
    assert (path.parent / f"{path.name}.zarr").exists(), "Synthetic fixture zarr missing"
    return open_dataset(path, load_tracks=True)


@pytest.fixture
def real_dataset_optional():
    path = Path("data/train/44b6_0b24845f")
    if (path.parent / f"{path.name}.zarr").exists():
        return open_dataset(path, load_tracks=True)
    return None


def test_displacement_profile_synthetic(synthetic_dataset):
    profile = compute_displacement_profile(synthetic_dataset)
    assert profile is not None
    assert profile.count == 10
    assert 1.0 < profile.mean_um < 1.5
    assert profile.max_um <= 3.0
    assert profile.recommended_search_radius_um >= 7.0


def test_density_profile_synthetic(synthetic_dataset):
    profile = compute_density_profile(synthetic_dataset)
    assert profile is not None
    assert profile.total_nodes == 12
    assert profile.num_timepoints_with_nodes == 5
    assert profile.mean_nodes_per_timepoint == 2.4
    assert profile.max_nodes_per_timepoint == 3
    assert profile.mean_nearest_neighbor_dist_um > 0


def test_mitosis_profile_synthetic(synthetic_dataset):
    profile = compute_mitosis_profile(synthetic_dataset)
    assert profile is not None
    assert profile.division_count == 1
    assert profile.division_rate_per_frame == 0.2
    assert 4.0 < profile.mean_daughter_separation_um < 6.0


def test_geff_to_napari_tracks_conversion(synthetic_dataset):
    tracks_arr, napari_graph, node_to_track = geff_to_napari_tracks(synthetic_dataset)

    # 12 total nodes must be present in tracks array
    assert len(tracks_arr) == 12
    assert tracks_arr.shape[1] == 5  # [track_id, t, z, y, x]
    assert len(node_to_track) == 12

    # Napari graph must have parent relationships for the 2 daughter branches
    assert len(napari_graph) == 2
    # Ensure time is sorted per track
    for tid in np.unique(tracks_arr[:, 0]):
        t_vals = tracks_arr[tracks_arr[:, 0] == tid, 1]
        assert np.all(np.diff(t_vals) >= 0), f"Track {tid} time is not monotonically increasing"


def test_render_mip_overlay(synthetic_dataset):
    mip, nodes_t, edge_coords = render_mip_overlay(synthetic_dataset, timepoint=2, proj_axis="z")
    assert mip.shape == (64, 64)
    assert nodes_t.height == 2
    # At t=2, parent cell divides into 2 edges to t=3
    # Total edges starting at t=2: 1 (cell 1) + 2 (cell 2 dividing) = 3
    assert len(edge_coords) == 3


def test_export_gallery_and_projections(synthetic_dataset, tmp_path):
    gallery_path = tmp_path / "gallery.png"
    out = export_mip_gallery(synthetic_dataset, gallery_path, num_frames=3)
    assert out.exists()
    assert out.stat().st_size > 1000

    ortho_path = tmp_path / "ortho.png"
    out_ortho = export_ortho_projections(synthetic_dataset, timepoint=1, out_path=ortho_path)
    assert out_ortho.exists()
    assert out_ortho.stat().st_size > 1000


def test_napari_viewer_creation(synthetic_dataset):
    viewer = create_napari_viewer(synthetic_dataset, show=False)
    if viewer is not None:
        layer_names = [l.name for l in viewer.layers]
        assert any("Volume" in name for name in layer_names)
        assert "Cell Detections" in layer_names
        assert "Lineage Tracks" in layer_names
        viewer.close()


def test_profile_dataset_full(synthetic_dataset, tmp_path):
    profile = profile_dataset(synthetic_dataset, out_dir=tmp_path)
    assert profile.dataset_name == "synthetic_clip"
    assert (tmp_path / "profile_synthetic_clip.json").exists()
    assert (tmp_path / "eda_distributions_synthetic_clip.png").exists()


def test_real_dataset_profile(real_dataset_optional):
    if real_dataset_optional is None:
        pytest.skip("Real dataset 44b6_0b24845f not present")

    tracks_arr, napari_graph, node_to_track = geff_to_napari_tracks(real_dataset_optional)
    assert len(tracks_arr) == 51
    assert len(node_to_track) == 51

    profile = compute_displacement_profile(real_dataset_optional)
    assert profile is not None
    assert profile.count == 49
    # Real cell biological velocity is in range 1-5 µm
    assert 1.0 < profile.mean_um < 5.0
    assert profile.recommended_search_radius_um == 7.0

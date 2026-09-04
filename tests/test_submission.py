"""Unit tests for competition submission generation and integrity validation."""

from pathlib import Path
import polars as pl
import pytest

from src.submission.validator import SubmissionValidator, validate_submission
from src.submission.writer import DatasetLineageResult, SubmissionWriter


def test_sample_submission_validation():
    """Verify that official data/sample_submission.csv passes validation without errors."""
    sample_csv = Path("data/sample_submission.csv")
    assert sample_csv.exists()

    result = validate_submission(sample_csv)
    assert result.is_valid is True
    assert len(result.errors) == 0
    assert result.num_rows == 20
    assert result.num_datasets == 4
    assert result.num_nodes == 12
    assert result.num_edges == 8


def test_submission_writer_single_and_multi_dataset():
    """Verify SubmissionWriter generates compliant schema and continuous 0-indexed IDs."""
    writer = SubmissionWriter(round_coordinates=True)

    # Dataset A
    nodes_a = pl.DataFrame(
        {
            "node_id": [1, 2],
            "t": [0, 1],
            "z": [10.2, 10.4],
            "y": [20.1, 20.3],
            "x": [30.5, 30.7],
        }
    )
    edges_a = pl.DataFrame({"source_id": [1], "target_id": [2]})
    res_a = DatasetLineageResult(dataset_name="dataset_A", nodes_df=nodes_a, edges_df=edges_a)

    # Dataset B
    nodes_b = pl.DataFrame(
        {
            "node_id": [100, 101, 102],
            "t": [5, 6, 7],
            "z": [5.0, 5.1, 5.2],
            "y": [15.0, 15.1, 15.2],
            "x": [25.0, 25.1, 25.2],
        }
    )
    edges_b = pl.DataFrame({"source_id": [100, 101], "target_id": [101, 102]})
    res_b = DatasetLineageResult(dataset_name="dataset_B", nodes_df=nodes_b, edges_df=edges_b)

    sub_df = writer.build_submission_dataframe([res_a, res_b])

    # Check total rows: 2 nodes + 1 edge + 3 nodes + 2 edges = 8 rows
    assert sub_df.height == 8
    assert sub_df["id"].to_list() == list(range(8))

    # Validate output
    val_res = validate_submission(sub_df)
    assert val_res.is_valid is True
    assert len(val_res.errors) == 0
    assert val_res.num_datasets == 2


def test_submission_validator_catches_errors():
    """Verify validator catches orphan edges, non-sequential IDs, self-loops, and backwards edges."""
    validator = SubmissionValidator()

    # 1. Non-sequential ID
    corrupt_id_df = pl.DataFrame(
        {
            "id": [0, 2],  # Gap!
            "dataset": ["ds1", "ds1"],
            "row_type": ["node", "node"],
            "node_id": [1, 2],
            "t": [0, 1],
            "z": [10, 10],
            "y": [10, 10],
            "x": [10, 10],
            "source_id": [-1, -1],
            "target_id": [-1, -1],
        }
    )
    res_id = validator.validate(corrupt_id_df)
    assert res_id.is_valid is False
    assert any("strictly sequential" in err for err in res_id.errors)

    # 2. Orphan edge (target_id 999 does not exist in nodes)
    corrupt_orphan_df = pl.DataFrame(
        {
            "id": [0, 1],
            "dataset": ["ds1", "ds1"],
            "row_type": ["node", "edge"],
            "node_id": [1, -1],
            "t": [0, -1],
            "z": [10, -1],
            "y": [10, -1],
            "x": [10, -1],
            "source_id": [-1, 1],
            "target_id": [-1, 999],  # Non-existent node!
        }
    )
    res_orphan = validator.validate(corrupt_orphan_df)
    assert res_orphan.is_valid is False
    assert any("orphan edges" in err for err in res_orphan.errors)

    # 3. Self loop (source_id == target_id)
    corrupt_loop_df = pl.DataFrame(
        {
            "id": [0, 1],
            "dataset": ["ds1", "ds1"],
            "row_type": ["node", "edge"],
            "node_id": [1, -1],
            "t": [0, -1],
            "z": [10, -1],
            "y": [10, -1],
            "x": [10, -1],
            "source_id": [-1, 1],
            "target_id": [-1, 1],  # Self loop!
        }
    )
    res_loop = validator.validate(corrupt_loop_df)
    assert res_loop.is_valid is False
    assert any("self-loops" in err for err in res_loop.errors)

    # 4. Backward edge (t_target <= t_source)
    corrupt_time_df = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "dataset": ["ds1", "ds1", "ds1"],
            "row_type": ["node", "node", "edge"],
            "node_id": [1, 2, -1],
            "t": [5, 2, -1],  # Node 1 at t=5, Node 2 at t=2
            "z": [10, 10, -1],
            "y": [10, 10, -1],
            "x": [10, 10, -1],
            "source_id": [-1, -1, 1],  # 1 -> 2 is going backward in time (5 -> 2)!
            "target_id": [-1, -1, 2],
        }
    )
    res_time = validator.validate(corrupt_time_df)
    assert res_time.is_valid is False
    assert any("strictly forward in time" in err for err in res_time.errors)


def test_end_to_end_submission_pipeline_fixture(tmp_path):
    """Verify EndToEndSubmissionPipeline runs on synthetic fixture and generates valid submission."""
    from src.submission.pipeline import EndToEndSubmissionPipeline

    fixture_path = Path("data/fixtures/synthetic_clip.zarr")
    assert fixture_path.exists()

    output_csv = tmp_path / "fixture_submission.csv"

    pipeline = EndToEndSubmissionPipeline(
        detector_checkpoint="checkpoints/best_detector.pt",
        tracker_checkpoint="checkpoints/best_tracker.pt",
        detection_threshold=0.3,
        max_peaks_per_frame=50,
        max_distance_um=7.0,
    )

    sub_df, val_res = pipeline.generate_submission(
        datasets=[fixture_path],
        output_csv=output_csv,
    )

    assert output_csv.exists()
    assert val_res.is_valid is True
    assert val_res.num_nodes > 0
    assert val_res.num_edges > 0
    assert val_res.num_datasets == 1
    assert sub_df.height == val_res.num_rows


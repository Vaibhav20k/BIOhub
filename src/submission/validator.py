"""Submission integrity validator for Kaggle Biohub competition requirements."""

from dataclasses import asdict, dataclass, field
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class SubmissionValidationResult:
    """Detailed validation outcome for a competition submission."""

    is_valid: bool
    num_rows: int
    num_datasets: int
    num_nodes: int
    num_edges: int
    datasets: List[str]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Union[bool, int, List[str]]]:
        return asdict(self)


class SubmissionValidator:
    """Validates submission DataFrames and CSV files against competition integrity constraints."""

    REQUIRED_COLUMNS = [
        "id",
        "dataset",
        "row_type",
        "node_id",
        "t",
        "z",
        "y",
        "x",
        "source_id",
        "target_id",
    ]

    def validate(self, submission: Union[pl.DataFrame, str, Path]) -> SubmissionValidationResult:
        """Execute comprehensive validation checks on a submission.

        Args:
            submission: Polars DataFrame or path to submission CSV file.

        Returns:
            SubmissionValidationResult indicating validity and listing any errors or warnings.
        """
        if isinstance(submission, (str, Path)):
            sub_path = Path(submission)
            if not sub_path.exists():
                return SubmissionValidationResult(
                    is_valid=False,
                    num_rows=0,
                    num_datasets=0,
                    num_nodes=0,
                    num_edges=0,
                    datasets=[],
                    errors=[f"Submission file does not exist at {sub_path}"],
                )
            df = pl.read_csv(sub_path)
        else:
            df = submission

        errors: List[str] = []
        warnings: List[str] = []

        # 1. Column presence check
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            return SubmissionValidationResult(
                is_valid=False,
                num_rows=df.height,
                num_datasets=0,
                num_nodes=0,
                num_edges=0,
                datasets=[],
                errors=errors,
            )

        n_rows = df.height
        if n_rows == 0:
            errors.append("Submission DataFrame is empty (0 rows).")
            return SubmissionValidationResult(
                is_valid=False,
                num_rows=0,
                num_datasets=0,
                num_nodes=0,
                num_edges=0,
                datasets=[],
                errors=errors,
            )

        # 2. Check null values
        for col in self.REQUIRED_COLUMNS:
            null_cnt = df[col].null_count()
            if null_cnt > 0:
                errors.append(f"Column '{col}' contains {null_cnt} null / NaN values.")

        # 3. Check ID continuity (strictly 0 to N-1)
        id_series = df["id"].to_list()
        expected_ids = list(range(n_rows))
        if id_series != expected_ids:
            errors.append("Column 'id' is not strictly sequential from 0 to N-1.")

        # 4. Check row_type vocabulary
        invalid_types = df.filter(~pl.col("row_type").is_in(["node", "edge"]))
        if invalid_types.height > 0:
            errors.append(f"Found {invalid_types.height} rows with invalid row_type (must be 'node' or 'edge').")

        # Split into nodes and edges
        nodes_df = df.filter(pl.col("row_type") == "node")
        edges_df = df.filter(pl.col("row_type") == "edge")

        num_nodes = nodes_df.height
        num_edges = edges_df.height
        datasets = sorted(df["dataset"].unique().to_list())

        if num_nodes == 0:
            errors.append("Submission contains 0 node rows.")

        # 5. Validate Node rows
        if num_nodes > 0:
            # source_id and target_id must be -1
            invalid_node_edges = nodes_df.filter((pl.col("source_id") != -1) | (pl.col("target_id") != -1))
            if invalid_node_edges.height > 0:
                errors.append(f"Found {invalid_node_edges.height} node rows with non-sentinel source_id or target_id (must be -1).")

            # coordinates and time must be non-negative
            invalid_coords = nodes_df.filter(
                (pl.col("t") < 0) | (pl.col("z") < 0) | (pl.col("y") < 0) | (pl.col("x") < 0)
            )
            if invalid_coords.height > 0:
                errors.append(f"Found {invalid_coords.height} node rows with negative coordinates or timepoints.")

            # Duplicate node_id check per dataset
            node_dups = nodes_df.group_by(["dataset", "node_id"]).len().filter(pl.col("len") > 1)
            if node_dups.height > 0:
                errors.append(f"Found {node_dups.height} duplicate (dataset, node_id) definitions.")

        # 6. Validate Edge rows & Referential Integrity
        if num_edges > 0:
            # node_id, t, z, y, x must be -1
            invalid_edge_nodes = edges_df.filter(
                (pl.col("node_id") != -1) | (pl.col("t") != -1) | (pl.col("z") != -1) | (pl.col("y") != -1) | (pl.col("x") != -1)
            )
            if invalid_edge_nodes.height > 0:
                errors.append(f"Found {invalid_edge_nodes.height} edge rows with non-sentinel node attributes (must be -1).")

            # Self loop check
            self_loops = edges_df.filter(pl.col("source_id") == pl.col("target_id"))
            if self_loops.height > 0:
                errors.append(f"Found {self_loops.height} edge rows with self-loops (source_id == target_id).")

            # Orphan edge check & Temporal causality check per dataset
            for d in datasets:
                d_nodes = nodes_df.filter(pl.col("dataset") == d)
                d_edges = edges_df.filter(pl.col("dataset") == d)

                node_id_to_t: Dict[int, int] = dict(zip(d_nodes["node_id"].to_list(), d_nodes["t"].to_list()))
                known_node_ids: Set[int] = set(node_id_to_t.keys())

                orphan_src = d_edges.filter(~pl.col("source_id").is_in(list(known_node_ids)))
                orphan_dst = d_edges.filter(~pl.col("target_id").is_in(list(known_node_ids)))

                if orphan_src.height > 0 or orphan_dst.height > 0:
                    errors.append(
                        f"Dataset '{d}' has orphan edges referencing non-existent nodes "
                        f"(src orphans: {orphan_src.height}, dst orphans: {orphan_dst.height})."
                    )

                # Temporal causality: t(target) > t(source)
                backward_edges = 0
                for u, v in zip(d_edges["source_id"].to_list(), d_edges["target_id"].to_list()):
                    if u in node_id_to_t and v in node_id_to_t:
                        if node_id_to_t[v] <= node_id_to_t[u]:
                            backward_edges += 1

                if backward_edges > 0:
                    errors.append(f"Dataset '{d}' contains {backward_edges} edges that do not flow strictly forward in time.")

        is_valid = len(errors) == 0

        return SubmissionValidationResult(
            is_valid=is_valid,
            num_rows=n_rows,
            num_datasets=len(datasets),
            num_nodes=num_nodes,
            num_edges=num_edges,
            datasets=datasets,
            errors=errors,
            warnings=warnings,
        )


def validate_submission(submission: Union[pl.DataFrame, str, Path]) -> SubmissionValidationResult:
    """Convenience function to validate submission compliance."""
    validator = SubmissionValidator()
    return validator.validate(submission)

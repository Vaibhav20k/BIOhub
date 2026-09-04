"""Kaggle Biohub competition submission writer and formatter."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import polars as pl
import tracksdata

from src.graph.candidate_graph import CandidateGraph
from src.optimization.ilp_solver import ILPSolution


@dataclass
class DatasetLineageResult:
    """Lineage results for a single dataset ready for submission serialization."""

    dataset_name: str
    nodes_df: pl.DataFrame  # ['node_id', 't', 'z', 'y', 'x']
    edges_df: pl.DataFrame  # ['source_id', 'target_id']


class SubmissionWriter:
    """Formats cell tracking lineage solutions into official Kaggle submission format."""

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

    def __init__(self, round_coordinates: bool = True):
        """
        Args:
            round_coordinates: Whether to round continuous voxel coordinates to integers or keep float.
        """
        self.round_coordinates = round_coordinates

    def create_dataset_result_from_ilp(
        self,
        dataset_name: str,
        graph: CandidateGraph,
        solution: ILPSolution,
    ) -> DatasetLineageResult:
        """Convert ILP solution on CandidateGraph into DatasetLineageResult."""
        selected_nodes = solution.selected_node_ids
        selected_edges = solution.selected_edge_ids

        node_rows = []
        for nid in selected_nodes:
            n = graph.nodes[nid]
            node_rows.append(
                {
                    "node_id": int(n.node_id),
                    "t": int(n.t),
                    "z": float(n.z),
                    "y": float(n.y),
                    "x": float(n.x),
                }
            )

        edge_rows = []
        for eid in selected_edges:
            e = graph.edges[eid]
            edge_rows.append(
                {
                    "source_id": int(e.source_id),
                    "target_id": int(e.target_id),
                }
            )

        nodes_df = pl.DataFrame(node_rows) if node_rows else pl.DataFrame(
            schema={"node_id": pl.Int64, "t": pl.Int64, "z": pl.Float64, "y": pl.Float64, "x": pl.Float64}
        )
        edges_df = pl.DataFrame(edge_rows) if edge_rows else pl.DataFrame(
            schema={"source_id": pl.Int64, "target_id": pl.Int64}
        )

        return DatasetLineageResult(
            dataset_name=dataset_name,
            nodes_df=nodes_df,
            edges_df=edges_df,
        )

    def create_dataset_result_from_tracksdata(
        self,
        dataset_name: str,
        graph: tracksdata.graph.IndexedRXGraph,
    ) -> DatasetLineageResult:
        """Extract DatasetLineageResult from a tracksdata IndexedRXGraph."""
        nodes_df = graph.node_attrs()
        edges_df = graph.edge_attrs()

        select_node_cols = ["node_id", "t", "z", "y", "x"]
        available_node_cols = [c for c in select_node_cols if c in nodes_df.columns]
        nodes_sub = nodes_df.select(available_node_cols)

        select_edge_cols = ["source_id", "target_id"]
        available_edge_cols = [c for c in select_edge_cols if c in edges_df.columns]
        edges_sub = edges_df.select(available_edge_cols)

        return DatasetLineageResult(
            dataset_name=dataset_name,
            nodes_df=nodes_sub,
            edges_df=edges_sub,
        )

    def build_submission_dataframe(
        self,
        results: List[DatasetLineageResult],
    ) -> pl.DataFrame:
        """Assemble multiple dataset lineage results into a unified Kaggle submission DataFrame.

        Args:
            results: List of DatasetLineageResult objects.

        Returns:
            Polars DataFrame strictly conforming to the Kaggle submission specification.
        """
        all_rows: List[Dict[str, Union[int, float, str]]] = []
        global_row_id = 0

        for res in results:
            d_name = res.dataset_name

            # 1. Append Node rows
            for r in res.nodes_df.to_dicts():
                z_val = round(float(r["z"])) if self.round_coordinates else float(r["z"])
                y_val = round(float(r["y"])) if self.round_coordinates else float(r["y"])
                x_val = round(float(r["x"])) if self.round_coordinates else float(r["x"])

                all_rows.append(
                    {
                        "id": global_row_id,
                        "dataset": d_name,
                        "row_type": "node",
                        "node_id": int(r["node_id"]),
                        "t": int(r["t"]),
                        "z": z_val,
                        "y": y_val,
                        "x": x_val,
                        "source_id": -1,
                        "target_id": -1,
                    }
                )
                global_row_id += 1

            # 2. Append Edge rows
            for r in res.edges_df.to_dicts():
                all_rows.append(
                    {
                        "id": global_row_id,
                        "dataset": d_name,
                        "row_type": "edge",
                        "node_id": -1,
                        "t": -1,
                        "z": -1,
                        "y": -1,
                        "x": -1,
                        "source_id": int(r["source_id"]),
                        "target_id": int(r["target_id"]),
                    }
                )
                global_row_id += 1

        if not all_rows:
            return pl.DataFrame(
                schema={
                    "id": pl.Int64,
                    "dataset": pl.Utf8,
                    "row_type": pl.Utf8,
                    "node_id": pl.Int64,
                    "t": pl.Int64,
                    "z": pl.Int64 if self.round_coordinates else pl.Float64,
                    "y": pl.Int64 if self.round_coordinates else pl.Float64,
                    "x": pl.Int64 if self.round_coordinates else pl.Float64,
                    "source_id": pl.Int64,
                    "target_id": pl.Int64,
                }
            )

        return pl.DataFrame(all_rows)

    def write_csv(
        self,
        results: List[DatasetLineageResult],
        output_path: Union[str, Path],
    ) -> Path:
        """Serialize submission to CSV on disk."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        df = self.build_submission_dataframe(results)
        df.write_csv(out_p)
        return out_p

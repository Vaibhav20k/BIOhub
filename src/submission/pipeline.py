"""End-to-end multi-dataset competition inference and submission generation pipeline."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import polars as pl
import torch

from src.data.zarr_reader import DatasetVolume, open_dataset
from src.detection.peak_detector import PeakDetector3D
from src.detection.unet3d import TemporalUNet3D
from src.graph.candidate_graph import CandidateGraphBuilder
from src.optimization.ilp_solver import LineageILPSolver
from src.representation.node_embedding import CellNodeEmbedding, extract_node_embeddings_for_dataset
from src.submission.validator import SubmissionValidationResult, SubmissionValidator
from src.submission.writer import DatasetLineageResult, SubmissionWriter
from src.tracking.transformer import SpatioTemporalTracker

logger = logging.getLogger(__name__)


class EndToEndSubmissionPipeline:
    """Full deep learning + ILP pipeline for batch submission inference on competition volumes."""

    def __init__(
        self,
        detector_checkpoint: Union[str, Path] = "checkpoints/best_detector.pt",
        tracker_checkpoint: Union[str, Path] = "checkpoints/best_tracker.pt",
        detection_threshold: float = 0.5,
        max_peaks_per_frame: Optional[int] = 50,
        max_distance_um: float = 7.0,
        allow_skips: bool = False,
        round_coordinates: bool = True,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            detector_checkpoint: Path to trained 3D Temporal U-Net detector.
            tracker_checkpoint: Path to trained Spatio-Temporal Transformer Tracker.
            detection_threshold: Peak detection cutoff.
            max_peaks_per_frame: Maximum highest-confidence detections to retain per frame (default: 50).
            max_distance_um: Candidate matching radius.
            allow_skips: Whether to build 2-frame skip candidate edges.
            round_coordinates: Round voxel coordinates in submission CSV.
            device: Compute device.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detection_threshold = detection_threshold
        self.max_peaks_per_frame = max_peaks_per_frame
        self.max_distance_um = max_distance_um
        self.allow_skips = allow_skips
        self.round_coordinates = round_coordinates

        logger.info(f"Initializing submission pipeline on device: {self.device}")

        # 1. Load Detector
        det_path = Path(detector_checkpoint)
        assert det_path.exists(), f"Detector checkpoint not found at {det_path}"
        ckpt_det = torch.load(det_path, map_location=self.device)
        self.detector = TemporalUNet3D(
            in_channels=1,
            base_channels=ckpt_det.get("config", {}).get("base_channels", 16),
            feature_dim=ckpt_det.get("config", {}).get("feature_dim", 32),
            use_temporal_attention=True,
        ).to(self.device)
        self.detector.load_state_dict(ckpt_det["model_state_dict"])
        self.detector.eval()

        # 2. Load Embedder & Tracker
        trk_path = Path(tracker_checkpoint)
        assert trk_path.exists(), f"Tracker checkpoint not found at {trk_path}"
        ckpt_trk = torch.load(trk_path, map_location=self.device)

        self.embedder = CellNodeEmbedding(
            in_visual_dim=32,
            embedding_dim=128,
        ).to(self.device)
        if "embedder_state_dict" in ckpt_trk:
            self.embedder.load_state_dict(ckpt_trk["embedder_state_dict"])
        self.embedder.eval()

        self.tracker = SpatioTemporalTracker(
            node_dim=128,
            rel_dim=32,
            hidden_dim=ckpt_trk.get("config", {}).get("hidden_dim", 128),
            num_layers=ckpt_trk.get("config", {}).get("num_layers", 2),
        ).to(self.device)
        self.tracker.load_state_dict(ckpt_trk["tracker_state_dict"])
        self.tracker.eval()

        # Helper components
        self.writer = SubmissionWriter(round_coordinates=round_coordinates)
        self.validator = SubmissionValidator()
        self.ilp_solver = LineageILPSolver()

    def process_single_volume(
        self,
        dataset: DatasetVolume,
        nodes_df: Optional[pl.DataFrame] = None,
        timepoints: Optional[List[int]] = None,
    ) -> DatasetLineageResult:
        """Run full cell tracking reconstruction for a single volume.

        Args:
            dataset: DatasetVolume instance.
            nodes_df: Optional pre-detected nodes. If None, runs 3D peak detector.
            timepoints: Optional explicit list of timepoint indices to evaluate.

        Returns:
            DatasetLineageResult containing solved nodes and edges.
        """
        logger.info(f"Processing volume '{dataset.name}' ({dataset.shape})")

        # 1. Peak detection
        if nodes_df is None:
            detector = PeakDetector3D(
                threshold=self.detection_threshold,
                scale=dataset.scale,
                max_peaks_per_frame=self.max_peaks_per_frame,
            )
            # Default to explicit timepoints, ground truth active range, or all frames
            if timepoints is not None:
                active_t = timepoints
            elif dataset.tracks is not None:
                active_t = sorted(dataset.tracks.node_attrs()["t"].unique().to_list())
            else:
                active_t = list(range(dataset.shape[0]))

            nodes_df = detector.detect_dataset_nodes(
                model=self.detector,
                dataset=dataset,
                device=self.device,
                timepoints=active_t,
            )
            logger.info(f"[{dataset.name}] Extracted {nodes_df.height} peaks across {len(active_t)} frames.")

        if nodes_df.height == 0:
            logger.warning(f"[{dataset.name}] Zero nodes detected.")
            return DatasetLineageResult(
                dataset_name=dataset.name,
                nodes_df=pl.DataFrame(schema={"node_id": pl.Int64, "t": pl.Int64, "z": pl.Float64, "y": pl.Float64, "x": pl.Float64}),
                edges_df=pl.DataFrame(schema={"source_id": pl.Int64, "target_id": pl.Int64}),
            )

        # 2. Extract multi-modal node embeddings
        sorted_nodes, node_embs = extract_node_embeddings_for_dataset(
            model=self.detector,
            embedder=self.embedder,
            dataset=dataset,
            nodes_df=nodes_df,
            device=self.device,
        )

        # 3. Assemble candidate graph
        builder = CandidateGraphBuilder(
            max_distance_um=self.max_distance_um,
            scale=dataset.scale,
            allow_frame_skips=self.allow_skips,
        )
        graph = builder.build_candidate_graph(
            nodes_df=sorted_nodes,
            tracker=self.tracker,
            node_embeddings=node_embs,
            device=self.device,
        )

        # 4. Global ILP solve
        solution = self.ilp_solver.solve(graph)
        logger.info(
            f"[{dataset.name}] Solved: {len(solution.selected_node_ids)} nodes, "
            f"{len(solution.selected_edge_ids)} edges in {solution.solve_time_sec:.2f}s"
        )

        return self.writer.create_dataset_result_from_ilp(
            dataset_name=dataset.name,
            graph=graph,
            solution=solution,
        )

    def generate_submission(
        self,
        datasets: List[Union[DatasetVolume, str, Path]],
        output_csv: Union[str, Path] = "submission.csv",
        timepoints: Optional[List[int]] = None,
    ) -> Tuple[pl.DataFrame, SubmissionValidationResult]:
        """Process all volumes and write a fully validated submission.csv file.

        Args:
            datasets: List of DatasetVolume objects or paths.
            output_csv: Path to save the output CSV.
            timepoints: Optional explicit list of timepoints to process per dataset.

        Returns:
            Tuple of (submission_dataframe, validation_result).
        """
        results: List[DatasetLineageResult] = []

        for ds_item in datasets:
            if isinstance(ds_item, DatasetVolume):
                ds = ds_item
            else:
                ds = open_dataset(ds_item, load_tracks=True)
            res = self.process_single_volume(ds, timepoints=timepoints)
            results.append(res)

        sub_df = self.writer.build_submission_dataframe(results)
        val_res = self.validator.validate(sub_df)

        if not val_res.is_valid:
            logger.error(f"Generated submission FAILED validation with {len(val_res.errors)} errors:")
            for err in val_res.errors:
                logger.error(f" - {err}")
        else:
            logger.info(f"Submission PASSED validation: {val_res.num_rows} rows across {val_res.num_datasets} datasets.")

        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sub_df.write_csv(out_path)
        logger.info(f"Saved submission to {out_path}")

        return sub_df, val_res

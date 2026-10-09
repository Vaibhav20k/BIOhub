"""Evaluation engine to benchmark full Phase 4-9 lineage reconstruction on held-out validation sequences."""

import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import polars as pl
import torch

from src.data.zarr_reader import DatasetVolume, open_dataset
from src.detection.peak_detector import PeakDetector3D
from src.detection.unet3d import TemporalUNet3D
from src.evaluation.tracking_metrics import CompetitionEvaluator, CompetitionTrackingMetrics
from src.graph.candidate_graph import CandidateGraphBuilder
from src.optimization.ilp_solver import LineageILPSolver
from src.representation.node_embedding import CellNodeEmbedding, extract_node_embeddings_for_dataset
from src.tracking.transformer import SpatioTemporalTracker

logger = logging.getLogger(__name__)


class HeldOutEvaluator:
    """Evaluates the end-to-end lineage pipeline and computes official competition metrics across validation sequences."""

    def __init__(
        self,
        detector: TemporalUNet3D,
        embedder: CellNodeEmbedding,
        tracker: SpatioTemporalTracker,
        detection_threshold: float = 0.5,
        max_peaks_per_frame: int = 50,
        max_distance_um: float = 7.0,
        diagnostic_mode: bool = False,
        window_size: int = 1,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = detector.to(self.device).eval()
        self.embedder = embedder.to(self.device).eval()
        self.tracker = tracker.to(self.device).eval()

        self.detection_threshold = detection_threshold
        self.max_peaks_per_frame = max_peaks_per_frame
        self.max_distance_um = max_distance_um
        self.diagnostic_mode = diagnostic_mode
        self.window_size = window_size

        self.ilp_solver = LineageILPSolver()
        self.competition_evaluator = CompetitionEvaluator(max_distance_um=max_distance_um)

    @classmethod
    def from_checkpoints(
        cls,
        detector_checkpoint: Union[str, Path] = "checkpoints/best_detector.pt",
        tracker_checkpoint: Union[str, Path] = "checkpoints/best_tracker.pt",
        detection_threshold: float = 0.5,
        max_peaks_per_frame: int = 50,
        max_distance_um: float = 7.0,
        diagnostic_mode: bool = False,
        window_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> "HeldOutEvaluator":
        """Factory method to load evaluator directly from checkpoint files."""
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load detector
        det_ckpt = torch.load(detector_checkpoint, map_location=dev)
        use_temporal_attn = det_ckpt.get("config", {}).get("use_temporal_attention", False)
        detector = TemporalUNet3D(
            in_channels=1,
            base_channels=det_ckpt.get("config", {}).get("base_channels", 16),
            feature_dim=det_ckpt.get("config", {}).get("feature_dim", 32),
            use_temporal_attention=use_temporal_attn,
        )
        detector.load_state_dict(det_ckpt["model_state_dict"])

        # Load tracker & embedder
        trk_ckpt = torch.load(tracker_checkpoint, map_location=dev)
        embedder = CellNodeEmbedding(in_visual_dim=32, embedding_dim=128)
        if "embedder_state_dict" in trk_ckpt:
            embedder.load_state_dict(trk_ckpt["embedder_state_dict"])

        tracker = SpatioTemporalTracker(
            node_dim=128,
            rel_dim=32,
            hidden_dim=trk_ckpt.get("config", {}).get("hidden_dim", 128),
            num_layers=trk_ckpt.get("config", {}).get("num_layers", 2),
        )
        tracker.load_state_dict(trk_ckpt["tracker_state_dict"])

        return cls(
            detector=detector,
            embedder=embedder,
            tracker=tracker,
            detection_threshold=detection_threshold,
            max_peaks_per_frame=max_peaks_per_frame,
            max_distance_um=max_distance_um,
            diagnostic_mode=diagnostic_mode,
            window_size=window_size,
            device=dev,
        )

    def evaluate_single_sequence(
        self,
        dataset: DatasetVolume,
        timepoints: Optional[List[int]] = None,
        return_timings: bool = False,
    ) -> Union[CompetitionTrackingMetrics, Tuple[CompetitionTrackingMetrics, float, float]]:
        """Run full Phase 4-9 lineage reconstruction on a single sequence and score against ground truth."""
        assert dataset.tracks is not None, f"Ground truth tracks required for validation of {dataset.name}"

        t_infer_start = time.time()

        # 1. Peak detection
        detector = PeakDetector3D(
            threshold=self.detection_threshold,
            scale=dataset.scale,
            max_peaks_per_frame=self.max_peaks_per_frame,
            diagnostic_mode=self.diagnostic_mode,
        )

        active_t = (
            timepoints
            if timepoints is not None
            else sorted(dataset.tracks.node_attrs()["t"].unique().to_list())
        )

        nodes_df = detector.detect_dataset_nodes(
            model=self.detector,
            dataset=dataset,
            device=self.device,
            timepoints=active_t,
            window_size=self.window_size,
        )

        if nodes_df.height == 0:
            logger.warning(f"[{dataset.name}] 0 nodes detected.")
            empty_m = CompetitionTrackingMetrics(
                node_tp=0,
                node_fp=0,
                node_fn=dataset.tracks.num_nodes(),
                node_precision=0.0,
                node_recall=0.0,
                node_f1=0.0,
                node_jaccard=0.0,
                pred_to_gt_ratio=0.0,
                j_adj=0.0,
                gt_divisions=0,
                pred_divisions=0,
                div_tp=0,
                div_fp=0,
                div_fn=0,
                j_div=0.0,
                final_score=0.0,
                edge_tp=0,
                edge_fp=0,
                edge_fn=dataset.tracks.num_edges(),
                edge_precision=0.0,
                edge_recall=0.0,
                edge_f1=0.0,
                edge_jaccard=0.0,
            )
            return (empty_m, time.time() - t_infer_start, 0.0) if return_timings else empty_m

        # 2. Node Embeddings
        sorted_nodes, node_embs = extract_node_embeddings_for_dataset(
            model=self.detector,
            embedder=self.embedder,
            dataset=dataset,
            nodes_df=nodes_df,
            device=self.device,
        )

        # 3. Candidate Graph
        builder = CandidateGraphBuilder(
            max_distance_um=self.max_distance_um,
            scale=dataset.scale,
            allow_frame_skips=False,
        )
        graph = builder.build_candidate_graph(
            nodes_df=sorted_nodes,
            tracker=self.tracker,
            node_embeddings=node_embs,
            device=self.device,
        )
        inference_time_sec = time.time() - t_infer_start

        # 4. ILP Solve
        t_ilp_start = time.time()
        solution = self.ilp_solver.solve(graph)
        ilp_runtime_sec = time.time() - t_ilp_start

        # 5. Extract solved nodes & edges
        solved_nodes_df = sorted_nodes.filter(pl.col("node_id").is_in(solution.selected_node_ids))
        selected_edge_ids_set = set(solution.selected_edge_ids)
        solved_edges_list = [
            {"source_id": edge.source_id, "target_id": edge.target_id}
            for eid, edge in graph.edges.items()
            if eid in selected_edge_ids_set
        ]
        solved_edges_df = pl.DataFrame(solved_edges_list) if solved_edges_list else pl.DataFrame(schema={"source_id": pl.Int64, "target_id": pl.Int64})

        # 6. Score against ground truth
        gt_nodes = dataset.tracks.node_attrs()
        gt_edges = dataset.tracks.edge_attrs()

        metrics = self.competition_evaluator.evaluate_lineages(
            pred_nodes=solved_nodes_df,
            pred_edges=solved_edges_df,
            gt_nodes=gt_nodes,
            gt_edges=gt_edges,
        )
        return (metrics, inference_time_sec, ilp_runtime_sec) if return_timings else metrics

    def evaluate_cohort(
        self,
        datasets: List[Union[DatasetVolume, str, Path]],
        timepoints: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Evaluate full pipeline across multiple validation sequences and compute micro/macro metrics."""
        sequence_metrics: Dict[str, Dict[str, Any]] = {}
        total_adj_jaccard = 0.0
        total_div_jaccard = 0.0
        total_final_score = 0.0
        total_node_prec = 0.0
        total_node_rec = 0.0
        total_edge_jaccard = 0.0
        total_node_tp = 0
        total_node_fp = 0
        total_node_fn = 0
        total_infer_time = 0.0
        total_ilp_time = 0.0

        for ds_item in datasets:
            ds = ds_item if isinstance(ds_item, DatasetVolume) else open_dataset(ds_item, load_tracks=True, require_tracks=True)
            logger.info(f"Evaluating held-out sequence: {ds.name}...")
            m, t_infer, t_ilp = self.evaluate_single_sequence(ds, timepoints=timepoints, return_timings=True)
            m_dict = m.to_dict()
            m_dict["inference_time_sec"] = t_infer
            m_dict["ilp_runtime_sec"] = t_ilp
            sequence_metrics[ds.name] = m_dict
            logger.info(
                f"[{ds.name}] Score={m.final_score:.4f} (J_adj={m.j_adj:.4f}, J_div={m.j_div:.4f}, "
                f"Edge_J={m.edge_jaccard:.4f}, Rec={m.node_recall:.4f}, Prec={m.node_precision:.4f}, "
                f"TP={m.node_tp}, FP={m.node_fp}, FN={m.node_fn}, Infer={t_infer:.2f}s, ILP={t_ilp:.2f}s)"
            )

            total_adj_jaccard += m.j_adj
            total_div_jaccard += m.j_div
            total_final_score += m.final_score
            total_node_prec += m.node_precision
            total_node_rec += m.node_recall
            total_edge_jaccard += m.edge_jaccard
            total_node_tp += m.node_tp
            total_node_fp += m.node_fp
            total_node_fn += m.node_fn
            total_infer_time += t_infer
            total_ilp_time += t_ilp

        n = max(1, len(datasets))
        summary = {
            "num_sequences": len(datasets),
            "macro_adjusted_jaccard": total_adj_jaccard / n,
            "macro_division_jaccard": total_div_jaccard / n,
            "macro_final_score": total_final_score / n,
            "macro_node_precision": total_node_prec / n,
            "macro_node_recall": total_node_rec / n,
            "macro_edge_jaccard": total_edge_jaccard / n,
            "total_node_tp": total_node_tp,
            "total_node_fp": total_node_fp,
            "total_node_fn": total_node_fn,
            "inference_time_sec": total_infer_time,
            "ilp_runtime_sec": total_ilp_time,
            "sequences": sequence_metrics,
        }
        return summary

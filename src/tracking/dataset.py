"""Dataset and batching utilities for training Spatio-Temporal Transformer tracking models."""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from src.data.zarr_reader import DatasetVolume, DEFAULT_SCALE
from src.representation.node_embedding import CellNodeEmbedding, extract_node_embeddings_for_dataset
from src.tracking.candidate_edges import CandidateEdgeBatch, CandidateEdgeBuilder


class TrackingPairDataset(Dataset):
    """Generates consecutive frame pairs (t, t+1) with candidate edges and ground-truth supervision."""

    def __init__(
        self,
        dataset: DatasetVolume,
        embedder: CellNodeEmbedding,
        detector_model: Optional[torch.nn.Module] = None,
        nodes_df: Optional[pl.DataFrame] = None,
        device: torch.device = torch.device("cpu"),
        max_distance_um: float = 7.0,
        scale: Tuple[float, float, float] = DEFAULT_SCALE,
    ):
        """
        Args:
            dataset: DatasetVolume with track annotations.
            embedder: CellNodeEmbedding instance.
            detector_model: Optional 3D U-Net detector model to extract features (if None, synthetic/dummy embeddings used).
            nodes_df: Optional DataFrame of nodes. Defaults to dataset.tracks.node_attrs().
            device: Compute device.
            max_distance_um: Maximum candidate matching cutoff in micrometers.
            scale: Physical scale (s_z, s_y, s_x).
        """
        super().__init__()
        assert dataset.tracks is not None, "TrackingPairDataset requires track annotations"
        self.dataset = dataset
        self.embedder = embedder
        self.scale = scale
        self.max_distance_um = max_distance_um

        self.edge_builder = CandidateEdgeBuilder(max_distance_um=max_distance_um, scale=scale)

        # Ground truth edges from tracks
        gt_edges_df = dataset.tracks.edge_attrs()
        self.gt_edges: Set[Tuple[int, int]] = set(
            zip(gt_edges_df["source_id"].to_list(), gt_edges_df["target_id"].to_list())
        )

        # Identify dividing source nodes (out-degree > 1)
        src_counts = gt_edges_df["source_id"].value_counts()
        self.dividing_nodes: Set[int] = set(
            src_counts.filter(pl.col("count") > 1)["source_id"].to_list()
        )

        # Nodes to track
        if nodes_df is None:
            raw_nodes = dataset.tracks.node_attrs()
            if "score" not in raw_nodes.columns:
                # Assign 1.0 confidence score to ground truth nodes
                raw_nodes = raw_nodes.with_columns(pl.lit(1.0).alias("score"))
            self.nodes_df = raw_nodes
        else:
            self.nodes_df = nodes_df

        # Group nodes by timepoint
        self.timepoints = sorted(self.nodes_df["t"].unique().to_list())
        # Filter to adjacent consecutive pairs (t_i, t_{i+1})
        self.frame_pairs: List[Tuple[int, int]] = []
        for i in range(len(self.timepoints) - 1):
            t_curr = self.timepoints[i]
            t_next = self.timepoints[i + 1]
            if t_next == t_curr + 1:  # strictly consecutive
                self.frame_pairs.append((t_curr, t_next))

        # Precompute/cache node embeddings per timepoint
        self.cached_embeddings: Dict[int, torch.Tensor] = {}
        self.cached_nodes_by_t: Dict[int, pl.DataFrame] = {}

        if detector_model is not None:
            sorted_nodes, all_embs = extract_node_embeddings_for_dataset(
                model=detector_model,
                embedder=embedder,
                dataset=dataset,
                nodes_df=self.nodes_df,
                device=device,
            )
            offset = 0
            for t in self.timepoints:
                t_df = sorted_nodes.filter(pl.col("t") == t)
                n_t = t_df.height
                self.cached_nodes_by_t[t] = t_df
                self.cached_embeddings[t] = all_embs[offset : offset + n_t]
                offset += n_t
        else:
            # Fallback: compute embeddings from Fourier positional encoder + default features
            for t in self.timepoints:
                t_df = self.nodes_df.filter(pl.col("t") == t)
                self.cached_nodes_by_t[t] = t_df
                n_t = t_df.height
                if n_t > 0:
                    coords_zyx = t_df.select(["z", "y", "x"]).to_numpy().astype(np.float32)
                    scores = t_df.select(["score"]).to_numpy().astype(np.float32)
                    coords_tzyx = np.hstack([np.full((n_t, 1), float(t), dtype=np.float32), coords_zyx])

                    with torch.no_grad():
                        dummy_vis = torch.zeros((n_t, 64), dtype=torch.float32)
                        embs = embedder(
                            visual_features=dummy_vis,
                            coords_tzyx=torch.from_numpy(coords_tzyx),
                            scores=torch.from_numpy(scores),
                        )
                    self.cached_embeddings[t] = embs
                else:
                    self.cached_embeddings[t] = torch.empty((0, embedder.embedding_dim))

    def __len__(self) -> int:
        return len(self.frame_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, CandidateEdgeBatch]:
        """Returns (src_embeddings, dst_embeddings, candidate_batch) for frame pair."""
        t_src, t_dst = self.frame_pairs[idx]

        src_nodes = self.cached_nodes_by_t[t_src]
        dst_nodes = self.cached_nodes_by_t[t_dst]

        src_embs = self.cached_embeddings[t_src]
        dst_embs = self.cached_embeddings[t_dst]

        candidate_batch = self.edge_builder.build_frame_pair_candidates(
            src_nodes=src_nodes,
            dst_nodes=dst_nodes,
            gt_edges=self.gt_edges,
            dividing_nodes=self.dividing_nodes,
        )

        return src_embs, dst_embs, candidate_batch

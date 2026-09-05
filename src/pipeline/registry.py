"""Persistent SQLite and JSONL registry for tracking chunked dataset processing and experiment results."""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


@dataclass
class ChunkRecord:
    """Comprehensive tracking record for a single dataset chunk."""

    chunk_id: str
    sequence_stems: List[str]
    embryo_group: str
    chunk_size_bytes: int
    role: str = "train"  # "train", "validation", "test"
    base_checkpoint: Optional[str] = None
    resulting_checkpoint: Optional[str] = None
    epochs_trained: int = 0
    train_loss_detector: Optional[float] = None
    train_loss_tracker: Optional[float] = None
    val_sequences_evaluated: List[str] = field(default_factory=list)
    val_node_recall: Optional[float] = None
    val_node_precision: Optional[float] = None
    val_mean_dist_um: Optional[float] = None
    val_edge_jaccard: Optional[float] = None
    val_div_jaccard: Optional[float] = None
    val_adj_jaccard: Optional[float] = None
    val_final_score: Optional[float] = None
    promoted_to_best: bool = False
    status: str = "PENDING"  # PENDING, DOWNLOADED, PROCESSED, EVALUATED, CLEANED_UP, FAILED, ROLLED_BACK
    disk_freed_bytes: int = 0
    cleanup_verified: bool = False
    duration_sec: float = 0.0
    error_message: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with JSON-serialized list fields."""
        data = asdict(self)
        return data


class ChunkRegistry:
    """Manages transactional SQLite storage and append-only JSONL logging of chunk experiments."""

    def __init__(
        self,
        sqlite_path: Union[str, Path] = "registry/experiments.sqlite",
        jsonl_path: Union[str, Path] = "registry/chunk_registry.jsonl",
    ):
        self.sqlite_path = Path(sqlite_path)
        self.jsonl_path = Path(jsonl_path)

        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a sqlite3 connection with Row factory."""
        conn = sqlite3.connect(str(self.sqlite_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize SQLite database tables and indices if not present."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    sequence_stems TEXT NOT NULL,
                    embryo_group TEXT NOT NULL,
                    chunk_size_bytes INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    base_checkpoint TEXT,
                    resulting_checkpoint TEXT,
                    epochs_trained INTEGER DEFAULT 0,
                    train_loss_detector REAL,
                    train_loss_tracker REAL,
                    val_sequences_evaluated TEXT,
                    val_node_recall REAL,
                    val_node_precision REAL,
                    val_mean_dist_um REAL,
                    val_edge_jaccard REAL,
                    val_div_jaccard REAL,
                    val_adj_jaccard REAL,
                    val_final_score REAL,
                    promoted_to_best INTEGER DEFAULT 0,
                    status TEXT NOT NULL,
                    disk_freed_bytes INTEGER DEFAULT 0,
                    cleanup_verified INTEGER DEFAULT 0,
                    duration_sec REAL DEFAULT 0.0,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_status ON chunks(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_role ON chunks(role)")
            conn.commit()

    def register_manifest(self, manifest_records: List[ChunkRecord]) -> int:
        """Register pre-computed chunk partitions into the database if not already present.

        Returns number of newly registered records.
        """
        new_count = 0
        with self._get_connection() as conn:
            for rec in manifest_records:
                cur = conn.execute("SELECT 1 FROM chunks WHERE chunk_id = ?", (rec.chunk_id,))
                if cur.fetchone() is None:
                    conn.execute(
                        """
                        INSERT INTO chunks (
                            chunk_id, sequence_stems, embryo_group, chunk_size_bytes, role,
                            status, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            rec.chunk_id,
                            json.dumps(rec.sequence_stems),
                            rec.embryo_group,
                            rec.chunk_size_bytes,
                            rec.role,
                            rec.status,
                            rec.created_at,
                        ),
                    )
                    new_count += 1
            conn.commit()
        logger.info(f"Registered {new_count} new chunk records into registry.")
        return new_count

    def update_chunk(self, record: ChunkRecord) -> None:
        """Update existing chunk record in SQLite and append to JSONL log."""
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE chunks SET
                    sequence_stems = ?,
                    embryo_group = ?,
                    chunk_size_bytes = ?,
                    role = ?,
                    base_checkpoint = ?,
                    resulting_checkpoint = ?,
                    epochs_trained = ?,
                    train_loss_detector = ?,
                    train_loss_tracker = ?,
                    val_sequences_evaluated = ?,
                    val_node_recall = ?,
                    val_node_precision = ?,
                    val_mean_dist_um = ?,
                    val_edge_jaccard = ?,
                    val_div_jaccard = ?,
                    val_adj_jaccard = ?,
                    val_final_score = ?,
                    promoted_to_best = ?,
                    status = ?,
                    disk_freed_bytes = ?,
                    cleanup_verified = ?,
                    duration_sec = ?,
                    error_message = ?,
                    completed_at = ?
                WHERE chunk_id = ?
                """,
                (
                    json.dumps(record.sequence_stems),
                    record.embryo_group,
                    record.chunk_size_bytes,
                    record.role,
                    record.base_checkpoint,
                    record.resulting_checkpoint,
                    record.epochs_trained,
                    record.train_loss_detector,
                    record.train_loss_tracker,
                    json.dumps(record.val_sequences_evaluated),
                    record.val_node_recall,
                    record.val_node_precision,
                    record.val_mean_dist_um,
                    record.val_edge_jaccard,
                    record.val_div_jaccard,
                    record.val_adj_jaccard,
                    record.val_final_score,
                    1 if record.promoted_to_best else 0,
                    record.status,
                    record.disk_freed_bytes,
                    1 if record.cleanup_verified else 0,
                    record.duration_sec,
                    record.error_message,
                    record.completed_at,
                    record.chunk_id,
                ),
            )
            conn.commit()

        # Append to JSONL audit log
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

    def get_chunk(self, chunk_id: str) -> Optional[ChunkRecord]:
        """Fetch chunk by ID."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,))
            row = cur.fetchone()
            if row is None:
                return None
            return self._row_to_record(row)

    def get_next_pending_chunk(self) -> Optional[ChunkRecord]:
        """Fetch the first chunk with status 'PENDING' ordered by chunk_id."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM chunks WHERE status = 'PENDING' ORDER BY chunk_id ASC LIMIT 1")
            row = cur.fetchone()
            if row is None:
                return None
            return self._row_to_record(row)

    def get_completed_sequences(self) -> Set[str]:
        """Return set of all sequence stems from successfully completed chunks."""
        completed = set()
        with self._get_connection() as conn:
            cur = conn.execute("SELECT sequence_stems FROM chunks WHERE status IN ('PROCESSED', 'CLEANED_UP')")
            for row in cur.fetchall():
                stems = json.loads(row["sequence_stems"])
                completed.update(stems)
        return completed

    def get_best_score(self) -> float:
        """Return the highest val_final_score achieved so far across completed chunks."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT MAX(val_final_score) as best FROM chunks WHERE val_final_score IS NOT NULL")
            row = cur.fetchone()
            if row and row["best"] is not None:
                return float(row["best"])
        return 0.0

    def get_all_chunks(self) -> List[ChunkRecord]:
        """Fetch all chunks ordered by chunk_id."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM chunks ORDER BY chunk_id ASC")
            return [self._row_to_record(r) for r in cur.fetchall()]

    def get_summary(self) -> Dict[str, Any]:
        """Aggregate high-level workflow execution statistics."""
        with self._get_connection() as conn:
            total_cur = conn.execute("SELECT COUNT(*), SUM(chunk_size_bytes) FROM chunks")
            total_count, total_bytes = total_cur.fetchone()

            completed_cur = conn.execute("SELECT COUNT(*), SUM(chunk_size_bytes), SUM(disk_freed_bytes) FROM chunks WHERE status = 'CLEANED_UP'")
            comp_count, comp_bytes, freed_bytes = completed_cur.fetchone()

            failed_cur = conn.execute("SELECT COUNT(*) FROM chunks WHERE status = 'FAILED'")
            failed_count = failed_cur.fetchone()[0]

            best_cur = conn.execute("SELECT MAX(val_final_score) FROM chunks")
            best_score = best_cur.fetchone()[0]

        return {
            "total_chunks": total_count or 0,
            "completed_chunks": comp_count or 0,
            "failed_chunks": failed_count or 0,
            "total_size_gb": (total_bytes or 0) / (1024**3),
            "processed_size_gb": (comp_bytes or 0) / (1024**3),
            "disk_freed_gb": (freed_bytes or 0) / (1024**3),
            "best_validation_score": best_score if best_score is not None else 0.0,
        }

    def _row_to_record(self, row: sqlite3.Row) -> ChunkRecord:
        """Convert a database row into a ChunkRecord dataclass instance."""
        return ChunkRecord(
            chunk_id=row["chunk_id"],
            sequence_stems=json.loads(row["sequence_stems"]),
            embryo_group=row["embryo_group"],
            chunk_size_bytes=row["chunk_size_bytes"],
            role=row["role"],
            base_checkpoint=row["base_checkpoint"],
            resulting_checkpoint=row["resulting_checkpoint"],
            epochs_trained=row["epochs_trained"],
            train_loss_detector=row["train_loss_detector"],
            train_loss_tracker=row["train_loss_tracker"],
            val_sequences_evaluated=json.loads(row["val_sequences_evaluated"]) if row["val_sequences_evaluated"] else [],
            val_node_recall=row["val_node_recall"],
            val_node_precision=row["val_node_precision"],
            val_mean_dist_um=row["val_mean_dist_um"],
            val_edge_jaccard=row["val_edge_jaccard"],
            val_div_jaccard=row["val_div_jaccard"],
            val_adj_jaccard=row["val_adj_jaccard"],
            val_final_score=row["val_final_score"],
            promoted_to_best=bool(row["promoted_to_best"]),
            status=row["status"],
            disk_freed_bytes=row["disk_freed_bytes"],
            cleanup_verified=bool(row["cleanup_verified"]),
            duration_sec=row["duration_sec"],
            error_message=row["error_message"],
            created_at=row["created_at"],
            completed_at=row["completed_at"],
        )

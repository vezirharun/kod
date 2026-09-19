"""Per-file heavy-pipeline timing. Additive; does not rewrite old job rows."""

from __future__ import annotations

import sqlite3
import threading
import time
from typing import Any

STAGE_COLS = (
    "hash",
    "metadata",
    "dino",
    "clip",
    "texture",
    "semantic",
    "dna",
    "ai_final",
    "patch",
    "ocr",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS index_v3_file_timings (
    file_id INTEGER PRIMARY KEY,
    hash_started_at REAL, hash_ended_at REAL, hash_ms REAL,
    metadata_started_at REAL, metadata_ended_at REAL, metadata_ms REAL,
    dino_started_at REAL, dino_ended_at REAL, dino_ms REAL,
    clip_started_at REAL, clip_ended_at REAL, clip_ms REAL,
    texture_started_at REAL, texture_ended_at REAL, texture_ms REAL,
    semantic_started_at REAL, semantic_ended_at REAL, semantic_ms REAL,
    dna_started_at REAL, dna_ended_at REAL, dna_ms REAL,
    ai_final_started_at REAL, ai_final_ended_at REAL, ai_final_ms REAL,
    patch_started_at REAL, patch_ended_at REAL, patch_ms REAL,
    ocr_started_at REAL, ocr_ended_at REAL, ocr_ms REAL,
    queue_wait_ms REAL,
    worker_wait_ms REAL,
    file_read_ms REAL,
    total_elapsed_ms REAL,
    last_claimed_at REAL
);
"""

_GA_STARTS = (
    "hash_started_at",
    "metadata_started_at",
    "dino_started_at",
    "clip_started_at",
    "texture_started_at",
    "semantic_started_at",
    "dna_started_at",
)


def ensure_timing_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)


class FileTimingStore:
    """One row per file_id. Existing stage timestamps are never overwritten."""

    def __init__(self, db_path: str) -> None:
        self.db_path = str(db_path)
        self._lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        ensure_timing_schema(conn)
        return conn

    def stage_start(
        self,
        file_id: int,
        stage: str,
        *,
        claimed_at: float = 0.0,
        queue_wait_ms: float | None = None,
        worker_wait_ms: float = 0.0,
    ) -> float:
        started = time.time()
        if stage not in STAGE_COLS:
            return started
        fid = int(file_id)
        with self._lock, self._connect() as conn:
            ensure_timing_schema(conn)
            conn.execute(
                "INSERT OR IGNORE INTO index_v3_file_timings(file_id) VALUES (?)",
                (fid,),
            )
            s_col = f"{stage}_started_at"
            e_col = f"{stage}_ended_at"
            conn.execute(
                f"""
                UPDATE index_v3_file_timings SET
                    last_claimed_at=CASE
                        WHEN COALESCE(last_claimed_at,0)=0 AND ? > 0 THEN ?
                        ELSE last_claimed_at END,
                    queue_wait_ms=CASE WHEN {e_col} IS NULL
                        THEN COALESCE(queue_wait_ms,0)+COALESCE(?,0)
                        ELSE queue_wait_ms END,
                    worker_wait_ms=CASE WHEN {e_col} IS NULL
                        THEN COALESCE(worker_wait_ms,0)+COALESCE(?,0)
                        ELSE worker_wait_ms END,
                    {s_col}=CASE WHEN {e_col} IS NULL THEN ? ELSE {s_col} END
                WHERE file_id=?
                """,
                (
                    float(claimed_at or 0),
                    float(claimed_at or 0),
                    queue_wait_ms,
                    float(worker_wait_ms or 0),
                    started,
                    fid,
                ),
            )
            conn.commit()
        return started

    def stage_end(
        self,
        file_id: int,
        stage: str,
        *,
        started_at: float,
        file_read_ms: float = 0.0,
    ) -> None:
        if stage not in STAGE_COLS:
            return
        ended = time.time()
        elapsed = max(0.0, (ended - float(started_at)) * 1000.0)
        fid = int(file_id)
        with self._lock, self._connect() as conn:
            ensure_timing_schema(conn)
            conn.execute(
                "INSERT OR IGNORE INTO index_v3_file_timings(file_id) VALUES (?)",
                (fid,),
            )
            e_col = f"{stage}_ended_at"
            m_col = f"{stage}_ms"
            conn.execute(
                f"""
                UPDATE index_v3_file_timings SET
                    {e_col}=CASE WHEN {e_col} IS NULL THEN ? ELSE {e_col} END,
                    {m_col}=CASE WHEN {m_col} IS NULL THEN ? ELSE {m_col} END,
                    file_read_ms=COALESCE(file_read_ms,0)+COALESCE(?,0)
                WHERE file_id=?
                """,
                (ended, elapsed, float(file_read_ms or 0), fid),
            )
            if stage == "dna":
                row = conn.execute(
                    "SELECT * FROM index_v3_file_timings WHERE file_id=?",
                    (fid,),
                ).fetchone()
                if row is not None and row["ai_final_ended_at"] is None:
                    starts = [
                        float(row[c])
                        for c in _GA_STARTS
                        if row[c] is not None
                    ]
                    af_start = min(starts) if starts else float(started_at)
                    af_end = ended
                    conn.execute(
                        """
                        UPDATE index_v3_file_timings SET
                            ai_final_started_at=?,
                            ai_final_ended_at=?,
                            ai_final_ms=?
                        WHERE file_id=? AND ai_final_ended_at IS NULL
                        """,
                        (
                            af_start,
                            af_end,
                            max(0.0, (af_end - af_start) * 1000.0),
                            fid,
                        ),
                    )
            self._refresh_total(conn, fid)
            conn.commit()

    def _refresh_total(self, conn: sqlite3.Connection, file_id: int) -> None:
        row = conn.execute(
            "SELECT * FROM index_v3_file_timings WHERE file_id=?",
            (int(file_id),),
        ).fetchone()
        if row is None:
            return
        starts: list[float] = []
        ends: list[float] = []
        for st in STAGE_COLS:
            if st == "ai_final":
                continue
            s, e = row[f"{st}_started_at"], row[f"{st}_ended_at"]
            if s is not None:
                starts.append(float(s))
            if e is not None:
                ends.append(float(e))
        if not starts or not ends:
            return
        total = max(0.0, (max(ends) - min(starts)) * 1000.0)
        conn.execute(
            "UPDATE index_v3_file_timings SET total_elapsed_ms=? WHERE file_id=?",
            (total, int(file_id)),
        )

    def get(self, file_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            ensure_timing_schema(conn)
            row = conn.execute(
                "SELECT * FROM index_v3_file_timings WHERE file_id=?",
                (int(file_id),),
            ).fetchone()
            return dict(row) if row is not None else None

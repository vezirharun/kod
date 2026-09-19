"""Persistent candidate store for archive self-audit findings.

Sidecar SQLite next to patterns.db. Never mutates file classifications.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any


def default_store_path(db_path: str) -> str:
    root = Path(str(db_path or "")).expanduser().resolve().parent
    return str(root / "archive_intelligence.db")


class ArchiveIntelligenceStore:
    def __init__(self, path: str) -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._ensure()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure(self) -> None:
        with self._connect() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS ai_candidates (
                    file_id INTEGER PRIMARY KEY,
                    kind TEXT NOT NULL,
                    reason TEXT DEFAULT '',
                    consistency REAL DEFAULT 0,
                    guess TEXT DEFAULT '',
                    labels_json TEXT DEFAULT '[]',
                    status TEXT DEFAULT 'pending',
                    scanned_at REAL DEFAULT 0,
                    updated_at REAL DEFAULT 0
                )
                """
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_ai_cand_status "
                "ON ai_candidates(status, consistency)"
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS ai_scan_cursor (
                    id INTEGER PRIMARY KEY CHECK (id=1),
                    last_file_id INTEGER DEFAULT 0,
                    updated_at REAL DEFAULT 0
                )
                """
            )
            c.execute(
                "INSERT OR IGNORE INTO ai_scan_cursor(id, last_file_id, updated_at) "
                "VALUES (1, 0, 0)"
            )

    def upsert_candidate(
        self,
        *,
        file_id: int,
        kind: str,
        reason: str,
        consistency: float,
        guess: str = "",
        labels_json: str = "[]",
        status: str = "pending",
    ) -> None:
        fid = int(file_id)
        if fid <= 0:
            return
        now = time.time()
        with self._connect() as c:
            c.execute(
                """
                INSERT INTO ai_candidates(
                    file_id, kind, reason, consistency, guess, labels_json,
                    status, scanned_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(file_id) DO UPDATE SET
                    kind=excluded.kind,
                    reason=excluded.reason,
                    consistency=excluded.consistency,
                    guess=excluded.guess,
                    labels_json=excluded.labels_json,
                    status=CASE
                        WHEN ai_candidates.status IN ('resolved','dismissed')
                             AND excluded.status='pending'
                        THEN ai_candidates.status
                        ELSE excluded.status
                    END,
                    scanned_at=excluded.scanned_at,
                    updated_at=excluded.updated_at
                """,
                (
                    fid,
                    str(kind or "suspicious"),
                    str(reason or "")[:400],
                    float(consistency),
                    str(guess or "")[:200],
                    str(labels_json or "[]")[:2000],
                    str(status or "pending"),
                    now,
                    now,
                ),
            )

    def clear_ok(self, file_id: int) -> None:
        """Drop pending finding when file looks consistent."""
        with self._connect() as c:
            c.execute(
                "DELETE FROM ai_candidates WHERE file_id=? AND status='pending'",
                (int(file_id),),
            )

    def resolve_files(self, file_ids: list[int], *, status: str = "resolved") -> int:
        ids = [int(x) for x in file_ids if int(x) > 0]
        if not ids:
            return 0
        now = time.time()
        ph = ",".join("?" * len(ids))
        with self._connect() as c:
            cur = c.execute(
                f"UPDATE ai_candidates SET status=?, updated_at=? "
                f"WHERE file_id IN ({ph}) AND status='pending'",
                (str(status), now, *ids),
            )
            return int(cur.rowcount or 0)

    def list_pending(self, *, limit: int = 40) -> list[dict[str, Any]]:
        with self._connect() as c:
            rows = c.execute(
                """
                SELECT file_id, kind, reason, consistency, guess, labels_json,
                       status, scanned_at
                FROM ai_candidates
                WHERE status='pending'
                ORDER BY consistency ASC, scanned_at ASC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_cursor(self) -> int:
        with self._connect() as c:
            row = c.execute(
                "SELECT last_file_id FROM ai_scan_cursor WHERE id=1"
            ).fetchone()
        return int(row["last_file_id"] if row else 0)

    def set_cursor(self, last_file_id: int) -> None:
        with self._connect() as c:
            c.execute(
                "UPDATE ai_scan_cursor SET last_file_id=?, updated_at=? WHERE id=1",
                (int(last_file_id), time.time()),
            )

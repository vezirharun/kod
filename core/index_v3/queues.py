"""Index Engine V3 — in-memory + SQLite job queues (isolated from legacy index_queue)."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

from core.index_v3.jumbo_phase import (
    JUMBO_AVAILABLE_AT_FAR,
    JUMBO_DEFER_MARKER,
    is_jumbo_for_queue,
    light_artifact_names,
)
from core.index_v3.timings import FileTimingStore, ensure_timing_schema
from core.index_v3.types import Artifact, Job, JobState, QueueKind
_SCHEMA = """
CREATE TABLE IF NOT EXISTS index_v3_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,
    artifact TEXT NOT NULL,
    queue TEXT NOT NULL,
    source_id INTEGER NOT NULL DEFAULT 0,
    path TEXT NOT NULL DEFAULT '',
    state TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    worker TEXT NOT NULL DEFAULT '',
    error_msg TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT '',
    available_at REAL NOT NULL DEFAULT 0,
    claimed_at REAL NOT NULL DEFAULT 0,
    heartbeat_at REAL NOT NULL DEFAULT 0,
    tail_seq INTEGER NOT NULL DEFAULT 0,
    UNIQUE(file_id, artifact)
);
CREATE INDEX IF NOT EXISTS idx_v3_jobs_queue_state
  ON index_v3_jobs(queue, state, id);
"""


class JobStore:
    """Idempotent artifact jobs — one (file_id, artifact) at a time."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self.timings = FileTimingStore(self.db_path)
        self._ensure()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            # Existing V3 job DBs were created before dependency backoff.
            cols = {
                str(r["name"])
                for r in conn.execute("PRAGMA table_info(index_v3_jobs)").fetchall()
            }
            if "available_at" not in cols:
                conn.execute(
                    "ALTER TABLE index_v3_jobs ADD COLUMN available_at REAL NOT NULL DEFAULT 0"
                )
            if "claimed_at" not in cols:
                conn.execute(
                    "ALTER TABLE index_v3_jobs ADD COLUMN claimed_at REAL NOT NULL DEFAULT 0"
                )
            if "heartbeat_at" not in cols:
                conn.execute(
                    "ALTER TABLE index_v3_jobs ADD COLUMN heartbeat_at REAL NOT NULL DEFAULT 0"
                )
            if "tail_seq" not in cols:
                conn.execute(
                    "ALTER TABLE index_v3_jobs ADD COLUMN tail_seq INTEGER NOT NULL DEFAULT 0"
                )
            if "enqueued_at" not in cols:
                conn.execute(
                    "ALTER TABLE index_v3_jobs ADD COLUMN enqueued_at REAL NOT NULL DEFAULT 0"
                )
            if "last_claimed_at" not in cols:
                conn.execute(
                    "ALTER TABLE index_v3_jobs ADD COLUMN last_claimed_at REAL NOT NULL DEFAULT 0"
                )
            ensure_timing_schema(conn)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_v3_jobs_available "
                "ON index_v3_jobs(queue, state, available_at, id)"
            )
            conn.commit()

    @staticmethod
    def _jumbo_defer_pair(
        job: Job, settings: Any | None
    ) -> tuple[float, str] | None:
        """Return (available_at, error_msg) for Phase-1 deferred jumbo light jobs."""
        if job.artifact.value not in light_artifact_names():
            return None
        if not is_jumbo_for_queue(
            str(job.path or ""),
            int(getattr(job, "file_size", 0) or 0),
            settings,
        ):
            return None
        return float(JUMBO_AVAILABLE_AT_FAR), JUMBO_DEFER_MARKER

    def enqueue(
        self,
        jobs: list[Job],
        *,
        reopen_permanent: bool = False,
        reopen_done: bool = False,
        settings: Any | None = None,
    ) -> int:
        """Insert/reopen artifact jobs idempotently.

        ``reopen_done`` yalnız planner'ın o artifact'i gerçekten eksik olarak
        raporladığı gap-reconciliation yolunda kullanılmalıdır. Böylece diskte
        silinmiş bir thumbnail/preview için eski ``done`` satırı tekrar
        çalıştırılabilir; normal keşif tamamlanmış işleri gereksiz yere açmaz.

        Jumbo TIFF light jobs (``fast_tif_defer_mb``) get ``available_at=FAR``
        so Phase-1 only claims normal FAST work.
        """
        if not jobs:
            return 0
        n = 0
        with self._lock, self._connect() as conn:
            for job in jobs:
                defer = self._jumbo_defer_pair(job, settings)
                avail = float(defer[0]) if defer else 0.0
                err0 = str(defer[1]) if defer else ""
                row = conn.execute(
                    """
                    SELECT state, queue, COALESCE(tail_seq,0) AS tail_seq,
                           COALESCE(available_at,0) AS available_at,
                           COALESCE(error_msg,'') AS error_msg
                    FROM index_v3_jobs
                    WHERE file_id=? AND artifact=?
                    """,
                    (int(job.file_id), job.artifact.value),
                ).fetchone()
                state = str(row["state"]) if row else ""
                if state in ("pending", "claimed"):
                    # Yanlış kuyrukta bekleyen işi retarget et (REPAIR↔HEAVY).
                    # claimed dokunma — worker elinde.
                    owl_rank = int(getattr(job, "owl_priority", 0) or 0)
                    queue_changed = str(row["queue"] or "") != job.queue.value
                    rank_changed = (
                        job.artifact.value == "owlv2"
                        and owl_rank > 0
                        and int(row["tail_seq"] or 0) != owl_rank
                    )
                    need_jumbo = (
                        state == "pending"
                        and defer is not None
                        and (
                            float(row["available_at"] or 0) < JUMBO_AVAILABLE_AT_FAR
                            or str(row["error_msg"] or "") != JUMBO_DEFER_MARKER
                        )
                    )
                    if state == "pending" and (
                        queue_changed or rank_changed or need_jumbo
                    ):
                        conn.execute(
                            """
                            UPDATE index_v3_jobs SET
                                queue=?, source_id=?, path=?,
                                tail_seq=CASE
                                    WHEN ?>0 AND artifact='owlv2' THEN ?
                                    ELSE tail_seq END,
                                available_at=CASE WHEN ? THEN ? ELSE available_at END,
                                error_msg=CASE WHEN ? THEN ? ELSE error_msg END,
                                updated_at=datetime('now')
                            WHERE file_id=? AND artifact=? AND state='pending'
                            """,
                            (
                                job.queue.value,
                                int(job.source_id),
                                str(job.path or ""),
                                int(owl_rank),
                                int(owl_rank),
                                1 if need_jumbo else 0,
                                float(avail),
                                1 if need_jumbo else 0,
                                err0 or JUMBO_DEFER_MARKER,
                                int(job.file_id),
                                job.artifact.value,
                            ),
                        )
                        if queue_changed or need_jumbo:
                            n += 1
                    continue
                if state == "done":
                    # Fiziksel artifact sonradan silinmişse planner bu işi
                    # yeniden eksik olarak üretir. Bu durumda eski DONE kaydı
                    # artık gerçeği temsil etmez ve güvenle tekrar açılabilir.
                    if reopen_done:
                        conn.execute(
                            """
                            UPDATE index_v3_jobs SET
                                queue=?, source_id=?, path=?, state='pending',
                                worker='', error_msg=?,
                                available_at=?, claimed_at=0, heartbeat_at=0,
                                updated_at=datetime('now')
                            WHERE file_id=? AND artifact=? AND state='done'
                            """,
                            (
                                job.queue.value,
                                int(job.source_id),
                                str(job.path or ""),
                                err0 or "physical_artifact_missing",
                                float(avail),
                                int(job.file_id),
                                job.artifact.value,
                            ),
                        )
                        n += 1
                    continue
                if state == "failed_permanent":
                    if not reopen_permanent:
                        continue
                    conn.execute(
                        """
                        UPDATE index_v3_jobs SET
                            queue=?, source_id=?, path=?, state='pending',
                            worker='', error_msg=?, attempts=0, tail_seq=0,
                            available_at=?, claimed_at=0, heartbeat_at=0,
                            updated_at=datetime('now')
                        WHERE file_id=? AND artifact=?
                        """,
                        (
                            job.queue.value,
                            int(job.source_id),
                            str(job.path or ""),
                            err0,
                            float(avail),
                            int(job.file_id),
                            job.artifact.value,
                        ),
                    )
                    n += 1
                    continue
                if row:
                    conn.execute(
                        """
                        UPDATE index_v3_jobs SET
                            queue=?, source_id=?, path=?, state='pending',
                            worker='', error_msg=?,
                            available_at=?,
                            updated_at=datetime('now')
                        WHERE file_id=? AND artifact=?
                        """,
                        (
                            job.queue.value,
                            int(job.source_id),
                            str(job.path or ""),
                            err0,
                            float(avail),
                            int(job.file_id),
                            job.artifact.value,
                        ),
                    )
                else:
                    owl_rank = int(getattr(job, "owl_priority", 0) or 0)
                    conn.execute(
                        """
                        INSERT INTO index_v3_jobs(
                            file_id, artifact, queue, source_id, path,
                            state, updated_at, enqueued_at, tail_seq,
                            available_at, error_msg
                        ) VALUES (?,?,?,?,?,'pending', datetime('now'),
                                  strftime('%s','now'), ?, ?, ?)
                        """,
                        (
                            int(job.file_id),
                            job.artifact.value,
                            job.queue.value,
                            int(job.source_id),
                            str(job.path or ""),
                            int(owl_rank) if job.artifact.value == "owlv2" else 0,
                            float(avail),
                            err0,
                        ),
                    )
                n += 1
            conn.commit()
        return n

    def count_claimable_fast_pending(
        self,
        *,
        source_ids: list[int] | None = None,
    ) -> int:
        """Phase-1: pending LIGHT/PREVIEW with available_at <= now (normals)."""
        self._ensure()
        sql = """
            SELECT COUNT(*) AS n FROM index_v3_jobs
            WHERE state='pending'
              AND queue IN ('light','preview')
              AND available_at <= strftime('%s','now')
        """
        params: list[Any] = []
        if source_ids:
            sph = ",".join("?" * len(source_ids))
            sql += f" AND source_id IN ({sph})"
            params.extend(int(x) for x in source_ids)
        with self._lock, self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return int(row["n"] if row else 0)

    def count_jumbo_deferred(
        self,
        *,
        source_ids: list[int] | None = None,
    ) -> int:
        """Pending light jobs parked for Phase-2 (marker and/or FAR available_at)."""
        self._ensure()
        sql = """
            SELECT COUNT(*) AS n FROM index_v3_jobs
            WHERE state='pending'
              AND queue IN ('light','preview')
              AND (
                error_msg=?
                OR available_at >= ?
              )
        """
        params: list[Any] = [JUMBO_DEFER_MARKER, float(JUMBO_AVAILABLE_AT_FAR)]
        if source_ids:
            sph = ",".join("?" * len(source_ids))
            sql += f" AND source_id IN ({sph})"
            params.extend(int(x) for x in source_ids)
        with self._lock, self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return int(row["n"] if row else 0)

    def release_jumbo_deferred(
        self,
        *,
        source_ids: list[int] | None = None,
    ) -> int:
        """Phase-1 complete → make jumbo light jobs claimable."""
        self._ensure()
        sql = """
            UPDATE index_v3_jobs SET
                available_at=0,
                error_msg=CASE
                    WHEN error_msg=? THEN ''
                    ELSE error_msg END,
                updated_at=datetime('now')
            WHERE state='pending'
              AND queue IN ('light','preview')
              AND (
                error_msg=?
                OR available_at >= ?
              )
        """
        params: list[Any] = [
            JUMBO_DEFER_MARKER,
            JUMBO_DEFER_MARKER,
            float(JUMBO_AVAILABLE_AT_FAR),
        ]
        if source_ids:
            sph = ",".join("?" * len(source_ids))
            sql += f" AND source_id IN ({sph})"
            params.extend(int(x) for x in source_ids)
        with self._lock, self._connect() as conn:
            cur = conn.execute(sql, params)
            conn.commit()
            return int(cur.rowcount or 0)

    def maybe_release_jumbo_phase(
        self,
        *,
        source_ids: list[int] | None = None,
    ) -> int:
        """If no claimable normal FAST work remains, release deferred jumbos."""
        if self.count_claimable_fast_pending(source_ids=source_ids) > 0:
            return 0
        if self.count_jumbo_deferred(source_ids=source_ids) <= 0:
            return 0
        return self.release_jumbo_deferred(source_ids=source_ids)

    def reopen_failed_permanent(
        self,
        *,
        source_ids: list[int] | None = None,
        artifacts: list[str] | None = None,
    ) -> int:
        """Renderer bağlandıktan sonra light fail'leri yeniden dene."""
        arts = list(artifacts) if artifacts is not None else []
        sql = (
            "UPDATE index_v3_jobs SET state='pending', worker='', "
            "error_msg='', attempts=0, updated_at=datetime('now') "
            "WHERE state='failed_permanent'"
        )
        params: list[Any] = []
        if arts:
            ph = ",".join("?" * len(arts))
            sql += f" AND artifact IN ({ph})"
            params.extend(arts)
        # Gerçek terminal durumlar otomatik yeniden açılmaz.
        # unsupported/renderer/empty/corrupt dosyalar sistem hatası değil,
        # bu formatın mevcut üretim yolu tarafından işlenemediği anlamına gelir.
        terminal = (
            "unsupported_visual",
            "renderer_missing",
            "empty_file",
            "corrupt_file",
        )
        sql += " AND " + " AND ".join(
            "IFNULL(error_msg,'') NOT LIKE ?" for _ in terminal
        )
        params.extend(f"%{t}%" for t in terminal)
        if source_ids:
            sph = ",".join("?" * len(source_ids))
            sql += f" AND source_id IN ({sph})"
            params.extend(int(x) for x in source_ids)
        with self._lock, self._connect() as conn:
            cur = conn.execute(sql, params)
            conn.commit()
            return int(cur.rowcount or 0)

    def claim(
        self,
        queue: QueueKind,
        worker: str,
        *,
        source_id: int | None = None,
        limit: int = 1,
        artifacts: tuple[Artifact, ...] | list[Artifact] | None = None,
        file_id: int | None = None,
        skip_dep_wait: bool = False,
    ) -> list[Job]:
        # Production jobs DB may predate timings/last_claimed_at.
        # Migrate on claim so a restarted worker binds instrumentation.
        self._ensure()
        with self._lock, self._connect() as conn:
            sql = """
                SELECT id, file_id, artifact, queue, source_id, path
                FROM index_v3_jobs
                WHERE queue=? AND state='pending'
              AND available_at <= strftime('%s','now')
            """
            params: list[Any] = [queue.value]
            if source_id is not None and int(source_id) > 0:
                sql += " AND source_id=?"
                params.append(int(source_id))
            arts = [a.value if isinstance(a, Artifact) else str(a) for a in (artifacts or ())]
            if arts:
                ph = ",".join("?" * len(arts))
                sql += f" AND artifact IN ({ph})"
                params.extend(arts)
            if file_id is not None and int(file_id) > 0:
                sql += " AND file_id=?"
                params.append(int(file_id))
            # Küçük source'taki sonsuz dep_wait (PATCH/OCR) büyük source'taki
            # HASH…DNA işlerini aç bırakmasın — fair_claim iki geçiş kullanır.
            if skip_dep_wait:
                sql += " AND COALESCE(error_msg,'')!='dep_wait'"
            # Artifact sırası: Hash/DINO/CLIP/Texture önce.
            # OWLv2, PATCH/OCR dep_wait döngüsünün arkasında aç kalmasın.
            # Patch hâlâ HASH…DNA ve OWL'den sonra (yavaş CPU işi).
            sql += (
                " ORDER BY CASE WHEN error_msg='dep_wait' THEN 1 ELSE 0 END,"
                " CASE artifact"
                " WHEN 'hash' THEN 0 WHEN 'metadata' THEN 1"
                " WHEN 'dino' THEN 2 WHEN 'clip' THEN 3"
                " WHEN 'texture' THEN 4"
                " WHEN 'semantic' THEN 5 WHEN 'dna' THEN 6"
                " WHEN 'object_concept' THEN 7"
                " WHEN 'owlv2' THEN 8"
                " WHEN 'patch' THEN 9 WHEN 'ocr' THEN 10 ELSE 11 END,"
                " CASE WHEN COALESCE(error_msg,'')='' THEN 0 ELSE 1 END,"
                " CASE WHEN artifact='owlv2' THEN"
                " CASE WHEN COALESCE(tail_seq,0)>0 THEN tail_seq ELSE 4 END"
                " WHEN COALESCE(tail_seq,0)>0 THEN tail_seq ELSE id END,"
                " id"
                " LIMIT ?"
            )
            params.append(int(limit))
            rows = conn.execute(sql, params).fetchall()
            out: list[Job] = []
            for row in rows:
                cur = conn.execute(
                    """
                    UPDATE index_v3_jobs SET
                        state='claimed', worker=?, attempts=attempts+1,
                        claimed_at=strftime('%s','now'),
                        last_claimed_at=strftime('%s','now'),
                        heartbeat_at=strftime('%s','now'),
                        updated_at=datetime('now')
                    WHERE id=? AND state='pending'
                    """,
                    (worker, int(row["id"])),
                )
                if cur.rowcount:
                    out.append(
                        Job(
                            file_id=int(row["file_id"]),
                            artifact=Artifact(row["artifact"]),
                            queue=QueueKind(row["queue"]),
                            source_id=int(row["source_id"] or 0),
                            path=str(row["path"] or ""),
                        )
                    )
            conn.commit()
            return out

    def release_dep_wait(
        self, file_id: int, artifact: Artifact, reason: str = "dep_wait"
    ) -> None:
        """Return claimed job to pending without consuming a failure attempt."""
        msg = str(reason or "dep_wait").strip()[:80] or "dep_wait"
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE index_v3_jobs SET
                    state='pending', worker='', error_msg=?,
                    attempts=CASE WHEN attempts>0 THEN attempts-1 ELSE 0 END,
                    available_at=(strftime('%s','now') + 1),
                    claimed_at=0, heartbeat_at=0,
                    updated_at=datetime('now')
                WHERE file_id=? AND artifact=? AND state='claimed'
                """,
                (msg, int(file_id), artifact.value),
            )
            conn.commit()

    def job_state(self, file_id: int, artifact: Artifact) -> str:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT state FROM index_v3_jobs WHERE file_id=? AND artifact=?",
                (int(file_id), artifact.value),
            ).fetchone()
            return str(row["state"] or "") if row else ""

    def complete(self, file_id: int, artifact: Artifact) -> None:
        self._ensure()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE index_v3_jobs SET state='done', worker='',
                    error_msg='', heartbeat_at=0,
                    last_claimed_at=CASE
                        WHEN COALESCE(last_claimed_at,0)>0 THEN last_claimed_at
                        ELSE claimed_at END,
                    updated_at=datetime('now')
                WHERE file_id=? AND artifact=?
                """,
                (int(file_id), artifact.value),
            )
            conn.commit()

    def list_pending_artifacts(self, file_id: int) -> list[Artifact]:
        """Pending artifact enums for one file (claim/gap reconcile)."""
        self._ensure()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT artifact FROM index_v3_jobs
                WHERE file_id=? AND state='pending'
                """,
                (int(file_id),),
            ).fetchall()
        out: list[Artifact] = []
        for row in rows:
            try:
                out.append(Artifact(str(row["artifact"])))
            except ValueError:
                continue
        return out

    def cancel_pending_artifact(self, file_id: int, artifact: Artifact) -> int:
        """Drop pending/claimed row (stale/unrunnable). Never touches done/failed."""
        self._ensure()
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                DELETE FROM index_v3_jobs
                WHERE file_id=? AND artifact=?
                  AND state IN ('pending','claimed')
                """,
                (int(file_id), artifact.value),
            )
            conn.commit()
            return int(cur.rowcount or 0)

    def claim_clock(self, file_id: int, artifact: Artifact) -> tuple[float, float]:
        """claimed_at, enqueued_at unix seconds. Missing → (0, 0)."""
        self._ensure()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT claimed_at, enqueued_at, last_claimed_at
                FROM index_v3_jobs WHERE file_id=? AND artifact=?
                """,
                (int(file_id), artifact.value),
            ).fetchone()
        if row is None:
            return 0.0, 0.0
        claimed = float(row["claimed_at"] or 0) or float(row["last_claimed_at"] or 0)
        return claimed, float(row["enqueued_at"] or 0)

    def fail(
        self,
        file_id: int,
        artifact: Artifact,
        *,
        error: str = "",
        permanent: bool = False,
        max_attempts: int = 3,
    ) -> None:
        state = JobState.FAILED_PERMANENT.value if permanent else JobState.FAILED.value
        self._ensure()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT attempts FROM index_v3_jobs WHERE file_id=? AND artifact=?",
                (int(file_id), artifact.value),
            ).fetchone()
            attempts = int(row["attempts"] if row else 0)
            tail_seq = 0
            # Soft retry stays behind never-failed pending (tail_seq). Small
            # available_at backoff stops a lone failing job from claim→fail spin.
            available_at_sql = "0"
            if not permanent and attempts < max_attempts:
                # Queue-tail retry: normal pending jobs finish first.
                state = JobState.PENDING.value
                tail_seq = int(
                    conn.execute(
                        """
                        SELECT COALESCE(
                            MAX(CASE WHEN COALESCE(tail_seq,0)>0 THEN tail_seq ELSE id END),
                            0
                        ) + 1
                        FROM index_v3_jobs
                        """
                    ).fetchone()[0]
                    or 1
                )
                delay = max(1, min(30, int(attempts)))
                available_at_sql = f"(strftime('%s','now') + {delay})"
            elif not permanent and attempts >= max_attempts:
                state = JobState.FAILED_PERMANENT.value
            conn.execute(
                f"""
                UPDATE index_v3_jobs SET state=?, worker='', error_msg=?,
                    last_claimed_at=CASE
                        WHEN COALESCE(last_claimed_at,0)>0 THEN last_claimed_at
                        ELSE claimed_at END,
                    claimed_at=0, heartbeat_at=0, tail_seq=?,
                    available_at={available_at_sql},
                    updated_at=datetime('now')
                WHERE file_id=? AND artifact=?
                """,
                (
                    state,
                    str(error or "")[:500],
                    int(tail_seq),
                    int(file_id),
                    artifact.value,
                ),
            )
            conn.commit()

    def heartbeat(self, file_id: int, artifact: Artifact, worker: str) -> bool:
        """Refresh a live claim lease without changing its queue state."""
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE index_v3_jobs SET heartbeat_at=strftime('%s','now'),
                    updated_at=datetime('now')
                WHERE file_id=? AND artifact=? AND state='claimed' AND worker=?
                """,
                (int(file_id), artifact.value, str(worker)),
            )
            conn.commit()
            return bool(cur.rowcount)

    def requeue_stale_claims(
        self,
        timeout_sec: float = 180.0,
        *,
        worker: str | None = None,
    ) -> int:
        """Return expired claimed jobs to pending, preserving retry attempts."""
        timeout = max(30.0, float(timeout_sec))
        cutoff = __import__("time").time() - timeout
        with self._lock, self._connect() as conn:
            sql = """
                UPDATE index_v3_jobs SET state='pending', worker='',
                    error_msg='stale_claim', available_at=0,
                    claimed_at=0, heartbeat_at=0, updated_at=datetime('now')
                                WHERE state='claimed'
                                    AND COALESCE(
                                                NULLIF(heartbeat_at, 0),
                                                NULLIF(claimed_at, 0),
                                                strftime('%s', updated_at)
                                            ) < ?
            """
            params: list[Any] = [cutoff]
            if worker:
                sql += " AND worker=?"
                params.append(str(worker))
            cur = conn.execute(sql, params)
            conn.commit()
            return int(cur.rowcount or 0)

    def requeue_claims_for_worker(self, worker: str, *, error: str = "worker_dead") -> int:
        """Release all claims owned by a lane known to have terminated."""
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE index_v3_jobs SET state='pending', worker='',
                    error_msg=?, available_at=0, claimed_at=0, heartbeat_at=0,
                    updated_at=datetime('now')
                WHERE state='claimed' AND worker=?
                """,
                (str(error)[:500], str(worker)),
            )
            conn.commit()
            return int(cur.rowcount or 0)

    def count_claimed_fresh(
        self,
        timeout_sec: float = 180.0,
        *,
        source_ids: list[int] | None = None,
    ) -> int:
        """Count only claims whose lease is still fresh."""
        cutoff = __import__("time").time() - max(30.0, float(timeout_sec))
        with self._connect() as conn:
            sql = """
                SELECT COUNT(*) AS n FROM index_v3_jobs
                WHERE state='claimed'
                  AND COALESCE(
                                                NULLIF(heartbeat_at, 0),
                                                NULLIF(claimed_at, 0),
                                                strftime('%s', updated_at)
                                            ) >= ?
            """
            params: list[Any] = [cutoff]
            if source_ids is not None:
                ids = [int(s) for s in source_ids if int(s) > 0]
                if not ids:
                    return 0
                ph = ",".join("?" * len(ids))
                sql += f" AND source_id IN ({ph})"
                params.extend(ids)
            row = conn.execute(sql, params).fetchone()
            return int(row["n"] or 0)

    def reset_stale_claims(self) -> int:
        """Compatibility wrapper: reclaim all existing claims at startup."""
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE index_v3_jobs SET state='pending', worker='',
                    available_at=0, error_msg='stale_claim',
                    claimed_at=0, heartbeat_at=0,
                    updated_at=datetime('now')
                WHERE state='claimed'
                """
            )
            conn.commit()
            return int(cur.rowcount or 0)

    def count_pending(
        self,
        queue: QueueKind | None = None,
        *,
        source_ids: list[int] | None = None,
        artifacts: tuple[Artifact, ...] | list[Artifact] | None = None,
    ) -> int:
        with self._connect() as conn:
            sql = "SELECT COUNT(*) AS n FROM index_v3_jobs WHERE state='pending'"
            params: list[Any] = []
            if queue is not None:
                sql += " AND queue=?"
                params.append(queue.value)
            if source_ids is not None:
                ids = [int(s) for s in source_ids if int(s) > 0]
                if not ids:
                    return 0
                ph = ",".join("?" * len(ids))
                sql += f" AND source_id IN ({ph})"
                params.extend(ids)
            arts = [a.value if isinstance(a, Artifact) else str(a) for a in (artifacts or ())]
            if arts:
                ph = ",".join("?" * len(arts))
                sql += f" AND artifact IN ({ph})"
                params.extend(arts)
            row = conn.execute(sql, params).fetchone()
            return int(row["n"] or 0)

    def count_pending_files(
        self,
        queue: QueueKind | None = None,
        *,
        queues: list[QueueKind] | tuple[QueueKind, ...] | None = None,
        source_ids: list[int] | None = None,
        artifacts: tuple[Artifact, ...] | list[Artifact] | None = None,
    ) -> int:
        """Pending job sayısı değil: kaç benzersiz dosyanın en az bir pending satırı var."""
        with self._connect() as conn:
            sql = (
                "SELECT COUNT(DISTINCT file_id) AS n FROM index_v3_jobs "
                "WHERE state='pending'"
            )
            params: list[Any] = []
            qlist: list[str] = []
            if queues:
                qlist = [q.value for q in queues]
            elif queue is not None:
                qlist = [queue.value]
            if qlist:
                ph = ",".join("?" * len(qlist))
                sql += f" AND queue IN ({ph})"
                params.extend(qlist)
            if source_ids is not None:
                ids = [int(s) for s in source_ids if int(s) > 0]
                if not ids:
                    return 0
                ph = ",".join("?" * len(ids))
                sql += f" AND source_id IN ({ph})"
                params.extend(ids)
            arts = [a.value if isinstance(a, Artifact) else str(a) for a in (artifacts or ())]
            if arts:
                ph = ",".join("?" * len(arts))
                sql += f" AND artifact IN ({ph})"
                params.extend(arts)
            row = conn.execute(sql, params).fetchone()
            return int(row["n"] or 0)

    def list_pending_owl_dep_wait(
        self,
        queue: QueueKind,
        *,
        source_ids: list[int] | None = None,
        limit: int = 10000,
    ) -> list[int]:
        """Pending OWLv2 rows still tagged dep_wait (claim sort key)."""
        cap = max(1, min(int(limit or 10000), 20000))
        with self._connect() as conn:
            sql = (
                "SELECT file_id FROM index_v3_jobs "
                "WHERE queue=? AND artifact='owlv2' AND state='pending' "
                "AND error_msg='dep_wait'"
            )
            params: list[Any] = [queue.value]
            if source_ids is not None:
                ids = [int(s) for s in source_ids if int(s) > 0]
                if not ids:
                    return []
                ph = ",".join("?" * len(ids))
                sql += f" AND source_id IN ({ph})"
                params.extend(ids)
            sql += " ORDER BY id LIMIT ?"
            params.append(cap)
            return [int(r["file_id"]) for r in conn.execute(sql, params).fetchall()]

    def clear_stale_owl_dep_wait(self, file_ids: list[int]) -> int:
        """Drop stale dep_wait on pending OWLv2. Does not claim or insert rows."""
        ids = [int(x) for x in file_ids if int(x) > 0]
        if not ids:
            return 0
        n = 0
        with self._lock, self._connect() as conn:
            for i in range(0, len(ids), 400):
                chunk = ids[i : i + 400]
                ph = ",".join("?" * len(chunk))
                cur = conn.execute(
                    f"""
                    UPDATE index_v3_jobs SET
                        error_msg='', available_at=0, updated_at=datetime('now')
                    WHERE artifact='owlv2' AND state='pending'
                      AND error_msg='dep_wait' AND file_id IN ({ph})
                    """,
                    chunk,
                )
                n += int(cur.rowcount or 0)
            conn.commit()
        return n

    def owlv2_queue_bands(self, *, source_ids: list[int] | None = None) -> dict[str, int]:
        """Pending+claimed OWLv2 ranks for UI color. No labels."""
        out = {"priority_pending": 0, "weak_pending": 0, "unranked_pending": 0}
        with self._connect() as conn:
            sql = (
                "SELECT COALESCE(tail_seq,0) AS t, COUNT(*) AS n "
                "FROM index_v3_jobs WHERE artifact='owlv2' "
                "AND state IN ('pending','claimed')"
            )
            params: list[Any] = []
            if source_ids is not None:
                ids = [int(s) for s in source_ids if int(s) > 0]
                if not ids:
                    return out
                ph = ",".join("?" * len(ids))
                sql += f" AND source_id IN ({ph})"
                params.extend(ids)
            sql += " GROUP BY COALESCE(tail_seq,0)"
            rows = conn.execute(sql, params).fetchall()
        for r in rows:
            t = int(r["t"] or 0)
            n = int(r["n"] or 0)
            if t in (1, 2, 3):
                out["priority_pending"] += n
            elif t == 4:
                out["weak_pending"] += n
            else:
                out["unranked_pending"] += n
        return out

    def count_pending_retry(
        self,
        queue: QueueKind | None = None,
        *,
        source_ids: list[int] | None = None,
    ) -> int:
        """Count distinct files that will be retried, not artifact jobs."""
        with self._connect() as conn:
            sql = (
                "SELECT COUNT(DISTINCT file_id) AS n FROM index_v3_jobs "
                "WHERE state='pending' "
                "AND COALESCE(error_msg,'')!='' "
                "AND error_msg!='dep_wait'"
            )
            params: list[Any] = []
            if queue is not None:
                sql += " AND queue=?"
                params.append(queue.value)
            if source_ids is not None:
                ids = [int(s) for s in source_ids if int(s) > 0]
                if not ids:
                    return 0
                ph = ",".join("?" * len(ids))
                sql += f" AND source_id IN ({ph})"
                params.extend(ids)
            row = conn.execute(sql, params).fetchone()
            return int(row["n"] or 0)

    def list_pending_retry_files(
        self,
        *,
        source_ids: list[int] | None = None,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        """Retry bekleyen farklı dosyaları, operatör ekranı için listeler."""
        with self._connect() as conn:
            sql = (
                "SELECT file_id, MAX(path) AS path, "
                "MAX(error_msg) AS error_msg "
                "FROM index_v3_jobs WHERE state='pending' "
                "AND COALESCE(error_msg,'')!='' AND error_msg!='dep_wait'"
            )
            params: list[Any] = []
            if source_ids is not None:
                ids = [int(x) for x in source_ids if int(x) > 0]
                if not ids:
                    return []
                ph = ",".join("?" * len(ids))
                sql += f" AND source_id IN ({ph})"
                params.extend(ids)
            sql += " GROUP BY file_id ORDER BY MIN(id) LIMIT ?"
            params.append(max(1, int(limit)))
            rows = conn.execute(sql, params).fetchall()
        return [
            {
                "file_id": int(r["file_id"] or 0),
                "path": str(r["path"] or ""),
                "error": str(r["error_msg"] or ""),
            }
            for r in rows
        ]

    def list_failed_permanent_files(
        self,
        *,
        source_ids: list[int] | None = None,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        """Kalıcı hatalı farklı dosyaları, operatör ekranı için listeler."""
        with self._connect() as conn:
            sql = (
                "SELECT file_id, MAX(path) AS path, "
                "MAX(error_msg) AS error_msg "
                "FROM index_v3_jobs WHERE state='failed_permanent' "
                "AND artifact NOT IN ('ocr','patch')"
            )
            params: list[Any] = []
            if source_ids is not None:
                ids = [int(x) for x in source_ids if int(x) > 0]
                if not ids:
                    return []
                ph = ",".join("?" * len(ids))
                sql += f" AND source_id IN ({ph})"
                params.extend(ids)
            sql += " GROUP BY file_id ORDER BY MIN(id) LIMIT ?"
            params.append(max(1, int(limit)))
            rows = conn.execute(sql, params).fetchall()
        return [
            {
                "file_id": int(r["file_id"] or 0),
                "path": str(r["path"] or ""),
                "error": str(r["error_msg"] or ""),
            }
            for r in rows
        ]

    def count_failed_permanent_files(
        self,
        *,
        source_ids: list[int] | None = None,
    ) -> int:
        """Kalıcı hatası bulunan farklı dosya sayısı (job değil dosya).

        UI'deki "Kalan dosya" hesabında aynı dosyanın birden fazla
        kalıcı hata işinin iki/üç kez sayılmasını engeller.
        OCR/PATCH fail Genel AI terminal sayılmaz.
        """
        with self._connect() as conn:
            sql = (
                "SELECT COUNT(DISTINCT file_id) AS n "
                "FROM index_v3_jobs WHERE state='failed_permanent' "
                "AND artifact NOT IN ('ocr','patch')"
            )
            params: list[Any] = []
            if source_ids is not None:
                ids = [int(s) for s in source_ids if int(s) > 0]
                if not ids:
                    return 0
                ph = ",".join("?" * len(ids))
                sql += f" AND source_id IN ({ph})"
                params.extend(ids)
            row = conn.execute(sql, params).fetchone()
            return int(row["n"] or 0)

    def count_failed_permanent(
        self,
        queue: QueueKind | None = None,
        *,
        source_ids: list[int] | None = None,
    ) -> int:
        with self._connect() as conn:
            sql = "SELECT COUNT(*) AS n FROM index_v3_jobs WHERE state='failed_permanent'"
            params: list[Any] = []
            if queue is not None:
                sql += " AND queue=?"
                params.append(queue.value)
            if source_ids is not None:
                ids = [int(s) for s in source_ids if int(s) > 0]
                if not ids:
                    return 0
                ph = ",".join("?" * len(ids))
                sql += f" AND source_id IN ({ph})"
                params.extend(ids)
            row = conn.execute(sql, params).fetchone()
            return int(row["n"] or 0)

    def count_pending_grouped(
        self,
        source_ids: list[int],
    ) -> list[tuple[int, str, int]]:
        """(source_id, queue, count) — pending_jobs() ile 64k satır çekme."""
        ids = [int(s) for s in source_ids if int(s) > 0]
        if not ids:
            return []
        ph = ",".join("?" * len(ids))
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT source_id, queue, COUNT(*) AS n
                FROM index_v3_jobs
                WHERE state='pending' AND source_id IN ({ph})
                GROUP BY source_id, queue
                """,
                ids,
            ).fetchall()
        return [
            (int(r["source_id"] or 0), str(r["queue"] or ""), int(r["n"] or 0))
            for r in rows
        ]

    def count_claimed(
        self,
        queue: QueueKind | None = None,
        *,
        source_ids: list[int] | None = None,
        queues: list[QueueKind] | tuple[QueueKind, ...] | None = None,
    ) -> int:
        """Aktif claim sayısı — UI İşleniyor için fresh şartı yok."""
        with self._connect() as conn:
            sql = "SELECT COUNT(*) AS n FROM index_v3_jobs WHERE state='claimed'"
            params: list[Any] = []
            qlist = list(queues or ())
            if queue is not None:
                qlist = [queue]
            if qlist:
                ph = ",".join("?" * len(qlist))
                sql += f" AND queue IN ({ph})"
                params.extend(q.value for q in qlist)
            if source_ids is not None:
                ids = [int(s) for s in source_ids if int(s) > 0]
                if not ids:
                    return 0
                ph = ",".join("?" * len(ids))
                sql += f" AND source_id IN ({ph})"
                params.extend(ids)
            row = conn.execute(sql, params).fetchone()
            return int(row["n"] or 0)

    def count_completed_recent(
        self,
        window_sec: float = 60.0,
        *,
        source_ids: list[int] | None = None,
        queues: list[QueueKind] | tuple[QueueKind, ...] | None = None,
    ) -> int:
        """Son N saniyede done olan job sayısı — UI throughput (coverage değil)."""
        win = max(5, int(float(window_sec) or 60))
        with self._connect() as conn:
            sql = """
                SELECT COUNT(*) AS n FROM index_v3_jobs
                WHERE state='done'
                  AND updated_at != ''
                  AND updated_at >= datetime('now', ?)
            """
            params: list[Any] = [f"-{win} seconds"]
            if queues:
                qlist = list(queues)
                ph = ",".join("?" * len(qlist))
                sql += f" AND queue IN ({ph})"
                params.extend(q.value for q in qlist)
            if source_ids is not None:
                ids = [int(s) for s in source_ids if int(s) > 0]
                if not ids:
                    return 0
                ph = ",".join("?" * len(ids))
                sql += f" AND source_id IN ({ph})"
                params.extend(ids)
            row = conn.execute(sql, params).fetchone()
            return int(row["n"] or 0)

    def pending_jobs(self, queue: QueueKind | None = None) -> list[Job]:
        with self._connect() as conn:
            if queue is None:
                rows = conn.execute(
                    """
                    SELECT file_id, artifact, queue, source_id, path
                    FROM index_v3_jobs WHERE state='pending' ORDER BY id
                    """
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT file_id, artifact, queue, source_id, path
                    FROM index_v3_jobs
                    WHERE state='pending' AND queue=? ORDER BY id
                    """,
                    (queue.value,),
                ).fetchall()
        return [
            Job(
                file_id=int(r["file_id"]),
                artifact=Artifact(r["artifact"]),
                queue=QueueKind(r["queue"]),
                source_id=int(r["source_id"] or 0),
                path=str(r["path"] or ""),
            )
            for r in rows
        ]

    def clear_done(self) -> int:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM index_v3_jobs WHERE state='done'")
            conn.commit()
            return int(cur.rowcount or 0)

    def cancel_jobs_for_file_ids(self, file_ids: list[int]) -> int:
        """pending/claimed işleri düşür (missing/orphan reconcile)."""
        ids = [int(x) for x in file_ids if int(x) > 0]
        if not ids:
            return 0
        with self._lock, self._connect() as conn:
            n = 0
            chunk = 400
            for i in range(0, len(ids), chunk):
                part = ids[i : i + chunk]
                ph = ",".join("?" * len(part))
                cur = conn.execute(
                    f"""
                    DELETE FROM index_v3_jobs
                    WHERE file_id IN ({ph})
                      AND state IN ('pending','claimed')
                    """,
                    part,
                )
                n += int(cur.rowcount or 0)
            conn.commit()
            return n

    def purge_unknown_file_ids(self, known_file_ids: list[int] | set[int]) -> int:
        """Başka DB döneminden kalan job'ları sil (file_id eşleşmez)."""
        known = {int(x) for x in known_file_ids if int(x) > 0}
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT file_id FROM index_v3_jobs"
            ).fetchall()
            drop = [int(r[0]) for r in rows if int(r[0]) not in known]
            n = 0
            chunk = 400
            for i in range(0, len(drop), chunk):
                part = drop[i : i + chunk]
                ph = ",".join("?" * len(part))
                cur = conn.execute(
                    f"DELETE FROM index_v3_jobs WHERE file_id IN ({ph})",
                    part,
                )
                n += int(cur.rowcount or 0)
            conn.commit()
            return n

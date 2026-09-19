"""Heavy pipeline timing: persist stage clocks without rewriting old jobs."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from core.index_v3.queues import JobStore
from core.index_v3.timings import STAGE_COLS
from core.index_v3.types import Artifact, Job, QueueKind
from core.index_v3.worker import ArtifactProcessor, Worker
from core.index_v3 import worker as worker_module


class _Db:
    def get_file_by_id(self, file_id):
        return {"id": file_id, "source_id": 1, "path": f"/x/{file_id}.jpg"}

    def get_features(self, file_id):
        return {}


class _Report:
    preview_ready = True
    ai_final = False

    def __init__(self) -> None:
        self._ready: set[str] = set()

    def ready(self, artifact):
        return artifact.value in self._ready


def test_timing_row_format_and_claimed_at_survives(tmp_path: Path, monkeypatch):
    store = JobStore(tmp_path / "jobs.db")
    fid = 42
    jobs = [
        Job(fid, Artifact.HASH, QueueKind.HEAVY, 1, "/x/42.jpg"),
        Job(fid, Artifact.METADATA, QueueKind.HEAVY, 1, "/x/42.jpg"),
        Job(fid, Artifact.DINO, QueueKind.HEAVY, 1, "/x/42.jpg"),
        Job(fid, Artifact.CLIP, QueueKind.HEAVY, 1, "/x/42.jpg"),
        Job(fid, Artifact.TEXTURE, QueueKind.HEAVY, 1, "/x/42.jpg"),
        Job(fid, Artifact.SEMANTIC, QueueKind.HEAVY, 1, "/x/42.jpg"),
        Job(fid, Artifact.DNA, QueueKind.HEAVY, 1, "/x/42.jpg"),
        Job(fid, Artifact.PATCH, QueueKind.HEAVY, 1, "/x/42.jpg"),
        Job(fid, Artifact.OCR, QueueKind.HEAVY, 1, "/x/42.jpg"),
    ]
    store.enqueue(jobs)
    report = _Report()

    def assess(*_a, **_k):
        return report

    def process(_db, job, stats):
        time.sleep(0.02)
        stats.file_read_ms = float(getattr(stats, "file_read_ms", 0) or 0) + 1.5
        report._ready.add(job.artifact.value)
        if job.artifact == Artifact.DNA:
            report.ai_final = True
        return True

    monkeypatch.setattr(worker_module, "assess_file", assess)
    worker = Worker(
        _Db(), store, processor=ArtifactProcessor(process_fn=process)
    )
    stats = worker.run_queue(QueueKind.HEAVY, [1])
    assert stats.completed == 9

    row = store.timings.get(fid)
    assert row is not None
    for stage in STAGE_COLS:
        assert f"{stage}_started_at" in row
        assert f"{stage}_ended_at" in row
        assert f"{stage}_ms" in row
    for extra in (
        "queue_wait_ms",
        "worker_wait_ms",
        "file_read_ms",
        "total_elapsed_ms",
        "last_claimed_at",
    ):
        assert extra in row
        assert row[extra] is not None

    assert row["hash_started_at"] > 0
    assert row["hash_ended_at"] >= row["hash_started_at"]
    assert row["hash_ms"] >= 15
    assert row["metadata_ms"] >= 15
    assert row["dino_ms"] >= 15
    assert row["dna_ended_at"] is not None
    assert row["ai_final_started_at"] == row["hash_started_at"]
    assert row["ai_final_ended_at"] == row["dna_ended_at"]
    assert row["ai_final_ms"] >= row["hash_ms"]
    assert row["patch_ms"] >= 15
    assert row["ocr_ms"] >= 15
    assert row["file_read_ms"] >= 1.5 * 9 - 0.01
    assert row["total_elapsed_ms"] >= row["ai_final_ms"]
    assert row["last_claimed_at"] > 0

    with sqlite3.connect(store.db_path) as conn:
        claimed = conn.execute(
            "SELECT artifact, claimed_at, last_claimed_at, state "
            "FROM index_v3_jobs WHERE file_id=? AND artifact='dna'",
            (fid,),
        ).fetchone()
    assert claimed[3] == "done"
    assert float(claimed[1] or 0) > 0
    assert float(claimed[2] or 0) > 0


def test_timing_does_not_rewrite_historical_jobs(tmp_path: Path):
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE index_v3_jobs (
            id INTEGER PRIMARY KEY,
            file_id INTEGER NOT NULL,
            artifact TEXT NOT NULL,
            queue TEXT NOT NULL DEFAULT 'heavy',
            source_id INTEGER NOT NULL DEFAULT 0,
            path TEXT NOT NULL DEFAULT '',
            state TEXT NOT NULL DEFAULT 'done',
            attempts INTEGER NOT NULL DEFAULT 1,
            worker TEXT NOT NULL DEFAULT '',
            error_msg TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL DEFAULT '2026-08-01 00:00:00',
            available_at REAL NOT NULL DEFAULT 0,
            claimed_at REAL NOT NULL DEFAULT 0,
            heartbeat_at REAL NOT NULL DEFAULT 0
        );
        INSERT INTO index_v3_jobs(file_id, artifact, claimed_at, state)
        VALUES (7, 'hash', 0, 'done');
        """
    )
    conn.commit()
    conn.close()
    store = JobStore(db)
    with sqlite3.connect(db) as c:
        row = c.execute(
            "SELECT claimed_at, state FROM index_v3_jobs WHERE file_id=7"
        ).fetchone()
        n_timing = c.execute("SELECT COUNT(*) FROM index_v3_file_timings").fetchone()[0]
    assert row[0] == 0
    assert row[1] == "done"
    assert n_timing == 0
    assert store.timings.get(7) is None


def test_legacy_production_jobs_schema_records_timings(tmp_path: Path, monkeypatch):
    """patterns.v3jobs.db production shape: no timings, no last_claimed_at."""
    db = tmp_path / "prod_shaped.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE index_v3_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER NOT NULL,
            artifact TEXT NOT NULL,
            queue TEXT NOT NULL DEFAULT 'heavy',
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
        """
    )
    conn.commit()
    conn.close()

    store = JobStore(db)
    fid = 99
    jobs = [
        Job(fid, Artifact.HASH, QueueKind.HEAVY, 1, "/x/99.jpg"),
        Job(fid, Artifact.METADATA, QueueKind.HEAVY, 1, "/x/99.jpg"),
        Job(fid, Artifact.DINO, QueueKind.HEAVY, 1, "/x/99.jpg"),
        Job(fid, Artifact.CLIP, QueueKind.HEAVY, 1, "/x/99.jpg"),
        Job(fid, Artifact.TEXTURE, QueueKind.HEAVY, 1, "/x/99.jpg"),
        Job(fid, Artifact.SEMANTIC, QueueKind.HEAVY, 1, "/x/99.jpg"),
        Job(fid, Artifact.DNA, QueueKind.HEAVY, 1, "/x/99.jpg"),
        Job(fid, Artifact.PATCH, QueueKind.HEAVY, 1, "/x/99.jpg"),
        Job(fid, Artifact.OCR, QueueKind.HEAVY, 1, "/x/99.jpg"),
    ]
    store.enqueue(jobs)
    report = _Report()

    def assess(*_a, **_k):
        return report

    def process(_db, job, stats):
        stats.file_read_ms = float(getattr(stats, "file_read_ms", 0) or 0) + 1.0
        report._ready.add(job.artifact.value)
        if job.artifact == Artifact.DNA:
            report.ai_final = True
        return True

    monkeypatch.setattr(worker_module, "assess_file", assess)
    worker = Worker(
        _Db(), store, processor=ArtifactProcessor(process_fn=process)
    )
    stats = worker.run_queue(QueueKind.HEAVY, [1])
    assert stats.completed == 9
    row = store.timings.get(fid)
    assert row is not None
    assert row["hash_ms"] is not None
    assert row["dino_ms"] is not None
    assert row["clip_ms"] is not None
    assert row["dna_ms"] is not None
    assert row["ai_final_ms"] is not None
    assert row["patch_ms"] is not None
    assert row["ocr_ms"] is not None
    assert row["total_elapsed_ms"] is not None
    with sqlite3.connect(db) as c:
        n = c.execute("SELECT COUNT(*) FROM index_v3_file_timings").fetchone()[0]
        cols = {r[1] for r in c.execute("PRAGMA table_info(index_v3_jobs)")}
    assert n == 1
    assert "last_claimed_at" in cols
    assert "enqueued_at" in cols


"""reopen_done only for Preview/Thumbnail physical gaps — not heavy AI done jobs."""

from __future__ import annotations

from pathlib import Path

from core.db import Database
from core.index_v3.discovery import enqueue_existing_gaps
from core.index_v3.queues import JobStore
from core.index_v3.types import Artifact, Job, Mode, QueueKind


def _seed_file(db: Database, root: Path, *, preview_ready: int = 0) -> int:
    root.mkdir(parents=True, exist_ok=True)
    p = root / "a.jpg"
    p.write_bytes(b"img")
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("s", str(root)),
        )
        conn.execute(
            """
            INSERT INTO files(
              id, source_id, path, filename, status, width, height,
              feature_preview_path, physical_preview_ready, format_metadata
            ) VALUES (1,1,?,?, 'indexed', 32,32, ?, ?, '{}')
            """,
            (str(p), "a.jpg", str(root / "missing_p.webp"), int(preview_ready)),
        )
        conn.execute(
            """
            INSERT INTO features(file_id, dino_embedding, clip_embedding, phash)
            VALUES (1, X'01020304', X'05060708', 'abc')
            """
        )
    return 1


def test_gap_scan_does_not_reopen_done_dino_when_blob_present(tmp_path: Path):
    db = Database(tmp_path / "p.db")
    _seed_file(db, tmp_path / "src", preview_ready=1)
    # Physical preview ready so GA would plan missing heavy — but DINO blob exists.
    with db.connect() as conn:
        conn.execute(
            "UPDATE files SET physical_preview_ready=1, feature_preview_path=? WHERE id=1",
            (str(tmp_path / "src" / "a.jpg"),),
        )
    store = JobStore(tmp_path / "jobs.db")
    store.enqueue([Job(1, Artifact.DINO, QueueKind.HEAVY, 1, "a.jpg")])
    store.complete(1, Artifact.DINO)
    assert store.job_state(1, Artifact.DINO) == "done"

    # Force assess to still see DINO? Blob present → assess READY → not in plan.
    # Simulate reopen noise path: mark feature missing incorrectly would reopen.
    # Instead: enqueue a done CLIP then clear blob so plan wants CLIP — reopen_done
    # must NOT reopen heavy done.
    store.enqueue([Job(1, Artifact.CLIP, QueueKind.HEAVY, 1, "a.jpg")])
    store.complete(1, Artifact.CLIP)
    with db.connect() as conn:
        conn.execute("UPDATE features SET clip_embedding=NULL WHERE file_id=1")

    enqueue_existing_gaps(db, store, source_id=1, mode=Mode.GENERAL_AI)
    assert store.job_state(1, Artifact.CLIP) == "done"
    assert store.job_state(1, Artifact.DINO) == "done"


def test_gap_scan_reopens_done_preview_when_physical_missing(tmp_path: Path):
    db = Database(tmp_path / "p.db")
    _seed_file(db, tmp_path / "src", preview_ready=0)
    store = JobStore(tmp_path / "jobs.db")
    store.enqueue([Job(1, Artifact.PREVIEW, QueueKind.PREVIEW, 1, "a.jpg")])
    store.complete(1, Artifact.PREVIEW)
    assert store.job_state(1, Artifact.PREVIEW) == "done"

    enqueue_existing_gaps(db, store, source_id=1, mode=Mode.COMPLETE)
    assert store.job_state(1, Artifact.PREVIEW) == "pending"

from __future__ import annotations

from pathlib import Path

from core.db import Database
from core.index_queue_manager import IndexQueueManager
from core.index_v3.discovery import discover_source
from core.index_v3.planner import plan_jobs_for_file
from core.index_v3.queues import JobStore
from core.index_v3.types import Artifact, ArtifactStatus, FileArtifactReport, Mode, QueueKind
from core.settings import AppSettings
from core.utils import file_id_from_path, normalize_path


def _report(**ready) -> FileArtifactReport:
    r = FileArtifactReport(file_id=1, source_id=1, path="x")
    for art in Artifact:
        r.status[art] = (
            ArtifactStatus.READY if ready.get(art.value) else ArtifactStatus.MISSING
        )
    return r


def test_complete_file_is_not_replanned():
    report = _report(
        preview=True,
        thumbnail=True,
        hash=True,
        metadata=True,
        dino=True,
        clip=True,
        texture=True,
        semantic=True,
        dna=True,
        object_concept=True,
        owlv2=True,
        patch=True,
        ocr=True,
    )
    assert plan_jobs_for_file(report, Mode.COMPLETE) == []
    assert plan_jobs_for_file(report, Mode.FAST) == []
    assert plan_jobs_for_file(report, Mode.GENERAL_AI) == []


def test_partial_file_queues_only_missing_artifacts():
    report = _report(preview=True, thumbnail=True, hash=True, dino=True, texture=True)
    jobs = plan_jobs_for_file(report, Mode.COMPLETE)
    arts = [j.artifact for j in jobs]
    assert Artifact.PREVIEW not in arts
    assert Artifact.THUMBNAIL not in arts
    assert Artifact.HASH not in arts
    assert Artifact.DINO not in arts
    assert Artifact.CLIP in arts
    assert Artifact.OBJECT_CONCEPT in arts
    assert Artifact.OWLV2 in arts
    assert Artifact.SEMANTIC in arts
    assert Artifact.DNA in arts
    assert all(j.queue == QueueKind.HEAVY for j in jobs)


def test_general_ai_waits_without_preview():
    jobs = plan_jobs_for_file(_report(thumbnail=True), Mode.GENERAL_AI)
    assert [j.artifact for j in jobs] == [Artifact.PREVIEW]


def test_discover_does_not_reopen_done_jobs_on_complete_rescan(tmp_path: Path):
    db = Database(tmp_path / "patterns.db")
    settings = AppSettings()
    settings.db_path = str(tmp_path / "patterns.db")
    settings.cache_dir = str(tmp_path / "cache")
    settings.ensure_dirs()
    root = tmp_path / "src"
    root.mkdir()
    path = root / "keep.jpg"
    path.write_bytes(b"keep")
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("src", str(root)),
        )
        source_id = int(conn.execute("SELECT id FROM sources").fetchone()[0])
    store = JobStore(tmp_path / "jobs.db")
    first = discover_source(
        db, store, source_id=source_id, root_path=str(root), mode=Mode.COMPLETE, settings=settings
    )
    assert first.inserted == 1
    assert first.jobs_enqueued > 0
    with store._connect() as conn:
        conn.execute("UPDATE index_v3_jobs SET state='done'")
        conn.commit()
        pending_before = int(
            conn.execute(
                "SELECT COUNT(*) FROM index_v3_jobs WHERE state='pending'"
            ).fetchone()[0]
        )
    second = discover_source(
        db, store, source_id=source_id, root_path=str(root), mode=Mode.COMPLETE, settings=settings
    )
    assert second.inserted == 0
    assert second.changed == 0
    with store._connect() as conn:
        pending_after = int(
            conn.execute(
                "SELECT COUNT(*) FROM index_v3_jobs WHERE state='pending'"
            ).fetchone()[0]
        )
        done = int(
            conn.execute(
                "SELECT COUNT(*) FROM index_v3_jobs WHERE state='done'"
            ).fetchone()[0]
        )
    assert pending_before == 0
    assert pending_after == 0
    assert done > 0


def test_new_files_only_join_preview_queue(tmp_path: Path):
    db = Database(tmp_path / "patterns.db")
    settings = AppSettings()
    settings.db_path = str(tmp_path / "patterns.db")
    settings.cache_dir = str(tmp_path / "cache")
    settings.ensure_dirs()
    root = tmp_path / "src"
    root.mkdir()
    for i in range(3):
        (root / f"old_{i}.jpg").write_bytes(f"old-{i}".encode())
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("src", str(root)),
        )
        source_id = int(conn.execute("SELECT id FROM sources").fetchone()[0])
    store = JobStore(tmp_path / "jobs.db")
    discover_source(
        db, store, source_id=source_id, root_path=str(root), mode=Mode.FAST, settings=settings
    )
    with store._connect() as conn:
        conn.execute("UPDATE index_v3_jobs SET state='done'")
        conn.commit()
    for i in range(2):
        (root / f"new_{i}.jpg").write_bytes(f"new-{i}".encode())
    second = discover_source(
        db, store, source_id=source_id, root_path=str(root), mode=Mode.FAST, settings=settings
    )
    assert second.inserted == 2
    with store._connect() as conn:
        pending = list(
            conn.execute(
                "SELECT path, artifact FROM index_v3_jobs WHERE state='pending'"
            )
        )
    assert len(pending) == 2
    assert {row["artifact"] for row in pending} == {"preview"}
    assert all("new_" in str(row["path"]) for row in pending)


def test_turkish_source_path_is_preserved(tmp_path: Path):
    db = Database(tmp_path / "patterns.db")
    settings = AppSettings()
    settings.cache_dir = str(tmp_path / "cache")
    settings.ensure_dirs()
    root = tmp_path / "karşıdan yüklemeler"
    root.mkdir()
    (root / "leopar.jpg").write_bytes(b"leopard")
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("tr", str(root)),
        )
        source_id = int(conn.execute("SELECT id FROM sources").fetchone()[0])
    store = JobStore(tmp_path / "jobs.db")
    discover_source(
        db, store, source_id=source_id, root_path=str(root), mode=Mode.FAST, settings=settings
    )
    with db.connect() as conn:
        stored = str(conn.execute("SELECT path FROM files").fetchone()[0])
    assert "karşıdan yüklemeler" in stored
    assert "leopar.jpg" in stored
    assert "┼" not in stored
    assert normalize_path(stored).endswith("leopar.jpg")
    assert file_id_from_path(stored)


def test_startup_incomplete_ignores_heavy_pending_complete_light(tmp_path: Path):
    db = Database(tmp_path / "patterns.db")
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("src", str(tmp_path)),
        )
        conn.execute(
            """INSERT INTO files(path, filename, source_id, status, light_status, heavy_status)
               VALUES (?,?,?,?,?,?)""",
            (str(tmp_path / "a.jpg"), "a.jpg", 1, "indexed", "done", "pending"),
        )
    summaries = IndexQueueManager(db).sources_with_incomplete_work()
    assert summaries == []

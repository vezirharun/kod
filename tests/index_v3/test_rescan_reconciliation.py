from __future__ import annotations

import os
import time
from pathlib import Path

from core.app_status import _remaining_reason_summary
from core.app_status import refresh_lane_counts
from core.db import Database
from core.index_v3.background_scan import BackgroundIndexScan
from core.index_v3.discovery import discover_source
from core.index_v3.engine import IndexEngineV3
from core.index_v3.live_contract import build_live_progress, remaining_vs_executable
from core.index_v3.queues import JobStore
from core.index_v3.types import Artifact, Job, Mode, QueueKind
from core.settings import AppSettings


def _write_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _setup_source(tmp_path: Path) -> tuple[Database, JobStore, AppSettings, int, Path]:
    db_path = tmp_path / "patterns.db"
    jobs_path = tmp_path / "patterns.v3jobs.db"
    db = Database(db_path)
    settings = AppSettings()
    settings.db_path = str(db_path)
    settings.cache_dir = str(tmp_path / "cache")
    settings.faiss_dino_path = str(tmp_path / "dino.faiss")
    settings.faiss_clip_path = str(tmp_path / "clip.faiss")
    settings.ensure_dirs()
    root = tmp_path / "src"
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("src", str(root)),
        )
        source_id = int(conn.execute("SELECT id FROM sources").fetchone()[0])
    return db, JobStore(jobs_path), settings, source_id, root


def test_discover_source_detects_new_deleted_and_modified(tmp_path: Path):
    db, store, settings, source_id, root = _setup_source(tmp_path)
    for i in range(10):
        _write_file(root / f"f{i}.jpg", f"seed-{i}".encode("utf-8"))

    first = discover_source(
        db,
        store,
        source_id=source_id,
        root_path=str(root),
        mode=Mode.FAST,
        mark_missing=True,
        settings=settings,
    )
    assert first.inserted == 10
    assert first.missing_marked == 0

    time.sleep(1.1)
    _write_file(root / "new_a.jpg", b"new-a")
    _write_file(root / "new_b.jpg", b"new-b")
    os.remove(root / "f0.jpg")
    os.remove(root / "f1.jpg")
    _write_file(root / "f2.jpg", b"changed")

    second = discover_source(
        db,
        store,
        source_id=source_id,
        root_path=str(root),
        mode=Mode.FAST,
        mark_missing=True,
        settings=settings,
    )
    assert second.inserted == 2
    assert second.changed == 1
    assert second.missing_marked == 2
    assert second.unchanged >= 7

    assert second.purged_missing == 0
    with db.connect() as conn:
        remaining = {
            str(row["path"])
            for row in conn.execute(
                "SELECT path FROM files WHERE source_id=? AND status!='missing' ORDER BY path",
                (source_id,),
            ).fetchall()
        }
        missing = {
            str(row["path"])
            for row in conn.execute(
                "SELECT path FROM files WHERE source_id=? AND status='missing'",
                (source_id,),
            ).fetchall()
        }
    assert str(root / "f0.jpg") in missing
    assert str(root / "f1.jpg") in missing
    assert str(root / "f0.jpg") not in remaining
    assert str(root / "f1.jpg") not in remaining
    assert len(remaining) == 10


def test_discover_source_marks_missing_when_source_becomes_empty(tmp_path: Path):
    db, store, settings, source_id, root = _setup_source(tmp_path)
    _write_file(root / "only_a.jpg", b"a")
    _write_file(root / "only_b.jpg", b"b")

    discover_source(
        db,
        store,
        source_id=source_id,
        root_path=str(root),
        mode=Mode.FAST,
        mark_missing=True,
        settings=settings,
    )
    os.remove(root / "only_a.jpg")
    os.remove(root / "only_b.jpg")

    second = discover_source(
        db,
        store,
        source_id=source_id,
        root_path=str(root),
        mode=Mode.FAST,
        mark_missing=True,
        settings=settings,
    )
    assert second.scanned == 0
    assert second.missing_marked == 2
    assert second.purged_missing == 0
    with db.connect() as conn:
        assert int(conn.execute("SELECT COUNT(*) FROM files WHERE source_id=?", (source_id,)).fetchone()[0]) == 2
        assert int(conn.execute("SELECT COUNT(*) FROM files WHERE source_id=? AND status='missing'", (source_id,)).fetchone()[0]) == 2


def test_ten_remaining_files_enqueue_and_drain(tmp_path: Path):
    db, store, settings, source_id, root = _setup_source(tmp_path)
    from PIL import Image
    root.mkdir(parents=True, exist_ok=True)
    for i in range(10):
        Image.new("RGB", (24, 24), (i * 20, 40, 80)).save(root / f"remaining_{i}.jpg", "JPEG")

    discovered = discover_source(
        db,
        store,
        source_id=source_id,
        root_path=str(root),
        mode=Mode.FAST,
        mark_missing=True,
        settings=settings,
    )
    pending_before = store.count_pending(source_ids=[source_id])
    assert discovered.inserted == 10
    assert pending_before == 10
    settings.selected_source_ids = [source_id]
    status = refresh_lane_counts(settings)
    assert status["pending_jobs"] == 10
    assert status["pending_light_jobs"] == 10
    assert status["pending_heavy_jobs"] == 0

    engine = IndexEngineV3(
        db,
        job_db_path=tmp_path / "patterns.v3jobs.db",
        settings=settings,
    )
    report = engine.run(
        mode=Mode.FAST,
        sources=[{"id": source_id, "root_path": str(root)}],
        walk_disk=False,
    )

    # Preview sonra planner Thumbnail ekler; Hash Genel AI'dedir.
    assert report.worker["completed"] == 20
    assert store.count_pending(source_ids=[source_id]) == 0
    assert store.count_claimed() == 0


def test_remaining_reason_summary_breaks_down_non_executable_files(tmp_path: Path):
    db, store, settings, source_id, root = _setup_source(tmp_path)
    queued = root / "queued.jpg"
    retryable = root / "retryable.jpg"
    broken = root / "broken.jpg"
    unsupported = root / "unsupported.jpg"
    for p in (queued, retryable, broken, unsupported):
        _write_file(p, p.name.encode("utf-8"))
        db.upsert_file(
            {
                "path": str(p),
                "filename": p.name,
                "source_id": source_id,
                "status": "pending",
            }
        )
    with db.connect() as conn:
        ids = {
            Path(r["path"]).name: int(r["id"])
            for r in conn.execute("SELECT id, path FROM files").fetchall()
        }
    with db.connect() as conn:
        conn.execute(
            "UPDATE files SET light_status='pending', status='pending' WHERE id=?",
            (ids["queued.jpg"],),
        )
        conn.execute(
            """
            UPDATE files
            SET light_status='pending', status='error',
                repair_failure_reason='source_unavailable',
                repair_last_error='network share offline',
                repair_retryable=1, error_msg='network share offline'
            WHERE id=?
            """,
            (ids["retryable.jpg"],),
        )
        conn.execute(
            """
            UPDATE files
            SET light_status='failed', status='error',
                quarantine_reason='decode_error',
                repair_retryable=0, error_msg='decode failed'
            WHERE id=?
            """,
            (ids["broken.jpg"],),
        )
        conn.execute(
            """
            UPDATE files
            SET light_status='failed', status='error',
                quarantine_reason='format_unsupported',
                repair_retryable=0, error_msg='unsupported format'
            WHERE id=?
            """,
            (ids["unsupported.jpg"],),
        )
    store.enqueue(
        [
            Job(
                file_id=ids["queued.jpg"],
                artifact=Artifact.THUMBNAIL,
                queue=QueueKind.LIGHT,
                source_id=source_id,
                path=str(queued),
            )
        ]
    )

    summary = _remaining_reason_summary(settings, db, [source_id])
    assert summary["remaining_light_total"] == 4
    assert summary["remaining_light_executable_files"] == 1
    assert summary["remaining_reason_counts"]["queued"] == 1
    assert summary["remaining_reason_counts"]["retryable"] == 1
    assert summary["remaining_reason_counts"]["decode_error"] == 1
    assert summary["remaining_reason_counts"]["unsupported"] == 1


def test_remaining_reason_status_path_does_not_call_exists(monkeypatch):
    from core.app_status import _classify_remaining_reason

    def _boom(*_a, **_k):
        raise AssertionError("status path must not os.path.exists")

    monkeypatch.setattr("os.path.exists", _boom)
    monkeypatch.setattr("core.utils.os.path.exists", _boom)
    row = {
        "id": 1,
        "path": r"\\server\imalat2\x.jpg",
        "light_status": "pending",
        "status": "pending",
        "repair_retryable": 1,
    }
    assert _classify_remaining_reason(row, set()) == "blocked"


def test_startup_first_cycle_finds_new_and_deleted_despite_long_walk_interval(tmp_path: Path):
    """Açılış: walk_interval dolmamış olsa bile ilk cycle disk'i tarar."""
    db, store, settings, source_id, root = _setup_source(tmp_path)
    _write_file(root / "keep.jpg", b"keep")
    _write_file(root / "gone.jpg", b"gone")
    discover_source(
        db,
        store,
        source_id=source_id,
        root_path=str(root),
        mode=Mode.FAST,
        mark_missing=True,
        settings=settings,
    )
    _write_file(root / "added.jpg", b"added")
    os.remove(root / "gone.jpg")

    scan = BackgroundIndexScan(
        db,
        store,
        settings=settings,
        chunk_pause_sec=0,
        walk_interval_sec=10_000.0,
    )
    sources = lambda: [
        {"id": source_id, "root_path": str(root), "name": "src"}
    ]
    scan._one_cycle(sources, None, force_walk=False)
    assert scan.stats.new_found == 1
    assert scan.stats.missing_marked == 1
    assert scan.stats.changed_found == 0

    scan._one_cycle(sources, None, force_walk=False)
    assert scan.stats.new_found == 1
    assert scan.stats.missing_marked == 1


def test_deleted_files_count_is_unique_files_not_purge_rows(tmp_path: Path):
    db, store, settings, source_id, root = _setup_source(tmp_path)
    _write_file(root / "keep.jpg", b"keep")
    _write_file(root / "gone.jpg", b"gone")
    discover_source(
        db,
        store,
        source_id=source_id,
        root_path=str(root),
        mode=Mode.FAST,
        mark_missing=True,
        settings=settings,
    )
    os.remove(root / "gone.jpg")

    scan = BackgroundIndexScan(db, store, settings=settings, chunk_pause_sec=0)
    result = scan.run_once(
        source_provider=lambda: [
            {"id": source_id, "root_path": str(root), "name": "src"}
        ],
        purge_missing=True,
    )
    assert result["deleted_files"] == 1
    assert result["missing_marked"] == 1
    assert result["deleted_files"] == result["missing_marked"]


def test_file_queue_uses_remaining_files_not_job_count():
    rve = remaining_vs_executable(total=63, ready=58, executable_jobs=10)
    assert rve["remaining"] == 5
    assert rve["executable"] == 10

    live = build_live_progress(
        {
            "total": 63,
            "preview_ready": 58,
            "ai_final_ready": 58,
            "pending_jobs": 10,
            "pending_light_jobs": 10,
            "pending_heavy_jobs": 10,
            "v3_executable_jobs": 10,
        }
    )
    assert live["fast"]["file_queue"] == 5
    assert live["fast"]["remaining"] == 5
    assert live["fast"]["job_queue"] == 10
    assert live["general_ai"]["file_queue"] == 5
    assert live["file_queue"] == 5
    assert live["job_queue"] == 10


def test_unavailable_source_never_purges_existing_index(tmp_path: Path):
    db, store, settings, source_id, root = _setup_source(tmp_path)
    _write_file(root / "keep.jpg", b"keep")
    discover_source(
        db, store, source_id=source_id, root_path=str(root),
        mode=Mode.FAST, mark_missing=True, settings=settings,
    )
    os.remove(root / "keep.jpg")
    root.rmdir()
    result = discover_source(
        db, store, source_id=source_id, root_path=str(root),
        mode=Mode.FAST, mark_missing=True, settings=settings,
    )
    assert "source_unavailable" in result.errors
    assert result.missing_marked == 0
    assert result.purged_missing == 0
    with db.connect() as conn:
        assert int(conn.execute("SELECT COUNT(*) FROM files WHERE source_id=?", (source_id,)).fetchone()[0]) == 1

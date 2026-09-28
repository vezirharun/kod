"""Disk delete under a watched source purges that index row immediately."""

from __future__ import annotations

import os
from pathlib import Path

from core.db import Database
from core.disk_delete_purge import purge_disk_deletes
from core.index_v3.background_scan import BackgroundIndexScan
from core.index_v3.queues import JobStore
from core.index_v3.types import Artifact, Job, QueueKind
from core.settings import AppSettings
from core.utils import normalize_path
from core.win_subtree_watch import (
    REMOVED_ACTIONS,
    parse_notify_buffer,
    removed_paths_from_buffer,
)


def _pack(entries: list[tuple[int, str]]) -> bytes:
    encoded = [(action, name.encode("utf-16-le")) for action, name in entries]
    parts: list[bytes] = []
    for index, (action, raw) in enumerate(encoded):
        size = 12 + len(raw)
        pad = (4 - (size % 4)) % 4
        size += pad
        nxt = size if index < len(encoded) - 1 else 0
        parts.append(nxt.to_bytes(4, "little"))
        parts.append(action.to_bytes(4, "little"))
        parts.append(len(raw).to_bytes(4, "little"))
        parts.append(raw)
        parts.append(b"\x00" * pad)
    return b"".join(parts)


def _setup(tmp_path: Path):
    db_path = tmp_path / "patterns.db"
    cache = tmp_path / "cache"
    (cache / "thumbnails").mkdir(parents=True)
    (cache / "feature_previews").mkdir(parents=True)
    root = tmp_path / "src"
    root.mkdir()
    (root / "nested").mkdir()
    settings = AppSettings()
    settings.db_path = str(db_path)
    settings.cache_dir = str(cache)
    settings.faiss_dino_path = str(tmp_path / "dino.faiss")
    settings.faiss_clip_path = str(tmp_path / "clip.faiss")
    settings.folder_watch_enabled = True
    db = Database(db_path)
    store = JobStore(db_path.with_name("patterns.v3jobs.db"))
    return db, store, settings, root, cache


def _track(db: Database, path: Path, cache: Path, *, source_id: int = 1) -> int:
    thumb = cache / "thumbnails" / f"{path.stem}.jpg"
    preview = cache / "feature_previews" / f"{path.stem}.jpg"
    thumb.write_bytes(b"thumb")
    preview.write_bytes(b"preview")
    fid = int(
        db.upsert_file(
            {
                "path": str(path),
                "filename": path.name,
                "status": "indexed",
                "source_id": source_id,
                "thumbnail_path": str(thumb),
                "feature_preview_path": str(preview),
            }
        )
    )
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO features(file_id, phash) VALUES (?, ?)",
            (fid, "abc"),
        )
        conn.execute(
            "INSERT INTO index_queue(file_id, stage, status) VALUES (?, 'light', 'pending')",
            (fid,),
        )
        conn.execute(
            "INSERT INTO user_feedback(result_file_id, action) VALUES (?, 'like')",
            (fid,),
        )
    db.update_file_text_index(fid, text_search_blob=f"pattern {path.stem}")
    return fid


def _enqueue(store: JobStore, fid: int, path: Path) -> None:
    store.enqueue(
        [
            Job(
                file_id=fid,
                artifact=Artifact.THUMBNAIL,
                queue=QueueKind.LIGHT,
                path=str(path),
            ),
            Job(
                file_id=fid,
                artifact=Artifact.DINO,
                queue=QueueKind.HEAVY,
                path=str(path),
            ),
            Job(
                file_id=fid,
                artifact=Artifact.CLIP,
                queue=QueueKind.HEAVY,
                path=str(path),
            ),
        ]
    )
    with store._connect() as conn:
        conn.execute(
            "UPDATE index_v3_jobs SET state='claimed' WHERE file_id=? AND artifact='thumbnail'",
            (fid,),
        )
        conn.execute(
            "UPDATE index_v3_jobs SET state='done' WHERE file_id=? AND artifact='dino'",
            (fid,),
        )
        conn.commit()


def _counts(db: Database, fid: int) -> dict[str, int]:
    with db.connect() as conn:
        def one(sql: str) -> int:
            row = conn.execute(sql, (fid,)).fetchone()
            return int(row[0] or 0)

        return {
            "files": one("SELECT COUNT(*) FROM files WHERE id=?"),
            "features": one("SELECT COUNT(*) FROM features WHERE file_id=?"),
            "fts": one("SELECT COUNT(*) FROM files_fts WHERE file_id=?"),
            "queue": one("SELECT COUNT(*) FROM index_queue WHERE file_id=?"),
            "feedback": one("SELECT COUNT(*) FROM user_feedback WHERE result_file_id=?"),
        }


def test_deleted_file_purges_index_jobs_cache_and_faiss_flag(tmp_path: Path, monkeypatch):
    db, store, settings, root, cache = _setup(tmp_path)
    gone = root / "gone.jpg"
    keep = root / "keep.jpg"
    gone.write_bytes(b"gone")
    keep.write_bytes(b"keep")
    gone_id = _track(db, gone, cache)
    keep_id = _track(db, keep, cache)
    _enqueue(store, gone_id, gone)
    _enqueue(store, keep_id, keep)
    excluded: list[list[int]] = []

    def _spy(settings_arg, file_ids):
        excluded.append([int(x) for x in file_ids])
        return False

    monkeypatch.setattr("core.index_v3.faiss_sync.exclude_file_ids", _spy)
    gone.unlink()

    result = purge_disk_deletes(
        db,
        settings,
        [str(gone)],
        watched_roots=[str(root)],
    )

    assert result["removed"] == 1
    assert gone_id in result["file_ids"]
    assert _counts(db, gone_id) == {
        "files": 0,
        "features": 0,
        "fts": 0,
        "queue": 0,
        "feedback": 0,
    }
    assert _counts(db, keep_id)["files"] == 1
    assert not (cache / "thumbnails" / "gone.jpg").exists()
    assert not (cache / "feature_previews" / "gone.jpg").exists()
    assert (cache / "thumbnails" / "keep.jpg").is_file()
    assert (cache / "feature_previews" / "keep.jpg").is_file()
    assert db.get_meta("faiss_needs_rebuild") == "1"
    assert excluded and gone_id in excluded[0] and keep_id not in excluded[0]
    with store._connect() as conn:
        rows = conn.execute(
            "SELECT artifact, state FROM index_v3_jobs WHERE file_id=? ORDER BY artifact",
            (gone_id,),
        ).fetchall()
        keep_rows = conn.execute(
            "SELECT COUNT(*) FROM index_v3_jobs WHERE file_id=? AND state IN ('pending','claimed')",
            (keep_id,),
        ).fetchone()
    assert [(r["artifact"], r["state"]) for r in rows] == [("dino", "done")]
    assert int(keep_rows[0]) == 2


def test_directory_event_purges_only_missing_direct_child(tmp_path: Path):
    db, _store, settings, root, cache = _setup(tmp_path)
    gone = root / "gone.jpg"
    keep = root / "keep.jpg"
    nested = root / "nested" / "still.jpg"
    for path in (gone, keep, nested):
        path.write_bytes(b"img")
    gone_id = _track(db, gone, cache)
    keep_id = _track(db, keep, cache)
    nested_id = _track(db, nested, cache)
    gone.unlink()

    result = purge_disk_deletes(
        db,
        settings,
        [str(root)],
        watched_roots=[str(root)],
    )

    assert result["removed"] == 1
    assert result["file_ids"] == [gone_id]
    assert db.get_file_by_id(keep_id) is not None
    assert db.get_file_by_id(nested_id) is not None
    assert (cache / "thumbnails" / "keep.jpg").is_file()
    assert not (cache / "thumbnails" / "gone.jpg").exists()


def test_deleted_subdirectory_purges_nested_files_only(tmp_path: Path):
    db, _store, settings, root, cache = _setup(tmp_path)
    nested_dir = root / "nested"
    gone = nested_dir / "still.jpg"
    keep = root / "keep.jpg"
    gone.write_bytes(b"img")
    keep.write_bytes(b"img")
    gone_id = _track(db, gone, cache)
    keep_id = _track(db, keep, cache)
    gone.unlink()
    nested_dir.rmdir()

    result = purge_disk_deletes(
        db,
        settings,
        [str(nested_dir)],
        watched_roots=[str(root)],
    )

    assert result["file_ids"] == [gone_id]
    assert db.get_file_by_id(gone_id) is None
    assert db.get_file_by_id(keep_id) is not None


def test_live_file_and_unwatched_path_are_kept(tmp_path: Path):
    db, _store, settings, root, cache = _setup(tmp_path)
    live = root / "keep.jpg"
    outside = tmp_path / "other.jpg"
    live.write_bytes(b"img")
    outside.write_bytes(b"img")
    live_id = _track(db, live, cache)
    outside_id = _track(db, outside, cache)
    outside.unlink()

    kept = purge_disk_deletes(
        db,
        settings,
        [str(live), str(outside)],
        watched_roots=[str(root)],
    )
    assert kept["removed"] == 0
    assert db.get_file_by_id(live_id) is not None
    assert db.get_file_by_id(outside_id) is not None


def test_tree_reconcile_purges_only_files_missing_on_disk(tmp_path: Path):
    db, _store, settings, root, cache = _setup(tmp_path)
    gone = root / "nested" / "still.jpg"
    keep = root / "keep.jpg"
    gone.write_bytes(b"img")
    keep.write_bytes(b"img")
    gone_id = _track(db, gone, cache)
    keep_id = _track(db, keep, cache)
    gone.unlink()

    result = purge_disk_deletes(
        db,
        settings,
        [],
        watched_roots=[str(root)],
        tree_paths=[str(root)],
    )
    assert result["file_ids"] == [gone_id]
    assert db.get_file_by_id(keep_id) is not None


def test_background_missing_purge_removes_cache(tmp_path: Path):
    db, store, settings, root, cache = _setup(tmp_path)
    gone = root / "gone.jpg"
    keep = root / "keep.jpg"
    gone.write_bytes(b"img")
    keep.write_bytes(b"img")
    gone_id = _track(db, gone, cache)
    keep_id = _track(db, keep, cache)
    with db.connect() as conn:
        conn.execute("UPDATE files SET status='missing' WHERE id=?", (gone_id,))

    scan = BackgroundIndexScan(db, store, settings=settings)
    scan._purge_missing_records()

    assert db.get_file_by_id(gone_id) is None
    assert db.get_file_by_id(keep_id) is not None
    assert not (cache / "thumbnails" / "gone.jpg").exists()
    assert not (cache / "feature_previews" / "gone.jpg").exists()
    assert (cache / "thumbnails" / "keep.jpg").is_file()
    assert scan.stats.purged_missing == 1


def test_removed_paths_from_buffer_joins_nested_names(tmp_path: Path):
    root = tmp_path / "src"
    data = _pack([(2, "nested\\gone.jpg"), (1, "keep.jpg"), (4, "old.jpg")])
    assert removed_paths_from_buffer(str(root), data) == [
        os.path.normpath(str(root / "nested" / "gone.jpg")),
        os.path.normpath(str(root / "old.jpg")),
    ]


def test_parse_notify_buffer_keeps_removed_and_renamed_away():
    data = _pack(
        [
            (1, "added.jpg"),
            (2, "gone.jpg"),
            (3, "touched.jpg"),
            (4, "old\\nested.jpg"),
            (5, "new.jpg"),
        ]
    )
    events = parse_notify_buffer(data)
    assert events[0] == (1, "added.jpg")
    assert events[1] == (2, "gone.jpg")
    assert [name for action, name in events if action in REMOVED_ACTIONS] == [
        "gone.jpg",
        "old\\nested.jpg",
    ]


def test_folder_watch_queues_delete_without_waiting_for_scan(tmp_path: Path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from ui.folder_watch_controller import _DiskPurgeThread, FolderWatchController

    _ = QApplication.instance() or QApplication([])
    db, store, settings, root, cache = _setup(tmp_path)
    gone = root / "gone.jpg"
    keep = root / "keep.jpg"
    gone.write_bytes(b"img")
    keep.write_bytes(b"img")
    gone_id = _track(db, gone, cache)
    keep_id = _track(db, keep, cache)
    _enqueue(store, gone_id, gone)
    gone.unlink()

    class _Sources:
        def __init__(self, database):
            self.db = database

    ctl = FolderWatchController(settings, _Sources(db))
    ctl._watch_roots = [normalize_path(root)]
    ctl._on_directory_changed(str(root))
    ctl._delete_timer.stop()

    assert normalize_path(root) in ctl._deleted_paths
    assert ctl._debounce.isActive()
    assert ctl._delete_timer.interval() == 100
    assert ctl._debounce.interval() >= 1000

    worker = _DiskPurgeThread(
        db,
        settings,
        list(ctl._deleted_paths),
        list(ctl._watch_roots),
        [],
    )
    worker.run()
    assert db.get_file_by_id(gone_id) is None
    assert db.get_file_by_id(keep_id) is not None
    assert not (cache / "thumbnails" / "gone.jpg").exists()
    with store._connect() as conn:
        pending = conn.execute(
            """
            SELECT COUNT(*) FROM index_v3_jobs
            WHERE file_id=? AND state IN ('pending','claimed')
            """,
            (gone_id,),
        ).fetchone()
    assert int(pending[0]) == 0

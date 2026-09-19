from __future__ import annotations

import time
from pathlib import Path

from core.db import Database
from core.index_v3.discovery import discover_source, should_walk_on_index_start
from core.index_v3.engine import IndexEngineV3
from core.index_v3.queues import JobStore
from core.index_v3.types import Mode
from core.settings import AppSettings
from core.sources import SourceManager


def test_should_walk_explicit_source_even_if_files_exist():
    assert should_walk_on_index_start(
        explicit_source_id=4,
        existing_file_count=500,
        root_path="F:/yeni",
    )
    assert not should_walk_on_index_start(
        explicit_source_id=0,
        existing_file_count=500,
        root_path="F:/eski",
    )
    assert should_walk_on_index_start(
        explicit_source_id=0,
        existing_file_count=0,
        root_path="F:/bos",
    )
    assert not should_walk_on_index_start(
        explicit_source_id=4,
        existing_file_count=0,
        root_path="F:/x",
        post_ga=True,
    )


def test_add_source_enqueues_jobs_and_indexes_first_file(tmp_path: Path):
    from PIL import Image

    db_path = tmp_path / "patterns.db"
    db = Database(db_path)
    settings = AppSettings()
    settings.db_path = str(db_path)
    settings.cache_dir = str(tmp_path / "cache")
    settings.faiss_dino_path = str(tmp_path / "dino.faiss")
    settings.faiss_clip_path = str(tmp_path / "clip.faiss")
    settings.ensure_dirs()
    root = tmp_path / "yeni_kaynak"
    root.mkdir()
    for i in range(3):
        Image.new("RGB", (32, 32), (i * 40, 80, 120)).save(root / f"n{i}.jpg", "JPEG")

    mgr = SourceManager(settings, run_maintenance=False)
    sid = int(mgr.add_source("yeni", str(root), is_active=True))
    assert sid > 0
    listed = mgr.list_sources()
    assert any(int(s["id"]) == sid for s in listed)

    store = JobStore(db_path.with_name("patterns.v3jobs.db"))
    t0 = time.perf_counter()
    discovered = discover_source(
        db,
        store,
        source_id=sid,
        root_path=str(root),
        mode=Mode.FAST,
        mark_missing=True,
        settings=settings,
    )
    discover_ms = (time.perf_counter() - t0) * 1000
    assert discovered.scanned == 3
    assert discovered.inserted == 3
    assert discovered.jobs_enqueued >= 3
    pending = store.count_pending(source_ids=[sid])
    assert pending >= 3

    engine = IndexEngineV3(
        db,
        job_db_path=tmp_path / "patterns.v3jobs.db",
        settings=settings,
        use_real_extractors=False,
    )
    t1 = time.perf_counter()
    report = engine.run(
        mode=Mode.FAST,
        sources=[{"id": sid, "root_path": str(root)}],
        walk_disk=False,
    )
    first_drain_ms = (time.perf_counter() - t1) * 1000
    assert int(report.worker.get("completed") or 0) >= 1
    assert store.count_pending(source_ids=[sid]) == 0
    print(
        f"discover_ms={discover_ms:.1f} first_drain_ms={first_drain_ms:.1f} "
        f"jobs={discovered.jobs_enqueued} completed={report.worker.get('completed')}"
    )


def _make_source(tmp_path: Path, name: str, n: int = 2):
    from PIL import Image

    db_path = tmp_path / "patterns.db"
    db = Database(db_path)
    settings = AppSettings()
    settings.db_path = str(db_path)
    settings.cache_dir = str(tmp_path / "cache")
    settings.faiss_dino_path = str(tmp_path / "dino.faiss")
    settings.faiss_clip_path = str(tmp_path / "clip.faiss")
    settings.ensure_dirs()
    root = tmp_path / name
    root.mkdir()
    for i in range(n):
        Image.new("RGB", (24, 24), (i * 20, 40, 80)).save(root / f"{name}{i}.jpg", "JPEG")
    mgr = SourceManager(settings, run_maintenance=False)
    sid = int(mgr.add_source(name, str(root), is_active=True))
    store = JobStore(db_path.with_name("patterns.v3jobs.db"))
    return db, settings, mgr, store, sid, root


def test_new_source_row_and_file_count_after_discover(tmp_path: Path):
    db, settings, mgr, store, sid, root = _make_source(tmp_path, "local_yeni", 2)
    mgr.db.update_source_stats(sid, file_count=0, error_count=0, cache_status="queued")
    row = mgr.get_source(sid)
    assert row is not None
    assert str(row.get("cache_status")) == "queued"
    st = discover_source(
        db, store, source_id=sid, root_path=str(root), mode=Mode.FAST, settings=settings
    )
    assert st.scanned == 2
    assert st.jobs_enqueued >= 2
    refreshed = mgr.get_source(sid)
    assert int(refreshed.get("file_count") or 0) == 2


def test_unc_source_root_is_dir(monkeypatch, tmp_path: Path):
    from core.index_v3.discovery import source_root_is_dir

    assert source_root_is_dir(str(tmp_path))
    assert not source_root_is_dir("")

    def _isdir(path: str) -> bool:
        return "imalat2" in str(path).replace("/", "\\").lower()

    monkeypatch.setattr("core.index_v3.discovery.os.path.isdir", _isdir)
    assert source_root_is_dir(r"\\server\imalat2")


def test_duplicate_add_does_not_duplicate_jobs(tmp_path: Path):
    db, settings, mgr, store, sid, root = _make_source(tmp_path, "dup", 2)
    first = discover_source(
        db, store, source_id=sid, root_path=str(root), mode=Mode.FAST, settings=settings
    )
    pending1 = store.count_pending(source_ids=[sid])
    second = discover_source(
        db, store, source_id=sid, root_path=str(root), mode=Mode.FAST, settings=settings
    )
    pending2 = store.count_pending(source_ids=[sid])
    assert first.inserted == 2
    assert second.inserted == 0
    assert pending2 == pending1


def test_existing_source_intact_when_new_source_queued(tmp_path: Path):
    db, settings, mgr, store, sid_a, root_a = _make_source(tmp_path, "eski", 2)
    discover_source(
        db, store, source_id=sid_a, root_path=str(root_a), mode=Mode.FAST, settings=settings
    )
    pending_a = store.count_pending(source_ids=[sid_a])
    files_a = db.count_files_for_source(sid_a)

    from PIL import Image

    root_b = tmp_path / "yeni"
    root_b.mkdir()
    Image.new("RGB", (24, 24), (9, 9, 9)).save(root_b / "b0.jpg", "JPEG")
    sid_b = int(mgr.add_source("yeni", str(root_b), is_active=True))
    discover_source(
        db, store, source_id=sid_b, root_path=str(root_b), mode=Mode.FAST, settings=settings
    )
    assert store.count_pending(source_ids=[sid_a]) == pending_a
    assert store.count_pending(source_ids=[sid_b]) >= 1
    assert db.count_files_for_source(sid_a) == files_a


def test_resume_partial_initial_scan(tmp_path: Path):
    db, settings, mgr, store, sid, root = _make_source(tmp_path, "resume", 3)
    discover_source(
        db, store, source_id=sid, root_path=str(root), mode=Mode.FAST, settings=settings
    )
    pending_before = store.count_pending(source_ids=[sid])
    assert pending_before >= 3
    engine = IndexEngineV3(
        db,
        job_db_path=tmp_path / "patterns.v3jobs.db",
        settings=settings,
        use_real_extractors=False,
    )
    engine.run(
        mode=Mode.FAST,
        sources=[{"id": sid, "root_path": str(root)}],
        walk_disk=False,
        max_jobs_per_queue=1,
    )
    leftover = store.count_pending(source_ids=[sid])
    assert leftover < pending_before
    engine.run(
        mode=Mode.FAST,
        sources=[{"id": sid, "root_path": str(root)}],
        walk_disk=False,
    )
    assert store.count_pending(source_ids=[sid]) == 0


def test_queued_status_not_overwritten_to_empty(tmp_path: Path):
    _db, settings, mgr, _store, sid, _root = _make_source(tmp_path, "bekleyen", 0)
    mgr.db.update_source_stats(sid, file_count=0, error_count=0, cache_status="queued")
    mgr.update_source_stats(sid)
    assert str((mgr.get_source(sid) or {}).get("cache_status")) == "queued"


def test_streaming_found_before_walk_finishes(tmp_path: Path):
    from PIL import Image

    db_path = tmp_path / "patterns.db"
    db = Database(db_path)
    settings = AppSettings()
    settings.db_path = str(db_path)
    settings.cache_dir = str(tmp_path / "cache")
    settings.faiss_dino_path = str(tmp_path / "dino.faiss")
    settings.faiss_clip_path = str(tmp_path / "clip.faiss")
    settings.ensure_dirs()
    root = tmp_path / "stream"
    root.mkdir()
    for i in range(4):
        Image.new("RGB", (20, 20), (i, 10, 20)).save(root / f"s{i}.jpg", "JPEG")
    mgr = SourceManager(settings, run_maintenance=False)
    sid = int(mgr.add_source("stream", str(root), is_active=True))
    store = JobStore(db_path.with_name("patterns.v3jobs.db"))
    seen: list[int] = []
    first_queued = []

    def _cb(info: dict):
        seen.append(int(info.get("found") or 0))
        if not first_queued:
            first_queued.append(int(info.get("queued") or info.get("jobs_enqueued") or 0))

    st = discover_source(
        db,
        store,
        source_id=sid,
        root_path=str(root),
        mode=Mode.FAST,
        settings=settings,
        progress_callback=_cb,
        stream_chunk=25,
    )
    assert st.scanned == 4
    assert seen
    assert seen[0] >= 1
    assert seen[0] < st.scanned or len(seen) >= 2
    assert seen[-1] == 4
    assert first_queued and first_queued[0] >= 1
    assert int((mgr.get_source(sid) or {}).get("file_count") or 0) == 4
    assert str((mgr.get_source(sid) or {}).get("cache_status")) == "scanning"


def test_unc_forward_slash_root_is_dir(monkeypatch):
    from core.index_v3.discovery import source_root_is_dir

    def _isdir(path: str) -> bool:
        norm = str(path).replace("/", "\\").lower()
        return "imalat2" in norm

    monkeypatch.setattr("core.index_v3.discovery.os.path.isdir", _isdir)
    assert source_root_is_dir("//server/imalat2")
    assert source_root_is_dir(r"\\server\imalat2")


def test_live_source_row_overlay():
    class _Host:
        def __init__(self):
            self._source_live_stats = {
                9: {
                    "found": 1247,
                    "indexed": 1180,
                    "queued": 67,
                    "label": "Taranıyor · 1.180/1.247 · kuyruk 67",
                }
            }

        def _merge_source_live(self, sources: list) -> list:
            out = []
            for src in sources:
                row = dict(src)
                live = self._source_live_stats.get(int(row.get("id") or 0))
                if live:
                    row["file_count"] = int(live.get("found") or 0)
                    row["cache_status"] = "scanning"
                    row["status_label"] = live["label"]
                out.append(row)
            return out

    merged = _Host()._merge_source_live(
        [{"id": 9, "file_count": 0, "cache_status": "empty"}]
    )
    assert merged[0]["file_count"] == 1247
    assert merged[0]["cache_status"] == "scanning"
    assert "Taranıyor" in merged[0]["status_label"]


def test_kickoff_skips_when_sidecar_already_active():
    class _W:
        def isRunning(self):
            return True

    class _Host:
        def __init__(self):
            self._source_fast_workers = {4: _W()}

        def _source_fast_index_active(self, source_id: int) -> bool:
            w = self._source_fast_workers.get(int(source_id or 0))
            return bool(w is not None and w.isRunning())

    host = _Host()
    assert host._source_fast_index_active(4)
    assert not host._source_fast_index_active(9)


def test_general_ssot_grows_with_new_source_and_drops_on_remove(tmp_path: Path):
    """11.791 + 1.000 = 12.791; kaynak kalkınca genel toplam düşer. Ayrı mini indeks yok."""
    from PIL import Image

    from core.index_v3.scope import resolve_index_scope
    from core.index_v3.ui_bridge import count_v3_ssot

    db_path = tmp_path / "patterns.db"
    db = Database(db_path)
    settings = AppSettings()
    settings.db_path = str(db_path)
    settings.cache_dir = str(tmp_path / "cache")
    settings.ensure_dirs()
    old = tmp_path / "eski"
    new = tmp_path / "yeni"
    old.mkdir()
    new.mkdir()
    for i in range(3):
        Image.new("RGB", (16, 16), (i, 10, 20)).save(old / f"e{i}.jpg", "JPEG")
    for i in range(2):
        Image.new("RGB", (16, 16), (i, 30, 40)).save(new / f"n{i}.jpg", "JPEG")

    mgr = SourceManager(settings, run_maintenance=False)
    old_id = int(mgr.add_source("eski", str(old), is_active=True))
    store = JobStore(db_path.with_name("patterns.v3jobs.db"))
    discover_source(
        db, store, source_id=old_id, root_path=str(old), mode=Mode.FAST, settings=settings
    )
    old_total = int(count_v3_ssot(db, [old_id]).get("total") or 0)
    assert old_total == 3

    new_id = int(mgr.add_source("yeni", str(new), is_active=True))
    discover_source(
        db, store, source_id=new_id, root_path=str(new), mode=Mode.FAST, settings=settings
    )
    combined = int(count_v3_ssot(db, [old_id, new_id]).get("total") or 0)
    assert combined == old_total + 2
    assert int(count_v3_ssot(db, [old_id]).get("total") or 0) == old_total

    mgr.remove_source(new_id)
    scope = resolve_index_scope(db, [])
    assert new_id not in scope.source_ids
    assert old_id in scope.source_ids
    after = int(count_v3_ssot(db, scope.source_ids).get("total") or 0)
    assert after == old_total


def test_engine_include_source_id_expands_session_scope():
    class _E:
        def include_source_id(self, source_id: int) -> None:
            ids = getattr(self, "_session_source_ids", None)
            sid = int(source_id or 0)
            if sid <= 0:
                return
            if ids is None:
                self._session_source_ids = [sid]
                return
            if sid not in ids:
                ids.append(sid)

    eng = _E()
    eng._session_source_ids = [8]
    eng.include_source_id(12)
    eng.include_source_id(12)
    assert eng._session_source_ids == [8, 12]


def _sql_files(n: int, prefix: str, source_id: int) -> list[dict]:
    return [
        {
            "path": f"{prefix}/f{i:05d}.jpg",
            "filename": f"f{i:05d}.jpg",
            "source_id": source_id,
            "file_size": 10,
            "mtime": 1.0 + i,
        }
        for i in range(n)
    ]


def test_ssot_11791_plus_1000_metadata_not_fake_100(tmp_path: Path):
    from core.index_v3.scope import resolve_index_scope

    db_path = tmp_path / "patterns.db"
    db = Database(db_path)
    settings = AppSettings()
    settings.db_path = str(db_path)
    settings.ensure_dirs()
    mgr = SourceManager(settings, run_maintenance=False)
    old_id = int(mgr.add_source("eski", str(tmp_path / "e"), is_active=True))
    new_id = int(mgr.add_source("yeni", str(tmp_path / "n"), is_active=True))
    db.bulk_insert_source_files(_sql_files(11_791, "E:/eski", old_id))
    db.bulk_insert_source_files(_sql_files(1_000, "N:/yeni", new_id))
    with db.connect() as conn:
        conn.execute(
            "UPDATE files SET width=64, format_metadata='{\"ok\":1}' WHERE source_id=?",
            (old_id,),
        )
        conn.commit()
        total, meta = conn.execute(
            """
            SELECT
              COUNT(*) AS total,
              SUM(CASE WHEN ifnull(format_metadata,'') NOT IN ('','{}','null')
                            OR ifnull(width,0)>0
                       THEN 1 ELSE 0 END) AS meta
            FROM files
            WHERE status NOT IN ('excluded_internal','missing')
              AND source_id IN (?,?)
            """,
            (old_id, new_id),
        ).fetchone()

    assert int(total) == 12_791
    assert int(meta) == 11_791
    remaining = int(total) - int(meta)
    assert remaining == 1_000
    pct = 100.0 * int(meta) / int(total)
    assert pct < 100

    with db.connect() as conn:
        conn.execute(
            "UPDATE files SET width=64, format_metadata='{\"ok\":1}' WHERE source_id=?",
            (new_id,),
        )
        conn.commit()
        done_meta = int(
            conn.execute(
                """
                SELECT SUM(CASE WHEN ifnull(format_metadata,'') NOT IN ('','{}','null')
                                  OR ifnull(width,0)>0
                                THEN 1 ELSE 0 END)
                FROM files
                WHERE status NOT IN ('excluded_internal','missing')
                  AND source_id IN (?,?)
                """,
                (old_id, new_id),
            ).fetchone()[0]
            or 0
        )
    assert done_meta == 12_791
    assert 100.0 * done_meta / 12_791 == 100

    mgr.remove_source(new_id)
    scope = resolve_index_scope(db, [])
    with db.connect() as conn:
        after = int(
            conn.execute(
                f"""
                SELECT COUNT(*) FROM files
                WHERE status NOT IN ('excluded_internal','missing')
                  AND source_id IN ({",".join("?" * len(scope.source_ids))})
                """,
                list(scope.source_ids),
            ).fetchone()[0]
            or 0
        )
    assert after == 11_791


def test_old_source_unchanged_not_requed(tmp_path: Path):
    from PIL import Image

    db_path = tmp_path / "patterns.db"
    db = Database(db_path)
    settings = AppSettings()
    settings.db_path = str(db_path)
    settings.cache_dir = str(tmp_path / "cache")
    settings.ensure_dirs()
    old = tmp_path / "eski"
    new = tmp_path / "yeni"
    old.mkdir()
    new.mkdir()
    Image.new("RGB", (16, 16), (1, 2, 3)).save(old / "a.jpg", "JPEG")
    Image.new("RGB", (16, 16), (4, 5, 6)).save(new / "b.jpg", "JPEG")
    mgr = SourceManager(settings, run_maintenance=False)
    old_id = int(mgr.add_source("eski", str(old), is_active=True))
    store = JobStore(db_path.with_name("patterns.v3jobs.db"))
    discover_source(
        db, store, source_id=old_id, root_path=str(old), mode=Mode.FAST, settings=settings
    )
    pending_old = store.count_pending(source_ids=[old_id])
    engine = IndexEngineV3(
        db,
        job_db_path=tmp_path / "patterns.v3jobs.db",
        settings=settings,
        use_real_extractors=False,
    )
    engine.run(
        mode=Mode.FAST,
        sources=[{"id": old_id, "root_path": str(old)}],
        walk_disk=False,
    )
    assert store.count_pending(source_ids=[old_id]) == 0
    new_id = int(mgr.add_source("yeni", str(new), is_active=True))
    discover_source(
        db, store, source_id=new_id, root_path=str(new), mode=Mode.FAST, settings=settings
    )
    assert store.count_pending(source_ids=[old_id]) == 0
    assert store.count_pending(source_ids=[new_id]) >= 1
    _ = pending_old


def test_fast_drain_starts_before_walk_finishes(tmp_path: Path):
    from PIL import Image

    db_path = tmp_path / "patterns.db"
    db = Database(db_path)
    settings = AppSettings()
    settings.db_path = str(db_path)
    settings.cache_dir = str(tmp_path / "cache")
    settings.ensure_dirs()
    root = tmp_path / "mix"
    root.mkdir()
    for i in range(8):
        Image.new("RGB", (16, 16), (i, 8, 8)).save(root / f"m{i}.jpg", "JPEG")
    mgr = SourceManager(settings, run_maintenance=False)
    sid = int(mgr.add_source("mix", str(root), is_active=True))
    found_while_pending: list[int] = []

    def _cb(info: dict):
        if info and info.get("phase") == "discovery":
            found_while_pending.append(int(info.get("found") or 0))

    engine = IndexEngineV3(
        db,
        job_db_path=tmp_path / "patterns.v3jobs.db",
        settings=settings,
        use_real_extractors=False,
    )
    report = engine.run(
        mode=Mode.FAST,
        sources=[{"id": sid, "root_path": str(root)}],
        walk_disk=True,
        progress_callback=_cb,
    )
    assert found_while_pending
    assert found_while_pending[0] >= 1
    assert int((report.discovery.get("sources") or [{}])[0].get("scanned") or 0) == 8
    assert int(report.worker.get("completed") or 0) >= 1


def test_delete_jobs_for_removed_source(tmp_path: Path):
    db_path = tmp_path / "patterns.db"
    db = Database(db_path)
    settings = AppSettings()
    settings.db_path = str(db_path)
    settings.ensure_dirs()
    mgr = SourceManager(settings, run_maintenance=False)
    sid = int(mgr.add_source("x", str(tmp_path / "x"), is_active=True))
    db.bulk_insert_source_files(_sql_files(3, "X:/x", sid))
    store = JobStore(db_path.with_name("patterns.v3jobs.db"))
    from core.index_v3.types import Artifact, Job, QueueKind

    fids = []
    with db.connect() as conn:
        fids = [int(r[0]) for r in conn.execute("SELECT id FROM files").fetchall()]
    store.enqueue(
        [
            Job(
                file_id=fid,
                artifact=Artifact.PREVIEW,
                queue=QueueKind.PREVIEW,
                source_id=sid,
                path=f"X:/x/f{i}.jpg",
            )
            for i, fid in enumerate(fids)
        ]
    )
    assert store.count_pending(source_ids=[sid]) >= 1
    assert store.delete_jobs_for_source(sid) >= 1
    assert store.count_pending(source_ids=[sid]) == 0

from __future__ import annotations

from pathlib import Path

from core.db import Database
from core.index_v3.discovery import count_indexable_files, discover_source
from core.index_v3.engine import IndexEngineV3
from core.index_v3.queues import JobStore
from core.index_v3.types import Mode
from core.index_v3.ui_bridge import count_v3_ssot
from core.settings import AppSettings
from core.sources import SourceManager


def test_count_indexable_files_reports_progress_then_total(tmp_path: Path):
    from PIL import Image

    root = tmp_path / "desenler"
    root.mkdir()
    seen: list[int] = []
    for i in range(4):
        Image.new("RGB", (8, 8), (i, 10, 20)).save(root / f"a{i}.jpg", "JPEG")
    (root / "readme.txt").write_text("not an image")
    total = count_indexable_files(str(root), on_progress=seen.append, every=2)
    assert total == 4
    assert seen[0] == 1
    assert seen[-1] == 4


def test_count_empty_folder_is_zero(tmp_path: Path):
    empty = tmp_path / "bos"
    empty.mkdir()
    assert count_indexable_files(str(empty)) == 0


def test_count_missing_root_is_zero(tmp_path: Path):
    assert count_indexable_files(str(tmp_path / "yok")) == 0


def _prep(tmp_path: Path) -> tuple:
    from PIL import Image

    db_path = tmp_path / "patterns.db"
    db = Database(db_path)
    settings = AppSettings()
    settings.db_path = str(db_path)
    settings.cache_dir = str(tmp_path / "cache")
    settings.faiss_dino_path = str(tmp_path / "dino.faiss")
    settings.faiss_clip_path = str(tmp_path / "clip.faiss")
    settings.ensure_dirs()
    return db, settings, Image


def test_discovery_grows_total_and_enqueues_before_walk_ends(tmp_path: Path):
    db, settings, Image = _prep(tmp_path)
    old = tmp_path / "eski"
    new = tmp_path / "yeni"
    old.mkdir()
    new.mkdir()
    for i in range(3):
        Image.new("RGB", (12, 12), (i, 1, 2)).save(old / f"e{i}.jpg", "JPEG")
    for i in range(5):
        Image.new("RGB", (12, 12), (i, 3, 4)).save(new / f"n{i}.jpg", "JPEG")
    mgr = SourceManager(settings, run_maintenance=False)
    old_id = int(mgr.add_source("eski", str(old), is_active=True))
    new_id = int(mgr.add_source("yeni", str(new), is_active=True))
    store = JobStore(Path(settings.db_path).with_name("patterns.v3jobs.db"))
    discover_source(
        db, store, source_id=old_id, root_path=str(old), mode=Mode.FAST, settings=settings
    )
    baseline = int(count_v3_ssot(db, [old_id, new_id]).get("total") or 0)
    assert baseline == 3
    seen_total: list[int] = []
    queued_early: list[int] = []

    def _cb(info: dict) -> None:
        seen_total.append(int(count_v3_ssot(db, [old_id, new_id]).get("total") or 0))
        if not queued_early:
            queued_early.append(int(info.get("jobs_enqueued") or info.get("queued") or 0))

    st = discover_source(
        db,
        store,
        source_id=new_id,
        root_path=str(new),
        mode=Mode.FAST,
        settings=settings,
        progress_callback=_cb,
        stream_chunk=25,
    )
    assert st.scanned == 5
    assert queued_early and queued_early[0] >= 1
    assert seen_total[0] >= baseline + 1
    assert seen_total[-1] == baseline + 5
    assert int(count_v3_ssot(db, [old_id, new_id]).get("total") or 0) == 8


def test_fast_engine_drains_jobs_from_streaming_discovery(tmp_path: Path):
    db, settings, Image = _prep(tmp_path)
    root = tmp_path / "kaynak"
    root.mkdir()
    for i in range(6):
        Image.new("RGB", (16, 16), (i, 8, 9)).save(root / f"k{i}.jpg", "JPEG")
    mgr = SourceManager(settings, run_maintenance=False)
    sid = int(mgr.add_source("k", str(root), is_active=True))
    engine = IndexEngineV3(
        db,
        job_db_path=Path(settings.db_path).with_name("patterns.v3jobs.db"),
        settings=settings,
        use_real_extractors=False,
    )
    saw_discover = []
    saw_claim = []

    def _cb(info=None):
        if not info:
            return
        if str(info.get("phase") or "") == "discovery":
            saw_discover.append(int(info.get("found") or 0))
        elif str(info.get("phase") or "") == "claimed":
            saw_claim.append(1)

    report = engine.run(
        mode=Mode.FAST,
        sources=[{"id": sid, "root_path": str(root)}],
        walk_disk=True,
        progress_callback=_cb,
    )
    scanned = 0
    for row in (report.discovery or {}).get("sources") or []:
        scanned = max(scanned, int(row.get("scanned") or 0))
    assert scanned == 6
    assert int(report.worker.get("completed") or 0) >= 1
    assert saw_discover
    assert saw_discover[0] >= 1


def test_complete_engine_drains_jobs_during_streaming_discovery(tmp_path: Path):
    db, settings, Image = _prep(tmp_path)
    root = tmp_path / "complete_src"
    root.mkdir()
    for i in range(6):
        Image.new("RGB", (16, 16), (i, 8, 9)).save(root / f"c{i}.jpg", "JPEG")
    mgr = SourceManager(settings, run_maintenance=False)
    sid = int(mgr.add_source("c", str(root), is_active=True))
    engine = IndexEngineV3(
        db,
        job_db_path=Path(settings.db_path).with_name("patterns.v3jobs.db"),
        settings=settings,
        use_real_extractors=False,
    )
    saw_discover: list[int] = []
    saw_claim: list[int] = []

    def _cb(info=None):
        if not info:
            return
        if str(info.get("phase") or "") == "discovery":
            saw_discover.append(int(info.get("found") or 0))
        elif str(info.get("phase") or "") == "claimed":
            saw_claim.append(1)

    report = engine.run(
        mode=Mode.COMPLETE,
        sources=[{"id": sid, "root_path": str(root)}],
        walk_disk=True,
        progress_callback=_cb,
    )
    scanned = 0
    for row in (report.discovery or {}).get("sources") or []:
        scanned = max(scanned, int(row.get("scanned") or 0))
    assert scanned == 6
    assert int(report.worker.get("completed") or 0) >= 1
    assert saw_discover
    assert saw_claim



def test_count_indexable_files_reports_progress_then_total(tmp_path: Path):
    from PIL import Image

    root = tmp_path / "desenler"
    root.mkdir()
    seen: list[int] = []
    for i in range(4):
        Image.new("RGB", (8, 8), (i, 10, 20)).save(root / f"a{i}.jpg", "JPEG")
    (root / "readme.txt").write_text("not an image")
    total = count_indexable_files(str(root), on_progress=seen.append, every=2)
    assert total == 4
    assert seen[0] == 1
    assert seen[-1] == 4


def test_count_empty_folder_is_zero(tmp_path: Path):
    empty = tmp_path / "bos"
    empty.mkdir()
    assert count_indexable_files(str(empty)) == 0


def test_count_missing_root_is_zero(tmp_path: Path):
    assert count_indexable_files(str(tmp_path / "yok")) == 0

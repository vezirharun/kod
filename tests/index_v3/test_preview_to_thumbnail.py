from pathlib import Path

from PIL import Image

from core.index_v3.real_processor import RealArtifactProcessor
from core.index_v3.types import Artifact, Job, QueueKind
from core.index_v3.worker import WorkerStats
from core.preview_cache import FeaturePreviewCache
from core.settings import AppSettings
from core.thumbnailer import Thumbnailer


class FakeDB:
    def __init__(self, row):
        self.row = dict(row)
        self.readiness = {}
    def get_file_by_id(self, _):
        return dict(self.row)
    def upsert_file(self, payload):
        self.row.update(payload)
    def update_physical_readiness(self, _fid, *, thumbnail_ready, preview_ready, requeue_missing=False):
        self.readiness = {
            "thumbnail": bool(thumbnail_ready),
            "preview": bool(preview_ready),
        }


def test_thumbnail_uses_existing_preview_without_source_read(tmp_path, monkeypatch):
    source = tmp_path / "source.tif"
    source.write_bytes(b"not-readable-source")
    cache = tmp_path / "cache"
    thumb = Thumbnailer(str(cache), 64, "webp")
    preview = FeaturePreviewCache(str(cache), 128, "webp")
    preview_path = preview.preview_path_for(str(source))
    Image.new("RGB", (128, 80), (30, 60, 90)).save(preview_path, "WEBP")

    settings = AppSettings(cache_dir=str(cache), ai_embedding_enabled=False)
    proc = RealArtifactProcessor(settings, retries=0, enable_ai=False)
    db = FakeDB({
        "path": str(source),
        "feature_preview_path": str(preview_path),
        "thumbnail_path": "",
        "mtime": 0,
    })
    job = Job(file_id=1, artifact=Artifact.THUMBNAIL, queue=QueueKind.LIGHT, source_id=1, path=str(source))

    def fail_source(*args, **kwargs):
        raise AssertionError("kaynak dosya okunmamalı")
    monkeypatch.setattr(proc, "_process_light_bundle", fail_source)

    assert proc.process(db, job, WorkerStats()) is True
    assert Path(db.row["thumbnail_path"]).is_file()
    assert db.readiness == {"thumbnail": True, "preview": True}


def _two_files(tmp_path: Path):
    from PIL import Image

    from core.db import Database
    from core.index_v3.queues import JobStore
    from core.index_v3.types import Job

    db = Database(tmp_path / "p.db")
    store = JobStore(tmp_path / "jobs.db")
    root = tmp_path / "src"
    root.mkdir()
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("s", str(root)),
        )
    fids = []
    for i in range(2):
        p = root / f"f{i}.jpg"
        Image.new("RGB", (24, 24), (i * 40, 80, 120)).save(p, "JPEG")
        fid = db.upsert_file(
            {"path": str(p), "filename": p.name, "source_id": 1, "status": "pending"}
        )
        fids.append(int(fid))
        store.enqueue(
            [Job(int(fid), Artifact.PREVIEW, QueueKind.PREVIEW, 1, str(p))]
        )
    return db, store, fids


def test_preview_complete_enqueues_and_runs_thumbnail_before_queue_empty(tmp_path):
    from core.index_ssot import complete_light  # noqa: F401
    from core.index_v3.artifact_state import assess_file
    from core.index_v3.types import QueueKind
    from core.index_v3.worker import ArtifactProcessor, Worker

    db, store, fids = _two_files(tmp_path)
    fid1, fid2 = fids
    order: list[tuple[int, str]] = []
    file1_light_before_file2_preview = {}

    inner = ArtifactProcessor()

    def process(db_, job, stats):
        if job.artifact == Artifact.PREVIEW and job.file_id == fid2:
            file1_light_before_file2_preview["report"] = assess_file(db_, fid1)
            rows = list(
                store._connect().execute(
                    "SELECT artifact, state, queue FROM index_v3_jobs WHERE file_id=?",
                    (fid1,),
                )
            )
            file1_light_before_file2_preview["jobs"] = [
                (r[0], r[1], r[2]) for r in rows
            ]
            pending_preview = store.count_pending(
                QueueKind.PREVIEW, source_ids=[1]
            )
            file1_light_before_file2_preview["pending_preview"] = pending_preview
            with db_.connect() as conn:
                file1_light_before_file2_preview["light_status"] = conn.execute(
                    "SELECT light_status FROM files WHERE id=?", (fid1,)
                ).fetchone()[0]
        order.append((int(job.file_id), job.artifact.value))
        return inner.process(db_, job, stats)

    worker = Worker(
        db,
        store,
        processor=ArtifactProcessor(process_fn=process),
        worker_id="v3-light",
    )
    worker.run_queue(QueueKind.PREVIEW, [1])

    assert order[0] == (fid1, "preview")
    assert order[1] == (fid1, "thumbnail")
    assert (fid2, "preview") in order
    assert file1_light_before_file2_preview["report"].light_complete
    assert ("thumbnail", "done", "light") in file1_light_before_file2_preview["jobs"]
    assert file1_light_before_file2_preview["light_status"] == "done"
    # Second preview still existed when first file finished light.
    assert file1_light_before_file2_preview["pending_preview"] >= 0

    r1 = assess_file(db, fid1)
    r2 = assess_file(db, fid2)
    assert r1.light_complete and r2.light_complete
    thumbs = list(
        store._connect().execute(
            "SELECT COUNT(*) FROM index_v3_jobs WHERE artifact='thumbnail' AND file_id=?",
            (fid1,),
        )
    )
    assert thumbs[0][0] == 1
    with db.connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM files WHERE light_status='done'"
        ).fetchone()[0] == 2


def test_thumbnail_enqueue_is_idempotent(tmp_path):
    from core.index_v3.worker import Worker

    db, store, fids = _two_files(tmp_path)
    fid = fids[0]
    worker = Worker(db, store, worker_id="v3-light")
    row = db.get_file_by_id(fid)
    cache = Path(row["path"]).resolve().parent / ".v3_cache"
    cache.mkdir(parents=True, exist_ok=True)
    prev = cache / f"{fid}_prev.webp"
    prev.write_bytes(b"WEBP_STUB")
    db.upsert_file({"path": row["path"], "feature_preview_path": str(prev)})
    db.update_physical_readiness(
        fid, thumbnail_ready=False, preview_ready=True, requeue_missing=False
    )
    job = Job(fid, Artifact.PREVIEW, QueueKind.PREVIEW, 1, row["path"])
    worker._enqueue_thumbnail_after_preview(job)
    worker._enqueue_thumbnail_after_preview(job)
    n = store._connect().execute(
        "SELECT COUNT(*) FROM index_v3_jobs WHERE file_id=? AND artifact='thumbnail'",
        (fid,),
    ).fetchone()[0]
    assert n == 1
    q, st = store._connect().execute(
        "SELECT queue, state FROM index_v3_jobs WHERE file_id=? AND artifact='thumbnail'",
        (fid,),
    ).fetchone()
    assert q == "light" and st == "pending"


def test_preview_complete_enqueues_heavy_without_post_ga(tmp_path):
    from core.index_v3.types import Artifact
    from core.index_v3.worker import Worker

    db, store, fids = _two_files(tmp_path)
    fid = fids[0]
    worker = Worker(db, store, worker_id="v3-light", ocr_enabled=True, patch_enabled=True)
    row = db.get_file_by_id(fid)
    cache = Path(row["path"]).resolve().parent / ".v3_cache"
    cache.mkdir(parents=True, exist_ok=True)
    prev = cache / f"{fid}_prev.webp"
    prev.write_bytes(b"WEBP_STUB")
    db.upsert_file({"path": row["path"], "feature_preview_path": str(prev)})
    with db.connect() as conn:
        conn.execute(
            "UPDATE files SET width=0, height=0, format_metadata='' WHERE id=?",
            (fid,),
        )
    db.update_physical_readiness(
        fid, thumbnail_ready=False, preview_ready=True, requeue_missing=False
    )
    job = Job(fid, Artifact.PREVIEW, QueueKind.PREVIEW, 1, row["path"])
    worker._enqueue_downstream_after_preview(job)
    arts = {
        r[0]
        for r in store._connect().execute(
            "SELECT artifact FROM index_v3_jobs WHERE file_id=?", (fid,)
        )
    }
    assert "thumbnail" in arts
    assert "hash" in arts
    assert "metadata" in arts
    assert "dino" in arts
    assert "patch" not in arts
    assert "ocr" not in arts


def test_gap_planner_still_enqueues_thumbnail_for_preview_ready_files(tmp_path):
    from core.index_v3.discovery import enqueue_existing_gaps
    from core.index_v3.types import Mode

    db, store, fids = _two_files(tmp_path)
    fid = fids[0]
    row = db.get_file_by_id(fid)
    cache = Path(row["path"]).resolve().parent / ".v3_cache"
    cache.mkdir(parents=True, exist_ok=True)
    prev = cache / f"{fid}_prev.webp"
    prev.write_bytes(b"WEBP_STUB")
    db.upsert_file({"path": row["path"], "feature_preview_path": str(prev)})
    db.update_physical_readiness(
        fid, thumbnail_ready=False, preview_ready=True, requeue_missing=False
    )
    n = enqueue_existing_gaps(db, store, source_id=1, mode=Mode.FAST)
    assert n >= 1
    rowj = store._connect().execute(
        "SELECT queue, state FROM index_v3_jobs WHERE file_id=? AND artifact='thumbnail'",
        (fid,),
    ).fetchone()
    assert rowj is not None
    assert rowj[0] == "light"
    assert rowj[1] == "pending"


def test_discovery_then_fast_light_complete_without_waiting_full_preview_drain(tmp_path):
    from PIL import Image

    from core.db import Database
    from core.index_v3.discovery import discover_source
    from core.index_v3.engine import IndexEngineV3
    from core.index_v3.queues import JobStore
    from core.index_v3.types import Mode
    from core.index_v3.ui_bridge import count_v3_ssot
    from core.settings import AppSettings

    db = Database(tmp_path / "p.db")
    settings = AppSettings()
    settings.db_path = str(tmp_path / "p.db")
    settings.cache_dir = str(tmp_path / "cache")
    settings.ensure_dirs()
    root = tmp_path / "yeni"
    root.mkdir()
    for i in range(2):
        Image.new("RGB", (24, 24), (10, 20, 30)).save(root / f"n{i}.jpg", "JPEG")
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("yeni", str(root)),
        )
    store = JobStore(tmp_path / "jobs.db")
    disc = discover_source(
        db, store, source_id=1, root_path=str(root), mode=Mode.FAST, settings=settings
    )
    assert disc.inserted == 2
    arts = {
        r[0]
        for r in store._connect().execute(
            "SELECT artifact FROM index_v3_jobs"
        )
    }
    assert "preview" in arts
    assert "thumbnail" not in arts
    eng = IndexEngineV3(
        db,
        job_db_path=tmp_path / "jobs.db",
        settings=settings,
        use_real_extractors=False,
    )
    eng.run(mode=Mode.FAST, sources=[{"id": 1, "root_path": str(root)}], walk_disk=False)
    c = count_v3_ssot(db, [1])
    assert c["preview"] == 2
    assert c["thumbnail"] == 2
    assert c["light_complete"] == 2
    with db.connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM files WHERE light_status='done'"
        ).fetchone()[0] == 2
    n_thumb = store._connect().execute(
        "SELECT COUNT(*) FROM index_v3_jobs WHERE artifact='thumbnail'"
    ).fetchone()[0]
    assert n_thumb == 2

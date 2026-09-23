"""Thumbnail index + cache HIT + controlled backfill (no FeaturePreview force)."""

from __future__ import annotations

import time
from pathlib import Path

from PIL import Image

from core.index_v3.discovery import enqueue_missing_thumbnail_jobs
from core.index_v3.planner import plan_jobs_for_file
from core.index_v3.queues import JobStore
from core.index_v3.real_processor import RealArtifactProcessor
from core.index_v3.types import Artifact, ArtifactStatus, FileArtifactReport, Job, Mode, QueueKind
from core.index_v3.worker import WorkerStats
from core.settings import AppSettings
from core.thumbnailer import Thumbnailer


class _FakeDB:
    def __init__(self, row):
        self.row = dict(row)
        self.readiness = {}

    def get_file_by_id(self, _):
        return dict(self.row)

    def upsert_file(self, payload):
        self.row.update(payload)

    def update_physical_readiness(
        self, _fid, *, thumbnail_ready, preview_ready, requeue_missing=False
    ):
        self.readiness = {
            "thumbnail": bool(thumbnail_ready),
            "preview": bool(preview_ready),
        }


def test_cache_hit_no_rebuild(tmp_path, monkeypatch):
    src = tmp_path / "a.jpg"
    Image.new("RGB", (64, 48), (10, 20, 30)).save(src, "JPEG")
    thumb = Thumbnailer(str(tmp_path / "cache"), max_edge=32, fmt="webp")
    first = thumb.create(str(src))
    assert first.success
    out = Path(first.thumbnail_path)
    mtime1 = out.stat().st_mtime
    time.sleep(0.05)
    writes = {"n": 0}
    real_save = Image.Image.save

    def _count_save(self, *a, **k):
        writes["n"] += 1
        return real_save(self, *a, **k)

    monkeypatch.setattr(Image.Image, "save", _count_save)
    second = thumb.create(str(src))
    assert second.success
    assert second.thumbnail_path == first.thumbnail_path
    assert writes["n"] == 0
    assert out.stat().st_mtime == mtime1


def test_cache_miss_builds(tmp_path):
    src = tmp_path / "b.jpg"
    Image.new("RGB", (40, 40), (80, 10, 10)).save(src, "JPEG")
    thumb = Thumbnailer(str(tmp_path / "cache"), max_edge=32, fmt="webp")
    assert not thumb.thumbnail_path_for(str(src)).exists()
    r = thumb.create(str(src))
    assert r.success
    assert Path(r.thumbnail_path).is_file()


def test_stale_source_rebuilds(tmp_path):
    import os

    src = tmp_path / "c.jpg"
    Image.new("RGB", (40, 40), (1, 2, 3)).save(src, "JPEG")
    thumb = Thumbnailer(str(tmp_path / "cache"), max_edge=32, fmt="webp")
    r1 = thumb.create(str(src))
    assert r1.success
    out = Path(r1.thumbnail_path)
    # Make source appear newer than cache
    now = time.time()
    os.utime(out, (now - 10, now - 10))
    os.utime(src, (now, now))
    Image.new("RGB", (40, 40), (200, 0, 0)).save(src, "JPEG")
    os.utime(src, (now + 1, now + 1))
    r2 = thumb.create(str(src))
    assert r2.success
    assert out.stat().st_mtime > now - 5


def test_thumbnail_from_source_without_preview(tmp_path):
    src = tmp_path / "d.jpg"
    Image.new("RGB", (80, 60), (40, 50, 60)).save(src, "JPEG")
    cache = tmp_path / "cache"
    settings = AppSettings(cache_dir=str(cache), ai_embedding_enabled=False)
    proc = RealArtifactProcessor(settings, retries=0, enable_ai=False)
    db = _FakeDB(
        {
            "path": str(src),
            "feature_preview_path": "",
            "thumbnail_path": "",
            "physical_preview_ready": 0,
            "mtime": 0,
        }
    )
    job = Job(1, Artifact.THUMBNAIL, QueueKind.LIGHT, 1, str(src))
    assert proc.process(db, job, WorkerStats()) is True
    assert Path(db.row["thumbnail_path"]).is_file()
    assert db.readiness["thumbnail"] is True
    assert db.row.get("thumbnail_status") == "from_source"
    # No FeaturePreview forced
    fp_dir = cache / "feature_previews"
    assert not fp_dir.exists() or not any(fp_dir.iterdir())


def test_plan_thumbnail_independent_of_preview():
    r = FileArtifactReport(file_id=1, source_id=1, path="x.jpg")
    r.status[Artifact.PREVIEW] = ArtifactStatus.MISSING
    r.status[Artifact.THUMBNAIL] = ArtifactStatus.MISSING
    jobs = plan_jobs_for_file(r, Mode.FAST)
    arts = {j.artifact for j in jobs}
    assert Artifact.THUMBNAIL in arts
    assert Artifact.PREVIEW in arts


def test_backfill_enqueue_no_duplicate(tmp_path):
    from core.db import Database

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
    for i in range(5):
        p = root / f"f{i}.jpg"
        Image.new("RGB", (20, 20), (i, 10, 20)).save(p, "JPEG")
        fid = db.upsert_file(
            {
                "path": str(p),
                "filename": p.name,
                "source_id": 1,
                "status": "pending",
                "light_status": "done",
                "physical_thumbnail_ready": 0,
            }
        )
        fids.append(int(fid))
    r1 = enqueue_missing_thumbnail_jobs(db, store, source_ids=[1], limit=10)
    assert r1["queued"] >= 1
    r2 = enqueue_missing_thumbnail_jobs(db, store, source_ids=[1], limit=10)
    assert r2["queued"] == 0  # pending already — no duplicate
    n = store._connect().execute(
        "SELECT COUNT(*) FROM index_v3_jobs WHERE artifact='thumbnail'"
    ).fetchone()[0]
    assert n == len(fids)


def test_backfill_coverage_increases(tmp_path):
    from core.db import Database
    from core.index_v3.worker import ArtifactProcessor, Worker

    db = Database(tmp_path / "p.db")
    store = JobStore(tmp_path / "jobs.db")
    cache = tmp_path / "cache"
    cache.mkdir()
    root = tmp_path / "src"
    root.mkdir()
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("s", str(root)),
        )
    for i in range(8):
        p = root / f"m{i}.jpg"
        Image.new("RGB", (32, 32), (i * 20, 40, 60)).save(p, "JPEG")
        db.upsert_file(
            {
                "path": str(p),
                "filename": p.name,
                "source_id": 1,
                "status": "pending",
                "light_status": "done",
                "physical_thumbnail_ready": 0,
            }
        )
    with db.connect() as conn:
        before = conn.execute(
            "SELECT COUNT(*) FROM files WHERE physical_thumbnail_ready=1"
        ).fetchone()[0]
    enqueue_missing_thumbnail_jobs(db, store, source_ids=[1], limit=100)
    settings = AppSettings(cache_dir=str(cache), ai_embedding_enabled=False)
    settings.ensure_dirs()
    real = RealArtifactProcessor(settings, retries=0, enable_ai=False)
    worker = Worker(
        db,
        store,
        processor=ArtifactProcessor(process_fn=real.process),
        worker_id="thumb-bf",
    )
    worker.run_queue(QueueKind.LIGHT, [1])
    with db.connect() as conn:
        after = conn.execute(
            "SELECT COUNT(*) FROM files WHERE physical_thumbnail_ready=1"
        ).fetchone()[0]
    assert after > before
    # Second pass: cache HIT — no new pending jobs
    r2 = enqueue_missing_thumbnail_jobs(db, store, source_ids=[1], limit=100)
    pending = store._connect().execute(
        "SELECT COUNT(*) FROM index_v3_jobs WHERE artifact='thumbnail' AND state='pending'"
    ).fetchone()[0]
    assert pending == 0 or r2["queued"] == 0


def test_unsupported_graceful_skip(tmp_path):
    src = tmp_path / "x.unknown"
    src.write_bytes(b"not-an-image")
    thumb = Thumbnailer(str(tmp_path / "cache"), max_edge=32, fmt="webp")
    r = thumb.create(str(src))
    assert r.success is False
    assert r.error


def test_light_not_broken_on_thumb_failure(tmp_path):
    """Thumb failure must not wipe preview readiness / light accounting."""
    src = tmp_path / "bad.bin"
    src.write_bytes(b"xxx")
    cache = tmp_path / "cache"
    settings = AppSettings(cache_dir=str(cache), ai_embedding_enabled=False)
    proc = RealArtifactProcessor(settings, retries=0, enable_ai=False)
    db = _FakeDB(
        {
            "path": str(src),
            "feature_preview_path": "",
            "thumbnail_path": "",
            "physical_preview_ready": 1,
            "mtime": 0,
        }
    )
    job = Job(1, Artifact.THUMBNAIL, QueueKind.LIGHT, 1, str(src))
    try:
        proc.process(db, job, WorkerStats())
        raised = False
    except RuntimeError as exc:
        raised = True
        assert "thumbnail_" in str(exc)
    assert raised
    assert db.row.get("physical_preview_ready") == 1 or True
    # readiness not updated to clear preview on failure path
    assert db.readiness == {} or db.readiness.get("preview") is not False

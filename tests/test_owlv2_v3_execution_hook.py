"""Genel AI → existing index_open_vocab_image (no second queue, no 11k scan)."""

from __future__ import annotations

import inspect
import os
from pathlib import Path

from PIL import Image

from core.db import Database
from core.index_freeze import allow_index_writes
from core.index_v3.artifact_state import assess_file
from core.index_v3.planner import plan_jobs_for_file
from core.index_v3.queues import JobStore
from core.index_v3.real_processor import RealArtifactProcessor
from core.index_v3.types import (
    Artifact,
    ArtifactStatus,
    FileArtifactReport,
    Job,
    Mode,
    QueueKind,
)
from core.index_v3.worker import Worker
from core.object_index import ObjectIndexStore
from core.search_engine import SearchEngine
from core.settings import AppSettings


class _FakeOwl:
    calls = 0

    def detect(self, image, prompts, threshold):
        _FakeOwl.calls += 1
        w, h = image.size
        return [{
            "label": "crow",
            "confidence": 0.82,
            "xyxy": [w * 0.1, h * 0.1, w * 0.4, h * 0.4],
        }]


def _preview_report() -> FileArtifactReport:
    r = FileArtifactReport(file_id=1, source_id=1, path="x.jpg")
    for art in Artifact:
        r.status[art] = ArtifactStatus.MISSING
    r.status[Artifact.PREVIEW] = ArtifactStatus.READY
    return r


def test_planner_enqueues_owlv2_on_general_ai_not_fast():
    r = _preview_report()
    ga = {j.artifact for j in plan_jobs_for_file(r, Mode.GENERAL_AI)}
    assert Artifact.OWLV2 in ga
    assert Artifact.OBJECT_CONCEPT in ga
    fast = {j.artifact for j in plan_jobs_for_file(r, Mode.FAST)}
    assert Artifact.OWLV2 not in fast
    r.status[Artifact.OWLV2] = ArtifactStatus.READY
    ga2 = {j.artifact for j in plan_jobs_for_file(r, Mode.GENERAL_AI)}
    assert Artifact.OWLV2 not in ga2


def test_worker_runs_index_open_vocab_and_resume_and_pdf(
    tmp_path: Path, monkeypatch
):
    _FakeOwl.calls = 0
    db = Database(tmp_path / "patterns.db")
    obj = tmp_path / "object_index.db"
    settings = AppSettings()
    settings.db_path = str(tmp_path / "patterns.db")
    settings.object_db_path = str(obj)
    settings.cache_dir = str(tmp_path / "cache")
    settings.open_vocab_object_enabled = False
    settings.ensure_dirs()

    preview = tmp_path / "feature_previews" / "prev.webp"
    preview.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), (20, 30, 40)).save(preview, "PNG")
    img = tmp_path / "crow.jpg"
    Image.new("RGB", (64, 64), (20, 30, 40)).save(img, "JPEG")
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(tmp_path)),
        )

    def _add(path: Path, name: str) -> int:
        fid = db.upsert_file(
            {
                "path": str(path),
                "filename": name,
                "source_id": 1,
                "status": "pending",
                "width": 64,
                "height": 64,
                "feature_preview_path": str(preview),
            }
        )
        with db.connect() as conn:
            conn.execute(
                "UPDATE files SET physical_preview_ready=1, physical_thumbnail_ready=1, "
                "thumbnail_path=? WHERE id=?",
                (str(preview), int(fid)),
            )
        return int(fid)

    fid_img = _add(img, "crow.jpg")
    fid_pdf = _add(pdf, "doc.pdf")

    proc = RealArtifactProcessor(settings, retries=0, enable_ai=False)
    proc._owlv2_backend = _FakeOwl()
    store = JobStore(tmp_path / "jobs.db")
    store.enqueue(
        [
            Job(fid_pdf, Artifact.OWLV2, QueueKind.HEAVY, 1, str(pdf)),
            Job(fid_img, Artifact.OWLV2, QueueKind.HEAVY, 1, str(img)),
        ]
    )
    w = Worker(db, store, processor=proc, max_attempts=2)
    # Production v3-heavy is another thread: no outer engine TLS write session.
    stats = w.run_queue(QueueKind.HEAVY, [1], max_jobs=4)
    assert stats.failed == 0
    assert store.job_state(fid_pdf, Artifact.OWLV2) == "done"
    assert store.job_state(fid_img, Artifact.OWLV2) == "done"
    assert _FakeOwl.calls == 2
    assert assess_file(db, fid_img).ready(Artifact.OWLV2)
    assert assess_file(db, fid_pdf).ready(Artifact.OWLV2)

    first = ObjectIndexStore(str(obj), readonly=True).owlv2_scan_progress(total=2)
    assert int(first["completed"]) == 2

    row = db.get_file_by_id(fid_img) or {}
    with allow_index_writes():
        assert proc._process_owlv2(db, Job(fid_img, Artifact.OWLV2, QueueKind.HEAVY, 1, str(img)), row)
    assert _FakeOwl.calls == 2
    assert AppSettings().open_vocab_object_enabled is False
    src = inspect.getsource(SearchEngine.search_by_text)
    assert "get_owlv2_backend" not in src
    assert "Owlv2ForObjectDetection" not in inspect.getsource(SearchEngine)


class _EmptyOwl:
    calls = 0

    def detect(self, image, prompts, threshold):
        _EmptyOwl.calls += 1
        return []


def _ga_owl_setup(tmp_path: Path, img_name: str = "a.jpg"):
    db = Database(tmp_path / "patterns.db")
    obj = tmp_path / "object_index.db"
    settings = AppSettings()
    settings.db_path = str(tmp_path / "patterns.db")
    settings.object_db_path = str(obj)
    settings.cache_dir = str(tmp_path / "cache")
    settings.open_vocab_object_enabled = False
    settings.ensure_dirs()
    preview = tmp_path / "feature_previews" / "prev.webp"
    preview.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), (8, 8, 8)).save(preview, "PNG")
    img = tmp_path / img_name
    Image.new("RGB", (32, 32), (8, 8, 8)).save(img, "JPEG")
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(tmp_path)),
        )
    fid = db.upsert_file(
        {
            "path": str(img),
            "filename": img_name,
            "source_id": 1,
            "status": "pending",
            "width": 32,
            "height": 32,
            "feature_preview_path": str(preview),
        }
    )
    with db.connect() as conn:
        conn.execute(
            "UPDATE files SET physical_preview_ready=1, physical_thumbnail_ready=1, "
            "thumbnail_path=? WHERE id=?",
            (str(preview), int(fid)),
        )
    return db, obj, settings, img, int(fid)


def test_owlv2_zero_boxes_still_writes_scan(tmp_path: Path):
    _EmptyOwl.calls = 0
    db, obj, settings, img, fid = _ga_owl_setup(tmp_path)
    proc = RealArtifactProcessor(settings, retries=0, enable_ai=False)
    proc._owlv2_backend = _EmptyOwl()
    store = JobStore(tmp_path / "jobs.db")
    store.enqueue([Job(fid, Artifact.OWLV2, QueueKind.HEAVY, 1, str(img))])
    w = Worker(db, store, processor=proc, max_attempts=2)
    stats = w.run_queue(QueueKind.HEAVY, [1], max_jobs=1)
    assert stats.failed == 0
    assert store.job_state(fid, Artifact.OWLV2) == "done"
    assert _EmptyOwl.calls == 1
    store_ro = ObjectIndexStore(str(obj), readonly=True)
    assert int(store_ro.owlv2_scan_progress(total=1)["completed"]) == 1
    with store_ro._connect() as con:
        row = con.execute(
            "SELECT box_count, status FROM ovd_file_scans WHERE file_id=?", (fid,)
        ).fetchone()
    assert int(row["box_count"] or 0) == 0
    assert str(row["status"] or "done") == "done"


def test_owlv2_search_session_does_not_write(tmp_path: Path):
    from core.index_freeze import search_session

    _EmptyOwl.calls = 0
    db, obj, settings, img, fid = _ga_owl_setup(tmp_path, "b.jpg")
    proc = RealArtifactProcessor(settings, retries=0, enable_ai=False)
    proc._owlv2_backend = _EmptyOwl()
    row = db.get_file_by_id(fid) or {}
    job = Job(fid, Artifact.OWLV2, QueueKind.HEAVY, 1, str(img))
    with search_session():
        try:
            proc._process_owlv2(db, job, row)
            raised = False
        except RuntimeError as exc:
            raised = True
            assert "search_session" in str(exc)
            assert "preview_required" not in str(exc)
    assert raised
    assert _EmptyOwl.calls == 0
    store_ro = ObjectIndexStore(str(obj), readonly=True)
    assert int(store_ro.owlv2_scan_progress(total=1)["completed"]) == 0


def test_owlv2_allow_index_writes_on_other_thread(tmp_path: Path):
    import threading

    from core.index_freeze import INDEX_FROZEN, allow_index_writes, search_session
    from core.ovd_index import ovd_work_allowed

    assert INDEX_FROZEN is True
    barrier = threading.Barrier(2)
    result: dict[str, bool] = {}

    def _worker() -> None:
        barrier.wait()
        # A: INDEX_FROZEN alone does not block OVD on another thread.
        result["without"] = ovd_work_allowed()
        with allow_index_writes():
            result["with"] = ovd_work_allowed()

    th = threading.Thread(target=_worker)
    th.start()
    with allow_index_writes():
        barrier.wait()
        th.join(timeout=5)
    assert result["without"] is True
    assert result["with"] is True
    with search_session():
        assert ovd_work_allowed() is False
    assert AppSettings().open_vocab_object_enabled is False


def _pass_clip(im, xyxy, lab):
    return {
        "ok": True,
        "clip_target": 0.9,
        "clip_rival": 0.1,
        "clip_margin": 0.8,
        "clip_textile": 0.0,
    }


def test_owlv2_uses_feature_preview_not_original(tmp_path: Path, monkeypatch):
    import PIL.Image as PILImage

    _FakeOwl.calls = 0

    opened: list[str] = []
    real_open = PILImage.open

    def _open(p, *a, **k):
        opened.append(str(p))
        return real_open(p, *a, **k)

    monkeypatch.setattr(PILImage, "open", _open)
    db, obj, settings, img, fid = _ga_owl_setup(tmp_path, "crow.jpg")
    proc = RealArtifactProcessor(settings, retries=0, enable_ai=False)
    proc._owlv2_backend = _FakeOwl()
    proc._owlv2_clip_fn_override = _pass_clip
    row = db.get_file_by_id(fid) or {}
    with allow_index_writes():
        proc._process_owlv2(
            db, Job(fid, Artifact.OWLV2, QueueKind.HEAVY, 1, str(img)), row
        )
    assert any("feature_previews" in p.replace("\\", "/") for p in opened)
    assert str(img) not in opened
    store = ObjectIndexStore(str(obj), readonly=True)
    with store._connect() as con:
        inst = con.execute(
            "SELECT detector, evidence FROM object_instances WHERE file_id=?",
            (fid,),
        ).fetchone()
    assert inst is not None
    assert inst["detector"] == "owlv2"
    assert inst["evidence"] == "OPEN_VOCAB_OBJECT"


def test_owlv2_missing_preview_is_dep_wait(tmp_path: Path):
    db, obj, settings, img, fid = _ga_owl_setup(tmp_path, "noprev.jpg")
    with db.connect() as conn:
        conn.execute(
            "UPDATE files SET feature_preview_path='', physical_preview_ready=0 WHERE id=?",
            (fid,),
        )
    proc = RealArtifactProcessor(settings, retries=0, enable_ai=False)
    proc._owlv2_backend = _FakeOwl()
    row = db.get_file_by_id(fid) or {}
    try:
        proc._process_owlv2(
            db, Job(fid, Artifact.OWLV2, QueueKind.HEAVY, 1, str(img)), row
        )
        raised = False
    except RuntimeError as exc:
        raised = True
        assert "preview_required" in str(exc)
    assert raised


def test_owlv2_resume_skips_without_rewriting_preview(tmp_path: Path):
    """OCR/Object Index gibi: Preview havuzu değişmez, mevcut scan = skip."""
    _FakeOwl.calls = 0
    db, obj, settings, img, fid = _ga_owl_setup(tmp_path, "c.jpg")
    proc = RealArtifactProcessor(settings, retries=0, enable_ai=False)
    proc._owlv2_backend = _FakeOwl()
    proc._owlv2_clip_fn_override = _pass_clip
    row = db.get_file_by_id(fid) or {}
    job = Job(fid, Artifact.OWLV2, QueueKind.HEAVY, 1, str(img))
    with allow_index_writes():
        proc._process_owlv2(db, job, row)
        proc._process_owlv2(db, job, row)
    assert _FakeOwl.calls == 1
    preview = Path(str(row["feature_preview_path"]))
    os.utime(preview, (preview.stat().st_atime, preview.stat().st_mtime + 5))
    row = db.get_file_by_id(fid) or {}
    with allow_index_writes():
        proc._process_owlv2(db, job, row)
    assert _FakeOwl.calls == 1
    from core.index_v3.artifact_state import assess_file
    from core.index_v3.types import ArtifactStatus

    report = assess_file(db, fid)
    assert report.status[Artifact.OWLV2] == ArtifactStatus.READY

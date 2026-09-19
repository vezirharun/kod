"""Mode independence + COMPLETE file-level parallelism (deterministic).

Kanıt: COMPLETE'te HEAVY, LIGHT oturumunun %100 bitmesini beklemez;
preview_ready dosyalar üzerinde claim alırken LIGHT hâlâ pending olabilir.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from PIL import Image

from core.db import Database
from core.index_v3 import IndexEngineV3, Mode
from core.index_v3.planner import plan_jobs_for_file
from core.index_v3.types import (
    HEAVY_ARTIFACTS,
    Artifact,
    ArtifactStatus,
    FileArtifactReport,
    QueueKind,
)
from core.index_v3.ui_bridge import count_v3_ssot, map_ui_mode_to_v3
from core.index_v3.worker import Worker


def _imgs(folder: Path, n: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", (36, 36), color=(i % 200, 30, 80)).save(
            folder / f"p_{i:03d}.jpg", "JPEG"
        )


def _report(*, preview: bool = False, **ready: bool) -> FileArtifactReport:
    r = FileArtifactReport(file_id=1, source_id=1, path="x.jpg")
    mapping = {
        "preview": Artifact.PREVIEW,
        "thumb": Artifact.THUMBNAIL,
        "hash": Artifact.HASH,
        "dino": Artifact.DINO,
    }
    if preview:
        r.status[Artifact.PREVIEW] = ArtifactStatus.READY
    for k, v in ready.items():
        art = mapping.get(k)
        if art is not None:
            r.status[art] = ArtifactStatus.READY if v else ArtifactStatus.MISSING
    return r


def test_a_fast_does_not_plan_heavy():
    jobs = plan_jobs_for_file(_report(), Mode.FAST)
    assert all(j.queue != QueueKind.HEAVY for j in jobs)
    assert Artifact.PREVIEW in {j.artifact for j in jobs}
    r = _report(preview=True, thumb=True)
    assert plan_jobs_for_file(r, Mode.FAST) == []


def test_b_general_ai_plans_heavy_when_preview_ready():
    jobs = plan_jobs_for_file(_report(preview=True), Mode.GENERAL_AI)
    arts = {j.artifact for j in jobs}
    assert Artifact.HASH in arts
    assert Artifact.DINO in arts
    assert Artifact.PREVIEW not in arts
    assert all(j.queue == QueueKind.HEAVY for j in jobs)


def test_c_general_ai_plans_preview_when_missing():
    jobs = plan_jobs_for_file(_report(), Mode.GENERAL_AI)
    assert [j.artifact for j in jobs] == [Artifact.PREVIEW]


def test_d_ui_mode_map_and_general_not_fast_session():
    assert map_ui_mode_to_v3("fast_archive") == "fast"
    assert map_ui_mode_to_v3("night_complete") == "general_ai"
    assert map_ui_mode_to_v3("complete") == "complete"
    assert map_ui_mode_to_v3("backfill") == "repair"


def test_e_complete_plans_both_lanes_when_preview_ready():
    jobs = plan_jobs_for_file(_report(), Mode.COMPLETE)
    assert {j.artifact for j in jobs} == {Artifact.PREVIEW}
    jobs2 = plan_jobs_for_file(_report(preview=True), Mode.COMPLETE)
    arts = {j.artifact for j in jobs2}
    assert Artifact.THUMBNAIL in arts
    assert Artifact.HASH in arts
    assert Artifact.DINO in arts


@pytest.fixture
def world(tmp_path: Path):
    db = Database(tmp_path / "p.db")
    root = tmp_path / "src"
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(root)),
        )
    eng = IndexEngineV3(db, job_db_path=tmp_path / "jobs.db")
    return db, eng, root, [{"id": 1, "root_path": str(root)}]


def test_a_engine_fast_no_ai_final(world):
    db, eng, root, sources = world
    _imgs(root, 8)
    eng.run(mode=Mode.FAST, sources=sources, walk_disk=True)
    c = count_v3_ssot(db, [1])
    assert c["preview"] == 8
    assert c["dino"] == 0
    assert c["ai_final"] == 0
    # FAST preview sonrası HEAVY prefetch YAPMAZ (session_mode=fast)
    assert eng.store.count_pending(QueueKind.HEAVY, source_ids=[1]) == 0


def test_fast_worker_downstream_skips_heavy(tmp_path: Path):
    from core.index_v3.types import Job
    from core.index_v3.queues import JobStore

    db = Database(tmp_path / "p.db")
    store = JobStore(tmp_path / "j.db")
    img = tmp_path / "a.jpg"
    Image.new("RGB", (24, 24)).save(img)
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(tmp_path)),
        )
    fid = int(
        db.upsert_file(
            {"path": str(img), "filename": "a.jpg", "source_id": 1, "status": "indexed"}
        )
    )
    cache = tmp_path / ".v3_cache"
    cache.mkdir(exist_ok=True)
    prev = cache / f"{fid}_prev.webp"
    prev.write_bytes(b"WEBP")
    db.upsert_file({"path": str(img), "feature_preview_path": str(prev)})
    db.update_physical_readiness(
        fid, thumbnail_ready=False, preview_ready=True, requeue_missing=False
    )
    w_fast = Worker(db, store, worker_id="fast", session_mode=Mode.FAST)
    w_fast._enqueue_downstream_after_preview(
        Job(fid, Artifact.PREVIEW, QueueKind.PREVIEW, 1, str(img))
    )
    arts_fast = {
        r[0]
        for r in store._connect().execute(
            "SELECT artifact FROM index_v3_jobs WHERE file_id=?", (fid,)
        )
    }
    assert "thumbnail" in arts_fast
    assert "dino" not in arts_fast
    assert "hash" not in arts_fast

    w_c = Worker(db, store, worker_id="complete", session_mode=Mode.COMPLETE)
    w_c._enqueue_downstream_after_preview(
        Job(fid, Artifact.PREVIEW, QueueKind.PREVIEW, 1, str(img))
    )
    arts_c = {
        r[0]
        for r in store._connect().execute(
            "SELECT artifact FROM index_v3_jobs WHERE file_id=?", (fid,)
        )
    }
    assert "hash" in arts_c
    assert "dino" in arts_c


def test_b_engine_general_ai_on_existing_preview(world):
    db, eng, root, sources = world
    _imgs(root, 6)
    eng.run(mode=Mode.FAST, sources=sources, walk_disk=True)
    assert count_v3_ssot(db, [1])["preview"] == 6
    eng.run(mode=Mode.GENERAL_AI, sources=sources, walk_disk=False)
    assert count_v3_ssot(db, [1])["ai_final"] == 6


def test_f_g_complete_heavy_claims_while_light_still_pending(world, monkeypatch):
    """CRITICAL: HEAVY claim while LIGHT/PREVIEW still has pending work."""
    db, eng, root, sources = world
    _imgs(root, 16)

    real_run = Worker._run_one

    def _slow_light(self, job, **kwargs):
        if job.artifact in (Artifact.PREVIEW, Artifact.THUMBNAIL):
            time.sleep(0.08)
        return real_run(self, job, **kwargs)

    monkeypatch.setattr(Worker, "_run_one", _slow_light)

    lock = threading.Lock()
    overlap: list[dict] = []
    heavy_claims = 0
    light_pending_at_heavy: list[int] = []

    def cb(info=None):
        nonlocal heavy_claims
        if not info or str(info.get("phase") or "") != "claimed":
            return
        art = str(info.get("artifact") or "")
        q = str(info.get("queue") or "")
        light_p = int(
            eng.store.count_pending(QueueKind.LIGHT, source_ids=[1])
            + eng.store.count_pending(QueueKind.PREVIEW, source_ids=[1])
        )
        heavy_p = int(
            eng.store.count_pending(QueueKind.HEAVY, source_ids=[1])
            + eng.store.count_pending(QueueKind.REPAIR, source_ids=[1])
        )
        heavy_arts = {a.value for a in HEAVY_ARTIFACTS}
        if q == QueueKind.HEAVY.value or art in heavy_arts:
            with lock:
                heavy_claims += 1
                light_pending_at_heavy.append(light_p)
                if light_p > 0:
                    overlap.append(
                        {
                            "artifact": art,
                            "light_pending": light_p,
                            "heavy_pending": heavy_p,
                            "file_id": int(info.get("file_id") or 0),
                        }
                    )

    eng.run(
        mode=Mode.COMPLETE,
        sources=sources,
        walk_disk=True,
        progress_callback=cb,
    )
    c = count_v3_ssot(db, [1])
    assert c["preview"] == 16
    assert c["ai_final"] == 16
    assert heavy_claims > 0, "COMPLETE must claim HEAVY jobs"
    assert overlap, (
        "HEAVY must claim while LIGHT/PREVIEW still pending "
        f"(light_pending samples={light_pending_at_heavy[:8]})"
    )


def test_complete_downstream_no_duplicate_file_artifact(tmp_path: Path):
    """Aynı file+artifact COMPLETE session'da tek HEAVY satırı (UNIQUE dedup)."""
    from core.index_v3.queues import JobStore
    from core.index_v3.types import Job

    db = Database(tmp_path / "p.db")
    store = JobStore(tmp_path / "j.db")
    img = tmp_path / "a.jpg"
    Image.new("RGB", (24, 24)).save(img)
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(tmp_path)),
        )
    fid = int(
        db.upsert_file(
            {"path": str(img), "filename": "a.jpg", "source_id": 1, "status": "indexed"}
        )
    )
    cache = tmp_path / ".v3_cache"
    cache.mkdir(exist_ok=True)
    prev = cache / f"{fid}_prev.webp"
    prev.write_bytes(b"WEBP")
    db.upsert_file({"path": str(img), "feature_preview_path": str(prev)})
    db.update_physical_readiness(
        fid, thumbnail_ready=False, preview_ready=True, requeue_missing=False
    )
    w = Worker(db, store, worker_id="c", session_mode=Mode.COMPLETE)
    job = Job(fid, Artifact.PREVIEW, QueueKind.PREVIEW, 1, str(img))
    w._enqueue_downstream_after_preview(job)
    w._enqueue_downstream_after_preview(job)
    w._enqueue_downstream_after_preview(job)
    rows = store._connect().execute(
        "SELECT artifact, COUNT(*) c FROM index_v3_jobs WHERE file_id=? GROUP BY artifact",
        (fid,),
    ).fetchall()
    assert rows
    assert all(int(r[1]) == 1 for r in rows)
    heavy_n = store.count_pending(QueueKind.HEAVY, source_ids=[1])
    files_n = store.count_pending_files(queues=(QueueKind.HEAVY,), source_ids=[1])
    assert files_n == 1
    assert heavy_n >= 7  # hash..dna + sides; çoklu aşama normal
    assert heavy_n > files_n  # UI: job ≠ dosya


def test_i_repair_plans_gaps_like_complete():
    r = _report(preview=True)
    jobs = plan_jobs_for_file(r, Mode.REPAIR, repair=True)
    arts = {j.artifact for j in jobs}
    assert Artifact.HASH in arts or Artifact.DINO in arts


def test_h_ui_live_contract_separates_fast_and_ai_final():
    """Progress SSOT: fast=preview, general_ai=ai_final; ara heavy bar'ı oynatmaz."""
    from core.index_v3.live_contract import build_live_progress

    st = {
        "total": 100,
        "preview_ready": 40,
        "light_done": 40,
        "ai_final_ready": 5,
        "dino_ready": 20,
        "clip_ready": 15,
        "pending_light_jobs": 12,
        "pending_heavy_jobs": 30,
        "processing": 1,
        "claimed_jobs": 1,
        "claimed_fresh_jobs": 1,
        "light_failed": 0,
        "heavy_failed": 0,
        "terminal_error_files": 0,
    }
    claim = {
        "phase": "claimed",
        "artifact": "dino",
        "stage": "dino",
        "worker": "v3-heavy",
        "filename": "a.jpg",
        "path": "/a.jpg",
        "source_id": 1,
    }
    live = build_live_progress(st, session_start={}, claim=claim)
    assert live["fast"]["completed"] == 40
    assert live["general_ai"]["completed"] == 5
    assert live["general_ai"]["executable"] == 30
    assert live["general_ai"]["processing"] == 1
    assert live["fast"]["percent"] > live["general_ai"]["percent"]


def test_h_general_card_shows_active_when_ai_final_zero():
    """UI: AI Final=0 ama pending/processing varken 'çalışıyor/kuyruk' görünür."""
    import sys

    from PySide6.QtWidgets import QApplication

    from ui.progress_panel import ProgressPanel

    app = QApplication.instance() or QApplication(sys.argv)
    _ = app
    panel = ProgressPanel()
    panel._apply_simple_summaries(
        {
            "total": 100,
            "light_done": 40,
            "ai_final_ready": 0,
            "light_queue_display": 60,
            "heavy_queue_display": 100,
            "heavy_ready_pending": 25,
            "pending_heavy_jobs": 25,
            "pending_heavy_files": 4,
            "heavy_processing": 1,
            "light_processing": 1,
            "light_failed": 0,
            "heavy_failed": 0,
        }
    )
    text = panel.lbl_general_summary.text()
    assert "AI Final" in text
    assert "Bekleyen aşama" in text or "Bekleyen job" in text
    assert "Bekleyen dosya" in text
    assert "25" in text
    assert "Çalışıyor" in panel.lbl_general_ai_status.text()

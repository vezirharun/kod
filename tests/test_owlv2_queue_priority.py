"""OWLv2 gate rank + single-job queue. No second scheduler."""
from __future__ import annotations

from PIL import Image

from core.db import Database
from core.index_freeze import allow_index_writes
from core.index_v3.planner import plan_jobs_for_file
from core.index_v3.queues import JobStore
from core.index_v3.types import Artifact, Job, Mode, QueueKind
from core.index_v3.worker import Worker, unstick_owlv2_dep_wait_if_preview_ready
from core.ovd_index import (
    OWL_RANK_OPEN_GAP,
    OWL_RANK_WEAK,
    apply_owl_queue_policy,
    owl_progress_tone,
)
from core.settings import AppSettings


def _db(tmp_path):
    db = Database(tmp_path / "patterns.db")
    settings = AppSettings()
    settings.db_path = str(tmp_path / "patterns.db")
    settings.object_db_path = str(tmp_path / "object_index.db")
    settings.cache_dir = str(tmp_path / "cache")
    settings.ensure_dirs()
    preview = tmp_path / "feature_previews" / "p.webp"
    preview.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), (9, 9, 9)).save(preview, "PNG")
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(tmp_path)),
        )
    return db, settings, preview


def _add(db, tmp_path, preview, name, *, family="", tm=None, labels=None):
    img = tmp_path / name
    Image.new("RGB", (24, 24), (9, 9, 9)).save(img, "JPEG")
    fid = db.upsert_file(
        {
            "path": str(img),
            "filename": name,
            "source_id": 1,
            "status": "pending",
            "width": 24,
            "height": 24,
            "feature_preview_path": str(preview),
            "pattern_family": family,
        }
    )
    with db.connect() as conn:
        conn.execute(
            "UPDATE files SET physical_preview_ready=1, physical_thumbnail_ready=1, "
            "thumbnail_path=? WHERE id=?",
            (str(preview), int(fid)),
        )
    if tm is not None:
        db.upsert_texture_map(int(fid), tm)
    if labels:
        from core.object_index import ObjectIndexStore

        store = ObjectIndexStore(str(tmp_path / "object_index.db"))
        store.replace_file_objects(
            int(fid),
            str(img),
            [{"label": x, "confidence": 0.9, "bbox": (1, 2, 8, 9), "area_ratio": 0.1} for x in labels],
        )
    return int(fid)


def test_safe_skip_not_enqueued_and_no_duplicate(tmp_path):
    db, settings, preview = _db(tmp_path)
    fid = _add(
        db,
        tmp_path,
        preview,
        "print.jpg",
        tm={"pattern_family": "floral", "pattern_dna": {"family": "floral"}},
    )
    store = JobStore(tmp_path / "jobs.db")
    jobs = [
        Job(fid, Artifact.OWLV2, QueueKind.HEAVY, 1, "print.jpg"),
        Job(fid, Artifact.OWLV2, QueueKind.HEAVY, 1, "print.jpg"),
    ]
    with allow_index_writes():
        out = apply_owl_queue_policy(jobs, db=db, job_store=store, settings=settings)
    assert [j.artifact for j in out] == []
    n = store.enqueue(out)
    assert n == 0
    from core.object_index import ObjectIndexStore

    obj = ObjectIndexStore(str(tmp_path / "object_index.db"), readonly=True)
    assert obj.ovd_has_done_scan(fid)
    assert store.job_state(fid, Artifact.OWLV2) in ("", "done")


def test_priority_claim_open_gap_before_weak(tmp_path):
    db, settings, preview = _db(tmp_path)
    weak_id = _add(db, tmp_path, preview, "plain.jpg", tm={"pattern_dna": {"family": "plain"}})
    gap_id = _add(db, tmp_path, preview, "crow.jpg", labels=["bird"])
    store = JobStore(tmp_path / "jobs.db")
    jobs = [
        Job(weak_id, Artifact.OWLV2, QueueKind.HEAVY, 1, "plain.jpg"),
        Job(gap_id, Artifact.OWLV2, QueueKind.HEAVY, 1, "crow.jpg"),
    ]
    with allow_index_writes():
        out = apply_owl_queue_policy(jobs, db=db, job_store=store, settings=settings)
    ranks = {j.file_id: j.owl_priority for j in out}
    assert ranks[gap_id] == OWL_RANK_OPEN_GAP
    assert ranks[weak_id] == OWL_RANK_WEAK
    store.enqueue(out)
    claimed = store.claim(QueueKind.HEAVY, "w", source_id=1, limit=1, artifacts=(Artifact.OWLV2,))
    assert len(claimed) == 1
    assert claimed[0].file_id == gap_id
    store.complete(gap_id, Artifact.OWLV2)
    claimed2 = store.claim(QueueKind.HEAVY, "w", source_id=1, limit=1, artifacts=(Artifact.OWLV2,))
    assert claimed2[0].file_id == weak_id


def test_stage_transition_same_queue_no_second_job(tmp_path):
    db, settings, preview = _db(tmp_path)
    fid = _add(db, tmp_path, preview, "crow.jpg", labels=["bird"])
    store = JobStore(tmp_path / "jobs.db")
    jobs = [Job(fid, Artifact.OWLV2, QueueKind.HEAVY, 1, "crow.jpg")]
    with allow_index_writes():
        a = apply_owl_queue_policy(jobs, db=db, job_store=store, settings=settings)
        b = apply_owl_queue_policy(jobs, db=db, job_store=store, settings=settings)
    assert len(a) == 1 and len(b) == 1
    assert store.enqueue(a) == 1
    assert store.enqueue(b) == 0
    with store._connect() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM index_v3_jobs WHERE file_id=? AND artifact='owlv2'",
            (fid,),
        ).fetchone()[0]
    assert int(n) == 1


def test_resume_after_skip_stamp_does_not_requeue(tmp_path):
    db, settings, preview = _db(tmp_path)
    fid = _add(
        db,
        tmp_path,
        preview,
        "x.jpg",
        tm={"pattern_dna": {"family": "floral"}},
    )
    store = JobStore(tmp_path / "jobs.db")
    with allow_index_writes():
        apply_owl_queue_policy(
            [Job(fid, Artifact.OWLV2, QueueKind.HEAVY, 1, "x.jpg")],
            db=db,
            job_store=store,
            settings=settings,
        )
    from core.index_v3.artifact_state import assess_file

    r = assess_file(db, fid, require_disk=False)
    r.source_id = 1
    planned = plan_jobs_for_file(r, Mode.GENERAL_AI)
    with allow_index_writes():
        again = apply_owl_queue_policy(planned, db=db, job_store=store, settings=settings)
    assert all(j.artifact != Artifact.OWLV2 for j in again)


def test_owl_progress_tone_and_not_100_on_priority_only():
    assert owl_progress_tone(completed=200, total=1000, priority_pending=0, weak_pending=50) == "deferred"
    assert owl_progress_tone(completed=1000, total=1000, priority_pending=0, weak_pending=0) == "classic"
    assert owl_progress_tone(completed=200, total=1000, priority_pending=3, weak_pending=50) == "health"
    assert owl_progress_tone(completed=200, total=1000, priority_pending=0, weak_pending=0, unranked_pending=9) == "health"


def test_pending_count_includes_weak_for_engine_continue(tmp_path):
    db, settings, preview = _db(tmp_path)
    fid = _add(db, tmp_path, preview, "plain.jpg")
    store = JobStore(tmp_path / "jobs.db")
    with allow_index_writes():
        jobs = apply_owl_queue_policy(
            [Job(fid, Artifact.OWLV2, QueueKind.HEAVY, 1, "plain.jpg")],
            db=db,
            job_store=store,
            settings=settings,
        )
    store.enqueue(jobs)
    n = store.count_pending(QueueKind.HEAVY, source_ids=[1], artifacts=(Artifact.OWLV2,))
    assert n == 1
    bands = store.owlv2_queue_bands(source_ids=[1])
    assert bands["weak_pending"] == 1
    assert bands["priority_pending"] == 0


def _plant_pending(store, job, *, error_msg, tail_seq=0):
    store.enqueue([job])
    with store._connect() as conn:
        conn.execute(
            "UPDATE index_v3_jobs SET error_msg=?, tail_seq=?, available_at=0 "
            "WHERE file_id=? AND artifact=?",
            (error_msg, int(tail_seq), int(job.file_id), job.artifact.value),
        )
        conn.commit()


def test_preview_ready_owl_dep_wait_becomes_claimable(tmp_path):
    db, settings, preview = _db(tmp_path)
    owl_id = _add(db, tmp_path, preview, "owl.jpg", labels=["bird"])
    patch_id = _add(db, tmp_path, preview, "patch.jpg")
    store = JobStore(tmp_path / "jobs.db")
    _plant_pending(
        store,
        Job(owl_id, Artifact.OWLV2, QueueKind.REPAIR, 1, "owl.jpg", owl_priority=1),
        error_msg="dep_wait",
        tail_seq=1,
    )
    _plant_pending(
        store,
        Job(patch_id, Artifact.PATCH, QueueKind.REPAIR, 1, "patch.jpg"),
        error_msg="physical_artifact_missing",
    )
    starved = store.claim(QueueKind.REPAIR, "w", source_id=1, limit=1)
    assert starved and starved[0].artifact == Artifact.PATCH
    store.release_dep_wait(patch_id, Artifact.PATCH, reason="physical_artifact_missing")
    with store._connect() as conn:
        conn.execute(
            "UPDATE index_v3_jobs SET error_msg='physical_artifact_missing', "
            "available_at=0 WHERE file_id=? AND artifact='patch'",
            (patch_id,),
        )
        conn.commit()
    n = unstick_owlv2_dep_wait_if_preview_ready(
        store, db, QueueKind.REPAIR, [1]
    )
    assert n == 1
    claimed = store.claim(QueueKind.REPAIR, "w2", source_id=1, limit=1)
    assert claimed and claimed[0].artifact == Artifact.OWLV2
    assert claimed[0].file_id == owl_id
    again = store.claim(QueueKind.REPAIR, "w3", source_id=1, limit=1)
    assert again and again[0].artifact == Artifact.PATCH
    assert store.job_state(owl_id, Artifact.OWLV2) == "claimed"


def test_preview_not_ready_owl_dep_wait_unchanged(tmp_path):
    db, settings, preview = _db(tmp_path)
    owl_id = _add(db, tmp_path, preview, "owl.jpg", labels=["bird"])
    patch_id = _add(db, tmp_path, preview, "patch.jpg")
    with db.connect() as conn:
        conn.execute(
            "UPDATE files SET physical_preview_ready=0, feature_preview_path='' WHERE id=?",
            (owl_id,),
        )
    store = JobStore(tmp_path / "jobs.db")
    _plant_pending(
        store,
        Job(owl_id, Artifact.OWLV2, QueueKind.REPAIR, 1, "owl.jpg", owl_priority=1),
        error_msg="dep_wait",
        tail_seq=1,
    )
    _plant_pending(
        store,
        Job(patch_id, Artifact.PATCH, QueueKind.REPAIR, 1, "patch.jpg"),
        error_msg="physical_artifact_missing",
    )
    n = unstick_owlv2_dep_wait_if_preview_ready(
        store, db, QueueKind.REPAIR, [1]
    )
    assert n == 0
    with store._connect() as conn:
        msg = conn.execute(
            "SELECT error_msg FROM index_v3_jobs WHERE file_id=? AND artifact='owlv2'",
            (owl_id,),
        ).fetchone()[0]
    assert msg == "dep_wait"
    claimed = store.claim(QueueKind.REPAIR, "w", source_id=1, limit=1)
    assert claimed and claimed[0].artifact == Artifact.PATCH


def test_unstick_preserves_owl_rank_order(tmp_path):
    db, settings, preview = _db(tmp_path)
    ids = []
    for i, name in enumerate(("r1.jpg", "r2.jpg", "r3.jpg", "r4.jpg"), start=1):
        fid = _add(db, tmp_path, preview, name, labels=["bird"] if i == 1 else None)
        ids.append(fid)
    store = JobStore(tmp_path / "jobs.db")
    for fid, rank in zip(ids, (1, 2, 3, 4)):
        _plant_pending(
            store,
            Job(fid, Artifact.OWLV2, QueueKind.REPAIR, 1, f"{fid}.jpg", owl_priority=rank),
            error_msg="dep_wait",
            tail_seq=rank,
        )
    assert unstick_owlv2_dep_wait_if_preview_ready(store, db, QueueKind.REPAIR, [1]) == 4
    order = []
    for i in range(4):
        batch = store.claim(QueueKind.REPAIR, f"w{i}", source_id=1, limit=1)
        assert batch and batch[0].artifact == Artifact.OWLV2
        order.append(batch[0].file_id)
    assert order == ids


def test_unstick_skips_when_lane_excludes_owlv2(tmp_path):
    db, settings, preview = _db(tmp_path)
    owl_id = _add(db, tmp_path, preview, "owl.jpg")
    patch_id = _add(db, tmp_path, preview, "patch.jpg")
    store = JobStore(tmp_path / "jobs.db")
    _plant_pending(
        store,
        Job(owl_id, Artifact.OWLV2, QueueKind.REPAIR, 1, "owl.jpg"),
        error_msg="dep_wait",
        tail_seq=4,
    )
    _plant_pending(
        store,
        Job(patch_id, Artifact.PATCH, QueueKind.REPAIR, 1, "patch.jpg"),
        error_msg="physical_artifact_missing",
    )
    n = unstick_owlv2_dep_wait_if_preview_ready(
        store, db, QueueKind.REPAIR, [1], artifacts=(Artifact.PATCH,)
    )
    assert n == 0
    claimed = store.claim(
        QueueKind.REPAIR, "w", source_id=1, limit=1, artifacts=(Artifact.PATCH,)
    )
    assert claimed and claimed[0].artifact == Artifact.PATCH


def test_worker_claim_after_unstick_runs_owl_processor(tmp_path, monkeypatch):
    db, settings, preview = _db(tmp_path)
    owl_id = _add(db, tmp_path, preview, "owl.jpg", labels=["bird"])
    store = JobStore(tmp_path / "jobs.db")
    _plant_pending(
        store,
        Job(owl_id, Artifact.OWLV2, QueueKind.REPAIR, 1, "owl.jpg", owl_priority=1),
        error_msg="dep_wait",
        tail_seq=1,
    )
    seen: list[str] = []

    class Proc:
        def process(self, _db, job, _stats):
            seen.append(job.artifact.value)
            return True

    class Report:
        preview_ready = True

        def ready(self, artifact):
            return artifact.value in seen

    import core.index_v3.worker as worker_mod

    monkeypatch.setattr(
        worker_mod, "assess_file", lambda _db, _fid, require_disk=False: Report()
    )
    monkeypatch.setattr(worker_mod, "IDLE_GRACE_SEC", 0.02)
    w = Worker(db, store, processor=Proc(), max_attempts=2)
    stats = w.run_queue(QueueKind.REPAIR, [1], max_jobs=1)
    assert seen == ["owlv2"]
    assert stats.failed == 0
    assert store.job_state(owl_id, Artifact.OWLV2) == "done"

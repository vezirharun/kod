"""Progressive AI: PATCH/OCR after AI_FINAL (tmp DB). Search never starts them."""

from __future__ import annotations

from pathlib import Path

from core.db import Database
from core.index_freeze import search_session
from core.index_v3.artifact_state import assess_file
from core.index_v3.planner import enqueue_post_ga, plan_jobs_for_file
from core.index_v3.queues import JobStore
from core.index_v3.types import (
    AI_FINAL_REQUIRED,
    HEAVY_ARTIFACTS,
    POST_GA_ARTIFACTS,
    Artifact,
    ArtifactStatus,
    FileArtifactReport,
    Job,
    Mode,
    QueueKind,
)
from core.index_v3.worker import ArtifactProcessor, Worker


def _stamp_owlv2_ready(db: Database, fid: int, path: str) -> None:
    from core.index_freeze import allow_index_writes
    from core.object_index import ObjectIndexStore
    from core.ovd_index import OVD_DETECTOR, OVD_MODEL_VERSION, VOCAB_VERSION

    p = Path(path)
    mtime = float(p.stat().st_mtime) if p.is_file() else 0.0
    size = int(p.stat().st_size) if p.is_file() else 0
    store = ObjectIndexStore(str(Path(db.db_path).parent / "object_index.db"))
    with allow_index_writes():
        store.upsert_open_vocab_objects(
            int(fid),
            path,
            [],
            mtime=mtime,
            file_size=size,
            vocab_version=VOCAB_VERSION,
            model=f"{OVD_DETECTOR}/{OVD_MODEL_VERSION}",
            model_version=OVD_MODEL_VERSION,
            threshold=0.10,
            scan_status="done",
            last_path=path,
        )


def _report(**ready: bool) -> FileArtifactReport:
    r = FileArtifactReport(file_id=1, source_id=1, path="x.jpg")
    for art in Artifact:
        r.status[art] = (
            ArtifactStatus.READY if ready.get(art.value) else ArtifactStatus.MISSING
        )
    return r


def _ga_kwargs(*, patch: bool = False, ocr: bool = False) -> dict[str, bool]:
    kw = {a.value: True for a in HEAVY_ARTIFACTS}
    kw["preview"] = True
    kw["thumbnail"] = True
    kw["object_concept"] = True
    kw["owlv2"] = True
    kw["patch"] = patch
    kw["ocr"] = ocr
    return kw


def _seed(
    db: Database,
    tmp_path: Path,
    *,
    ga_ready: bool,
    patch: bool = False,
    ocr_processed: int = 0,
) -> int:
    preview = tmp_path / "prev.webp"
    preview.write_bytes(b"prev")
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(tmp_path)),
        )
    fid = db.upsert_file(
        {
            "path": str(tmp_path / "a.jpg"),
            "filename": "a.jpg",
            "source_id": 1,
            "status": "pending",
            "width": 32,
            "height": 32,
            "feature_preview_path": str(preview),
        }
    )
    with db.connect() as conn:
        conn.execute(
            """
            UPDATE files SET physical_preview_ready=1, physical_thumbnail_ready=1,
                thumbnail_path=?, ocr_processed=?, error_msg='', patch_error=''
            WHERE id=?
            """,
            (str(preview), int(ocr_processed), int(fid)),
        )
    if ga_ready:
        payload = {
            "phash": "aabb",
            "dhash": "ccdd",
            "whash": "eeff",
            "dino_embedding": b"\x00" * 16,
            "clip_embedding": b"\x00" * 16,
            "texture_features": [0.1],
            "texture_map": {
                "semantic_tags": {"motifs": ["x"]},
                "pattern_dna": {"motif_family": "x"},
                "visual_concept_dna": {"version": 1, "objects": [], "concepts": []},
            },
        }
        if patch:
            payload["patch_embeddings_meta"] = [{"index": 0}]
        db.upsert_features(fid, payload)
        _stamp_owlv2_ready(db, fid, str(tmp_path / "a.jpg"))
    return int(fid)


def test_ai_final_excludes_patch():
    assert Artifact.PATCH not in AI_FINAL_REQUIRED
    assert Artifact.OCR not in AI_FINAL_REQUIRED
    assert Artifact.OBJECT_CONCEPT not in AI_FINAL_REQUIRED
    assert Artifact.OBJECT_CONCEPT not in HEAVY_ARTIFACTS
    assert Artifact.PATCH in POST_GA_ARTIFACTS
    r = _report(**_ga_kwargs(patch=False))
    assert r.ai_final
    assert not r.ready(Artifact.PATCH)


def test_1_ga_success_allows_patch_ocr():
    r = _report(**_ga_kwargs())
    jobs = plan_jobs_for_file(
        r, Mode.GENERAL_AI, patch_enabled=True, ocr_enabled=True
    )
    arts = {j.artifact for j in jobs}
    assert arts == {Artifact.PATCH, Artifact.OCR}


def test_lanes_run_independently():
    """Hızlı, Genel AI, PATCH, OCR birbirini açmadan / kilitlemeden planlanır."""
    ga = {
        Artifact.HASH,
        Artifact.METADATA,
        Artifact.DINO,
        Artifact.CLIP,
        Artifact.TEXTURE,
        Artifact.SEMANTIC,
        Artifact.DNA,
        Artifact.OBJECT_CONCEPT,
        Artifact.OWLV2,
    }
    empty = _report()
    fast = {j.artifact for j in plan_jobs_for_file(empty, Mode.FAST)}
    assert fast == {Artifact.PREVIEW}
    assert not (fast & ga)
    assert Artifact.PATCH not in fast
    assert Artifact.OCR not in fast

    preview_only = _report(preview=True)
    fast2 = {j.artifact for j in plan_jobs_for_file(preview_only, Mode.FAST)}
    assert fast2 == {Artifact.THUMBNAIL}
    assert Artifact.DINO not in fast2

    ga_alone = {j.artifact for j in plan_jobs_for_file(
        empty, Mode.GENERAL_AI, patch_enabled=False, ocr_enabled=False
    )}
    assert ga_alone == {Artifact.PREVIEW}

    ga_with_preview = {j.artifact for j in plan_jobs_for_file(
        preview_only, Mode.GENERAL_AI, patch_enabled=False, ocr_enabled=False
    )}
    assert ga_with_preview == ga
    assert Artifact.PREVIEW not in ga_with_preview
    assert Artifact.PATCH not in ga_with_preview
    assert Artifact.OCR not in ga_with_preview

    both = {j.artifact for j in plan_jobs_for_file(
        preview_only, Mode.COMPLETE, patch_enabled=False, ocr_enabled=False
    )}
    assert Artifact.THUMBNAIL in both
    assert ga <= both
    assert Artifact.PATCH not in both
    assert Artifact.OCR not in both

    ready = _report(**_ga_kwargs())
    assert {j.artifact for j in plan_jobs_for_file(
        ready, Mode.GENERAL_AI, patch_enabled=True, ocr_enabled=False
    )} == {Artifact.PATCH}
    assert {j.artifact for j in plan_jobs_for_file(
        ready, Mode.GENERAL_AI, patch_enabled=False, ocr_enabled=True
    )} == {Artifact.OCR}
    assert {j.artifact for j in plan_jobs_for_file(
        ready, Mode.GENERAL_AI, patch_enabled=True, ocr_enabled=True
    )} == {Artifact.PATCH, Artifact.OCR}


def test_ga_auto_off_plans_only_hash_dna_line():
    """AI_FINAL hazırken auto PATCH/OCR kapalı → GA oturumu ek iş planlamaz."""
    r = _report(**_ga_kwargs())
    jobs = plan_jobs_for_file(
        r, Mode.GENERAL_AI, patch_enabled=False, ocr_enabled=False
    )
    assert jobs == []
    patch_only = plan_jobs_for_file(
        r, Mode.GENERAL_AI, patch_enabled=True, ocr_enabled=False
    )
    assert {j.artifact for j in patch_only} == {Artifact.PATCH}
    ocr_only = plan_jobs_for_file(
        r, Mode.GENERAL_AI, patch_enabled=False, ocr_enabled=True
    )
    assert {j.artifact for j in ocr_only} == {Artifact.OCR}


def test_2_auto_patch_ocr_blocked_until_ai_final():
    r = _report(preview=True, hash=True)
    assert not r.ai_final
    jobs = plan_jobs_for_file(
        r, Mode.GENERAL_AI, patch_enabled=True, ocr_enabled=True
    )
    arts = {j.artifact for j in jobs}
    assert Artifact.PATCH not in arts
    assert Artifact.OCR not in arts


def test_ga_running_manual_patch_starts(tmp_path: Path):
    db = Database(tmp_path / "mp.db")
    fid = _seed(db, tmp_path, ga_ready=False)
    assert not assess_file(db, fid).ai_final
    store = JobStore(tmp_path / "j.db")
    assert enqueue_post_ga(db, store, [1], patch=True, ocr=False) == 1
    assert store.job_state(fid, Artifact.PATCH) == "pending"
    started: list[str] = []

    def proc(_db, job, _s):
        started.append(job.artifact.value)
        _db.upsert_features(
            job.file_id, {"patch_embeddings_meta": [{"index": 0}]}
        )
        return True

    w = Worker(db, store, processor=ArtifactProcessor(process_fn=proc), max_attempts=2)
    w.run_queue(QueueKind.HEAVY, [1], max_jobs=4)
    assert "patch" in started
    assert store.job_state(fid, Artifact.PATCH) == "done"


def test_ga_running_manual_ocr_starts(tmp_path: Path):
    db = Database(tmp_path / "mo.db")
    fid = _seed(db, tmp_path, ga_ready=False)
    assert not assess_file(db, fid).ai_final
    store = JobStore(tmp_path / "j.db")
    assert enqueue_post_ga(db, store, [1], patch=False, ocr=True) == 1
    assert store.job_state(fid, Artifact.OCR) == "pending"
    started: list[str] = []

    def proc(_db, job, _s):
        started.append(job.artifact.value)
        with _db.connect() as conn:
            conn.execute(
                "UPDATE files SET ocr_processed=1, ocr_text='m' WHERE id=?",
                (int(job.file_id),),
            )
        return True

    w = Worker(db, store, processor=ArtifactProcessor(process_fn=proc), max_attempts=2)
    w.run_queue(QueueKind.HEAVY, [1], max_jobs=4)
    assert "ocr" in started
    assert store.job_state(fid, Artifact.OCR) == "done"


def test_3_patch_fail_ocr_still_runs(tmp_path: Path):
    db = Database(tmp_path / "p.db")
    fid = _seed(db, tmp_path, ga_ready=True, patch=False)
    assert assess_file(db, fid).ai_final
    store = JobStore(tmp_path / "j.db")
    store.enqueue(
        [
            Job(fid, Artifact.PATCH, QueueKind.HEAVY, 1, "a.jpg"),
            Job(fid, Artifact.OCR, QueueKind.HEAVY, 1, "a.jpg"),
        ]
    )

    def proc(_db, job, _s):
        if job.artifact == Artifact.PATCH:
            raise RuntimeError("patch boom")
        with _db.connect() as conn:
            conn.execute(
                "UPDATE files SET ocr_processed=1, ocr_text='OK' WHERE id=?",
                (int(job.file_id),),
            )
        return True

    w = Worker(db, store, processor=ArtifactProcessor(process_fn=proc), max_attempts=1)
    w.run_queue(QueueKind.HEAVY, [1], max_jobs=8)
    after = assess_file(db, fid)
    assert after.ai_final
    assert after.get(Artifact.PATCH) == ArtifactStatus.INVALID
    assert after.ready(Artifact.OCR)
    row = db.get_file_by_id(fid) or {}
    assert str(row.get("error_msg") or "") == ""
    assert str(row.get("heavy_status") or "") != "failed"


def test_4_ocr_fail_patch_still_runs(tmp_path: Path):
    db = Database(tmp_path / "o.db")
    fid = _seed(db, tmp_path, ga_ready=True, patch=False)
    store = JobStore(tmp_path / "j.db")
    store.enqueue(
        [
            Job(fid, Artifact.OCR, QueueKind.HEAVY, 1, "a.jpg"),
            Job(fid, Artifact.PATCH, QueueKind.HEAVY, 1, "a.jpg"),
        ]
    )

    def proc(_db, job, _s):
        if job.artifact == Artifact.OCR:
            raise RuntimeError("ocr boom")
        _db.upsert_features(
            job.file_id,
            {"patch_embeddings_meta": [{"index": 0, "tag": "v3"}]},
        )
        return True

    w = Worker(db, store, processor=ArtifactProcessor(process_fn=proc), max_attempts=1)
    w.run_queue(QueueKind.HEAVY, [1], max_jobs=8)
    after = assess_file(db, fid)
    assert after.ai_final
    assert after.ready(Artifact.PATCH)
    assert after.get(Artifact.OCR) == ArtifactStatus.INVALID
    assert str((db.get_file_by_id(fid) or {}).get("error_msg") or "") == ""


def test_5_6_7_manual_starts(tmp_path: Path):
    db = Database(tmp_path / "m.db")
    fid = _seed(db, tmp_path, ga_ready=True)
    store = JobStore(tmp_path / "j.db")
    assert enqueue_post_ga(db, store, [1], patch=True, ocr=False) == 1
    assert store.job_state(fid, Artifact.PATCH) == "pending"
    store2 = JobStore(tmp_path / "j2.db")
    assert enqueue_post_ga(db, store2, [1], patch=False, ocr=True) == 1
    assert store2.job_state(fid, Artifact.OCR) == "pending"
    store3 = JobStore(tmp_path / "j3.db")
    assert enqueue_post_ga(db, store3, [1], patch=True, ocr=True) == 2


def test_8_completed_does_not_rerun(tmp_path: Path):
    db = Database(tmp_path / "c.db")
    fid = _seed(db, tmp_path, ga_ready=True, patch=True, ocr_processed=1)
    r = assess_file(db, fid)
    assert r.ready(Artifact.PATCH) and r.ready(Artifact.OCR)
    assert plan_jobs_for_file(
        r, Mode.GENERAL_AI, patch_enabled=True, ocr_enabled=True
    ) == []
    store = JobStore(tmp_path / "j.db")
    store.enqueue([Job(fid, Artifact.PATCH, QueueKind.HEAVY, 1, "a.jpg")])
    started: list[str] = []

    def proc(_db, job, _s):
        started.append(job.artifact.value)
        return True

    w = Worker(db, store, processor=ArtifactProcessor(process_fn=proc), max_attempts=2)
    w.run_queue(QueueKind.HEAVY, [1], max_jobs=1)
    assert started == []
    assert enqueue_post_ga(db, store, [1], patch=True, ocr=True) == 0
    assert enqueue_post_ga(db, store, [1], patch=True, ocr=True, reopen_done=True) >= 1


def test_9_file_change_needs_jobs_again(tmp_path: Path):
    db = Database(tmp_path / "ch.db")
    fid = _seed(db, tmp_path, ga_ready=True, patch=True, ocr_processed=1)
    db.mark_file_pipeline_reset_on_change(fid, content_changed=True)
    row = db.get_file_by_id(fid) or {}
    assert int(row.get("ocr_processed") or 0) == 0
    assert not assess_file(db, fid).ai_final
    store = JobStore(tmp_path / "j.db")
    assert enqueue_post_ga(db, store, [1], patch=True, ocr=True) >= 1
    db.upsert_file(
        {
            "path": str(tmp_path / "a.jpg"),
            "filename": "a.jpg",
            "source_id": 1,
            "width": 32,
            "feature_preview_path": str(tmp_path / "prev.webp"),
        }
    )
    with db.connect() as conn:
        conn.execute(
            "UPDATE files SET physical_preview_ready=1, width=32 WHERE id=?",
            (fid,),
        )
    db.upsert_features(
        fid,
        {
            "phash": "aabb",
            "dino_embedding": b"\x00" * 16,
            "clip_embedding": b"\x00" * 16,
            "texture_features": [0.1],
            "texture_map": {
                "semantic_tags": {"motifs": ["x"]},
                "pattern_dna": {"motif_family": "x"},
                "visual_concept_dna": {"version": 1, "objects": [], "concepts": []},
            },
        },
    )
    after = assess_file(db, fid)
    assert after.ai_final
    jobs = plan_jobs_for_file(
        after, Mode.GENERAL_AI, patch_enabled=True, ocr_enabled=True
    )
    arts = {j.artifact for j in jobs}
    assert Artifact.PATCH in arts
    assert Artifact.OCR in arts


def test_10_search_session_does_not_start_patch_ocr(tmp_path: Path):
    db = Database(tmp_path / "s.db")
    fid = _seed(db, tmp_path, ga_ready=True)
    store = JobStore(tmp_path / "j.db")
    store.enqueue([Job(fid, Artifact.PATCH, QueueKind.HEAVY, 1, "a.jpg")])
    started: list[str] = []

    def proc(_db, job, _s):
        started.append(job.artifact.value)
        return True

    w = Worker(db, store, processor=ArtifactProcessor(process_fn=proc), max_attempts=2)
    with search_session():
        deferred = w._run_one(Job(fid, Artifact.PATCH, QueueKind.HEAVY, 1, "a.jpg"))
    assert deferred is True
    assert started == []
    assert not assess_file(db, fid).ready(Artifact.PATCH)


def test_fast_and_general_ai_lanes_are_parallel():
    from core.index_v3.ui_bridge import index_lane_for_mode, lanes_are_parallel

    assert index_lane_for_mode("fast_archive") == "fast"
    assert index_lane_for_mode("night_complete") == "general_ai"
    assert index_lane_for_mode("general_ai") == "general_ai"
    assert lanes_are_parallel("fast", "general_ai") is True
    assert lanes_are_parallel("general_ai", "fast") is True
    assert lanes_are_parallel("fast", "fast") is False
    assert lanes_are_parallel("complete", "fast") is False
    assert lanes_are_parallel("general_ai", "complete") is False


def test_starting_other_lane_does_not_stop_running():
    from core.index_v3.ui_bridge import index_lane_for_mode, lanes_are_parallel

    class _W:
        def __init__(self, mode: str) -> None:
            self.index_mode = mode
            self.stopped = False

        def isRunning(self) -> bool:
            return not self.stopped

        def stop(self) -> None:
            self.stopped = True

    def decide(running_mode: str, new_mode: str) -> str:
        """Return keep | stop — same policy as MainWindow._run_index."""
        a = index_lane_for_mode(running_mode)
        b = index_lane_for_mode(new_mode)
        if a == b:
            return "keep"
        if lanes_are_parallel(a, b):
            return "keep"
        return "stop"

    fast = _W("fast_archive")
    assert decide(fast.index_mode, "night_complete") == "keep"
    assert fast.isRunning()
    ga = _W("general_ai")
    assert decide(ga.index_mode, "fast_archive") == "keep"
    assert ga.isRunning()
    complete = _W("complete")
    assert decide(complete.index_mode, "fast_archive") == "stop"


def test_same_file_followup_does_not_include_post_ga_after_dna():
    from core.index_v3.worker import _same_file_followup_artifacts

    dna = Job(1, Artifact.DNA, QueueKind.HEAVY, 1, "a.jpg")
    follow = _same_file_followup_artifacts(dna, None)
    assert follow is not None
    assert Artifact.PATCH not in follow
    assert Artifact.OCR not in follow
    assert Artifact.DNA in follow
    patch = Job(1, Artifact.PATCH, QueueKind.HEAVY, 1, "a.jpg")
    post = _same_file_followup_artifacts(patch, None)
    assert post == POST_GA_ARTIFACTS


def test_same_file_chain_does_not_run_patch_before_other_hash(tmp_path: Path):
    """DNA/HASH aynı dosyada PATCH'e atlamaz; diğer dosyanın HASH'i önce gelir."""
    db = Database(tmp_path / "chain.db")
    fid_a = _seed(db, tmp_path, ga_ready=False)
    preview = tmp_path / "prev.webp"
    fid_b = db.upsert_file(
        {
            "path": str(tmp_path / "b.jpg"),
            "filename": "b.jpg",
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
            (str(preview), int(fid_b)),
        )
    store = JobStore(tmp_path / "j.db")
    store.enqueue(
        [
            Job(fid_a, Artifact.HASH, QueueKind.HEAVY, 1, "a.jpg"),
            Job(fid_a, Artifact.PATCH, QueueKind.HEAVY, 1, "a.jpg"),
            Job(fid_a, Artifact.OCR, QueueKind.HEAVY, 1, "a.jpg"),
            Job(fid_b, Artifact.HASH, QueueKind.HEAVY, 1, "b.jpg"),
        ]
    )
    started: list[tuple[int, str]] = []

    def proc(_db, job, _s):
        started.append((int(job.file_id), job.artifact.value))
        if job.artifact == Artifact.HASH:
            _db.upsert_features(job.file_id, {"phash": "aabb"})
        elif job.artifact == Artifact.PATCH:
            _db.upsert_features(
                job.file_id, {"patch_embeddings_meta": [{"index": 0}]}
            )
        elif job.artifact == Artifact.OCR:
            with _db.connect() as conn:
                conn.execute(
                    "UPDATE files SET ocr_processed=1 WHERE id=?",
                    (int(job.file_id),),
                )
        return True

    w = Worker(db, store, processor=ArtifactProcessor(process_fn=proc), max_attempts=2)
    w.run_queue(QueueKind.HEAVY, [1], max_jobs=4)
    assert (fid_b, "hash") in started
    assert (fid_a, "patch") in started
    assert started.index((fid_b, "hash")) < started.index((fid_a, "patch"))
    assert started.index((fid_a, "hash")) < started.index((fid_b, "hash"))


def test_manual_post_ga_still_chains_patch_then_ocr(tmp_path: Path):
    db = Database(tmp_path / "man.db")
    fid = _seed(db, tmp_path, ga_ready=True)
    store = JobStore(tmp_path / "j.db")
    assert enqueue_post_ga(db, store, [1], patch=True, ocr=True) == 2
    started: list[str] = []

    def proc(_db, job, _s):
        started.append(job.artifact.value)
        if job.artifact == Artifact.PATCH:
            _db.upsert_features(
                job.file_id, {"patch_embeddings_meta": [{"index": 0}]}
            )
        else:
            with _db.connect() as conn:
                conn.execute(
                    "UPDATE files SET ocr_processed=1 WHERE id=?",
                    (int(job.file_id),),
                )
        return True

    w = Worker(db, store, processor=ArtifactProcessor(process_fn=proc), max_attempts=2)
    w.run_queue(QueueKind.HEAVY, [1], max_jobs=2)
    assert started == ["patch", "ocr"]


def test_ga_worker_cannot_claim_patch_via_heavy_artifact_filter(tmp_path: Path):
    from core.index_v3.scheduler import claim_fair

    store = JobStore(tmp_path / "secret.db")
    store.enqueue(
        [
            Job(1, Artifact.PATCH, QueueKind.HEAVY, 1, "a.jpg"),
            Job(1, Artifact.OCR, QueueKind.HEAVY, 1, "a.jpg"),
        ]
    )
    batch = claim_fair(
        store, QueueKind.HEAVY, "ga", [1], artifacts=HEAVY_ARTIFACTS
    )
    assert batch == []
    batch = claim_fair(
        store, QueueKind.HEAVY, "patch", [1], artifacts=(Artifact.PATCH,)
    )
    assert batch and batch[0].artifact == Artifact.PATCH


def test_engine_auto_patch_ocr_wait_for_ga_pool(tmp_path: Path, monkeypatch):
    from core.index_v3 import IndexEngineV3

    monkeypatch.setattr("core.index_v3.worker.IDLE_GRACE_SEC", 0.02)
    monkeypatch.setattr(
        "core.index_v3.ui_bridge.count_v3_ssot",
        lambda _db, _sids=None: {
            "preview": 10,
            "ai_final": 7,
            "waiting_preview": 0,
            "total": 10,
        },
    )
    db = Database(tmp_path / "pool.db")
    fid = _seed(db, tmp_path, ga_ready=True, patch=False)
    seen: list[str] = []

    def proc(_db, job, _s):
        seen.append(job.artifact.value)
        return True

    eng = IndexEngineV3(
        db,
        job_db_path=tmp_path / "pool.jobs.db",
        processor=ArtifactProcessor(process_fn=proc),
    )
    eng.store.enqueue(
        [
            Job(fid, Artifact.PATCH, QueueKind.HEAVY, 1, "a.jpg"),
            Job(fid, Artifact.OCR, QueueKind.HEAVY, 1, "a.jpg"),
        ]
    )
    eng.run(
        mode=Mode.GENERAL_AI,
        sources=[{"id": 1, "root_path": str(tmp_path)}],
        walk_disk=False,
        max_jobs_per_queue=4,
    )
    assert "patch" not in seen
    assert "ocr" not in seen
    assert eng.store.job_state(fid, Artifact.PATCH) == "pending"
    assert eng.store.job_state(fid, Artifact.OCR) == "pending"

    monkeypatch.setattr(
        "core.index_v3.ui_bridge.count_v3_ssot",
        lambda _db, _sids=None: {
            "preview": 10,
            "ai_final": 10,
            "waiting_preview": 0,
            "total": 10,
        },
    )
    # cache in engine is per run; new run
    eng2 = IndexEngineV3(
        db,
        job_db_path=tmp_path / "pool.jobs.db",
        processor=ArtifactProcessor(process_fn=proc),
    )
    eng2.run(
        mode=Mode.GENERAL_AI,
        sources=[{"id": 1, "root_path": str(tmp_path)}],
        walk_disk=False,
        max_jobs_per_queue=4,
    )
    assert "patch" in seen
    assert "ocr" in seen


def test_manual_patch_and_ocr_modes_ignore_ga_pool(tmp_path: Path, monkeypatch):
    from core.index_v3 import IndexEngineV3

    monkeypatch.setattr("core.index_v3.worker.IDLE_GRACE_SEC", 0.02)
    monkeypatch.setattr(
        "core.index_v3.ui_bridge.count_v3_ssot",
        lambda _db, _sids=None: {"preview": 10, "ai_final": 1, "total": 10},
    )
    db = Database(tmp_path / "man.db")
    fid = _seed(db, tmp_path, ga_ready=True, patch=False)
    seen: list[str] = []

    def proc(_db, job, _s):
        seen.append(job.artifact.value)
        if job.artifact == Artifact.PATCH:
            _db.upsert_features(
                job.file_id, {"patch_embeddings_meta": [{"index": 0}]}
            )
        if job.artifact == Artifact.OCR:
            with _db.connect() as conn:
                conn.execute(
                    "UPDATE files SET ocr_processed=1 WHERE id=?",
                    (int(job.file_id),),
                )
        return True

    jobs_path = tmp_path / "man.jobs.db"
    store_seed = JobStore(jobs_path)
    store_seed.enqueue(
        [
            Job(fid, Artifact.PATCH, QueueKind.HEAVY, 1, "a.jpg"),
            Job(fid, Artifact.OCR, QueueKind.HEAVY, 1, "a.jpg"),
        ]
    )
    eng_p = IndexEngineV3(
        db,
        job_db_path=jobs_path,
        processor=ArtifactProcessor(process_fn=proc),
    )
    eng_p.run(
        mode=Mode.PATCH,
        sources=[{"id": 1, "root_path": str(tmp_path)}],
        walk_disk=False,
        max_jobs_per_queue=2,
    )
    assert seen == ["patch"]
    assert assess_file(db, fid).ready(Artifact.PATCH)
    assert eng_p.store.job_state(fid, Artifact.OCR) == "pending"

    eng_o = IndexEngineV3(
        db,
        job_db_path=jobs_path,
        processor=ArtifactProcessor(process_fn=proc),
    )
    eng_o.run(
        mode=Mode.OCR,
        sources=[{"id": 1, "root_path": str(tmp_path)}],
        walk_disk=False,
        max_jobs_per_queue=2,
    )
    assert "ocr" in seen
    with db.connect() as conn:
        row = conn.execute(
            "SELECT ocr_processed FROM files WHERE id=?", (int(fid),)
        ).fetchone()
    assert int(row[0] if not hasattr(row, "keys") else row["ocr_processed"]) == 1


def test_manual_stop_leaves_completed_patch(tmp_path: Path, monkeypatch):
    from core.index_v3 import IndexEngineV3

    monkeypatch.setattr("core.index_v3.worker.IDLE_GRACE_SEC", 0.02)
    db = Database(tmp_path / "st.db")
    fid = _seed(db, tmp_path, ga_ready=True, patch=False)

    def proc(_db, job, _s):
        if job.artifact == Artifact.PATCH:
            _db.upsert_features(
                job.file_id, {"patch_embeddings_meta": [{"index": 0}]}
            )
        return True

    jobs_path = tmp_path / "st.jobs.db"
    JobStore(jobs_path).enqueue(
        [Job(fid, Artifact.PATCH, QueueKind.HEAVY, 1, "a.jpg")]
    )
    eng = IndexEngineV3(
        db,
        job_db_path=jobs_path,
        processor=ArtifactProcessor(process_fn=proc),
    )
    eng.run(
        mode=Mode.PATCH,
        sources=[{"id": 1, "root_path": str(tmp_path)}],
        walk_disk=False,
        max_jobs_per_queue=1,
    )
    assert assess_file(db, fid).ready(Artifact.PATCH)
    eng.request_stop()
    assert assess_file(db, fid).ready(Artifact.PATCH)


def test_lanes_patch_parallel_with_general_ai():
    from core.index_v3.ui_bridge import index_lane_for_mode, lanes_are_parallel

    assert index_lane_for_mode("patch") == "patch"
    assert index_lane_for_mode("ocr") == "ocr"
    assert lanes_are_parallel("general_ai", "patch") is True
    assert lanes_are_parallel("general_ai", "ocr") is True
    assert lanes_are_parallel("patch", "ocr") is True
    assert lanes_are_parallel("complete", "patch") is False


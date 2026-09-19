"""OCR after General AI complete (tmp DB only). Search never runs OCR."""

from __future__ import annotations

from pathlib import Path

from core.db import Database
from core.index_v3.artifact_state import assess_file
from core.index_v3.planner import plan_jobs_for_file
from core.index_v3.queues import JobStore
from core.index_v3.real_processor import RealArtifactProcessor
from core.index_v3.types import (
    AI_FINAL_REQUIRED,
    HEAVY_ARTIFACTS,
    Artifact,
    ArtifactStatus,
    FileArtifactReport,
    Job,
    Mode,
    QueueKind,
)
from core.index_v3.ui_bridge import count_v3_ssot
from core.index_v3.worker import ArtifactProcessor, Worker
from core.settings import AppSettings


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


def _ga_ready_kwargs() -> dict[str, bool]:
    kw = {a.value: True for a in HEAVY_ARTIFACTS}
    kw["preview"] = True
    kw["thumbnail"] = True
    kw["object_concept"] = True
    kw["owlv2"] = True
    kw["patch"] = True
    return kw


def _seed_file(
    db: Database,
    tmp_path: Path,
    *,
    ga_ready: bool,
    ocr_processed: int = 0,
    ocr_text: str = "",
    name: str = "a.jpg",
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
            "path": str(tmp_path / name),
            "filename": name,
            "source_id": 1,
            "status": "pending",
            "width": 32,
            "height": 32,
            "feature_preview_path": str(preview),
            "ocr_text": ocr_text,
        }
    )
    with db.connect() as conn:
        conn.execute(
            """
            UPDATE files SET physical_preview_ready=1, physical_thumbnail_ready=1,
                thumbnail_path=?, ocr_processed=?, ocr_text=?, error_msg=''
            WHERE id=?
            """,
            (str(preview), int(ocr_processed), ocr_text, int(fid)),
        )
    if ga_ready:
        db.upsert_features(
            fid,
            {
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
                "patch_embeddings_meta": [{"index": 0}],
            },
        )
        _stamp_owlv2_ready(db, fid, str(tmp_path / name))
    return int(fid)


def test_a_ocr_not_planned_before_ga_complete():
    r = _report(preview=True, thumbnail=True, hash=True, dino=True)
    jobs = plan_jobs_for_file(r, Mode.GENERAL_AI, ocr_enabled=True)
    arts = [j.artifact for j in jobs]
    assert Artifact.OCR not in arts
    assert Artifact.HASH not in arts
    assert Artifact.CLIP in arts
    assert Artifact.PATCH not in arts
    assert Artifact.PATCH not in HEAVY_ARTIFACTS
    assert Artifact.OCR not in HEAVY_ARTIFACTS
    assert Artifact.OBJECT_CONCEPT not in HEAVY_ARTIFACTS
    assert Artifact.OBJECT_CONCEPT not in AI_FINAL_REQUIRED


def test_b_ocr_planned_after_ga_complete():
    r = _report(**_ga_ready_kwargs())
    jobs = plan_jobs_for_file(r, Mode.GENERAL_AI, ocr_enabled=True)
    assert [j.artifact for j in jobs] == [Artifact.OCR]


def test_c_unchanged_file_skips_ocr():
    r = _report(**_ga_ready_kwargs(), ocr=True)
    assert plan_jobs_for_file(r, Mode.GENERAL_AI, ocr_enabled=True) == []
    assert plan_jobs_for_file(r, Mode.COMPLETE, ocr_enabled=True) == []


def test_d_content_change_resets_ocr_and_replans(tmp_path: Path):
    db = Database(tmp_path / "chg.db")
    fid = _seed_file(db, tmp_path, ga_ready=True, ocr_processed=1, ocr_text="OLD")
    assert int((db.get_file_by_id(fid) or {}).get("ocr_processed") or 0) == 1
    db.mark_file_pipeline_reset_on_change(fid, content_changed=True)
    row = db.get_file_by_id(fid) or {}
    assert int(row.get("ocr_processed") or 0) == 0
    assert str(row.get("ocr_text") or "") == ""
    report = assess_file(db, fid)
    jobs = plan_jobs_for_file(report, Mode.GENERAL_AI, ocr_enabled=True)
    assert Artifact.OCR not in [j.artifact for j in jobs]
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
            "patch_embeddings_meta": [{"index": 0}],
        },
    )
    after = assess_file(db, fid)
    assert after.ai_final
    jobs2 = plan_jobs_for_file(after, Mode.GENERAL_AI, ocr_enabled=True)
    assert [j.artifact for j in jobs2] == [Artifact.OCR]


def test_a_auto_ocr_not_planned_before_ga(tmp_path: Path):
    """Otomatik OCR AI_FINAL öncesi planlanmaz. Kuyruğa düşen OCR (manuel) worker'da çalışır."""
    r = _report(preview=True, hash=True)
    assert not r.ai_final
    jobs = plan_jobs_for_file(r, Mode.GENERAL_AI, ocr_enabled=True, patch_enabled=True)
    arts = {j.artifact for j in jobs}
    assert Artifact.OCR not in arts
    assert Artifact.PATCH not in arts

    db = Database(tmp_path / "w.db")
    fid = _seed_file(db, tmp_path, ga_ready=False)
    store = JobStore(tmp_path / "jobs.db")
    store.enqueue([Job(fid, Artifact.OCR, QueueKind.HEAVY, 1, "a.jpg")])
    started: list[str] = []

    def proc(_db, job, _stats):
        started.append(job.artifact.value)
        with _db.connect() as conn:
            conn.execute(
                "UPDATE files SET ocr_processed=1 WHERE id=?",
                (int(job.file_id),),
            )
        return True

    w = Worker(db, store, processor=ArtifactProcessor(process_fn=proc), max_attempts=2)
    w.run_queue(QueueKind.HEAVY, [1], max_jobs=2)
    assert "ocr" in started


def test_b_e_worker_runs_ocr_after_ga_and_persists(tmp_path: Path):
    db = Database(tmp_path / "w2.db")
    fid = _seed_file(db, tmp_path, ga_ready=True)
    store = JobStore(tmp_path / "jobs.db")
    store.enqueue([Job(fid, Artifact.OCR, QueueKind.HEAVY, 1, "a.jpg")])
    started: list[str] = []

    def proc(adb, job, _stats):
        started.append(job.artifact.value)
        if job.artifact == Artifact.OCR:
            with adb.connect() as conn:
                conn.execute(
                    "UPDATE files SET ocr_text=?, ocr_processed=1 WHERE id=?",
                    ("GUCCI POLO", int(job.file_id)),
                )
        return True

    w = Worker(db, store, processor=ArtifactProcessor(process_fn=proc), max_attempts=2)
    w.run_queue(QueueKind.HEAVY, [1], max_jobs=1)
    assert started == ["ocr"]
    row = db.get_file_by_id(fid) or {}
    assert row.get("ocr_text") == "GUCCI POLO"
    assert int(row.get("ocr_processed") or 0) == 1
    hits = db.search_ocr_text("GUCCI", limit=10)
    assert any(int(h["id"]) == fid for h in hits)


def test_c_reprocess_skips_ocr_job(tmp_path: Path):
    db = Database(tmp_path / "w3.db")
    fid = _seed_file(db, tmp_path, ga_ready=True, ocr_processed=1, ocr_text="KEEP")
    store = JobStore(tmp_path / "jobs.db")
    store.enqueue([Job(fid, Artifact.OCR, QueueKind.HEAVY, 1, "a.jpg")])
    started: list[str] = []

    def proc(_db, job, _stats):
        started.append(job.artifact.value)
        return True

    w = Worker(db, store, processor=ArtifactProcessor(process_fn=proc), max_attempts=2)
    w.run_queue(QueueKind.HEAVY, [1], max_jobs=1)
    assert started == []
    assert (db.get_file_by_id(fid) or {}).get("ocr_text") == "KEEP"


def test_f_ocr_text_usable_in_brand_text_search(tmp_path: Path):
    db = Database(tmp_path / "s.db")
    fid = _seed_file(
        db, tmp_path, ga_ready=True, ocr_processed=1, ocr_text="GUCCI SEXINESS"
    )
    hits = db.search_text_candidates(["gucci"], limit=80)
    ids = {int(r["id"]) for r in hits}
    assert fid in ids


def test_f_search_path_does_not_call_ocr(tmp_path: Path, monkeypatch):
    import core.ocr_engine as ocr_mod

    def boom(*_a, **_k):
        raise AssertionError("search must not run OCR")

    monkeypatch.setattr(ocr_mod.OCREngine, "extract_text", boom)
    db = Database(tmp_path / "s2.db")
    fid = _seed_file(
        db, tmp_path, ga_ready=True, ocr_processed=1, ocr_text="AMIRI"
    )
    hits = db.search_text_candidates(["amiri"], limit=20)
    assert any(int(r["id"]) == fid for r in hits)
    assert db.search_ocr_text("AMIRI")


def test_g_ocr_error_does_not_fail_ga(tmp_path: Path, monkeypatch):
    db = Database(tmp_path / "err.db")
    fid = _seed_file(db, tmp_path, ga_ready=True)
    assert assess_file(db, fid).ai_final
    assert Artifact.OCR not in AI_FINAL_REQUIRED
    settings = AppSettings()
    settings.ocr_enabled = True
    settings.cache_dir = str(tmp_path / "cache")
    proc = RealArtifactProcessor(settings, retries=0, enable_ai=False)

    class BoomOCR:
        enabled = True

        def extract_text(self, *_a, **_k):
            raise RuntimeError("tesseract down")

    monkeypatch.setattr(proc, "_get_ocr", lambda: BoomOCR())
    monkeypatch.setattr(proc, "_feature_source", lambda *_a, **_k: str(tmp_path / "prev.webp"))
    job = Job(fid, Artifact.OCR, QueueKind.HEAVY, 1, str(tmp_path / "a.jpg"))
    store = JobStore(tmp_path / "jobs.db")
    store.enqueue([job])
    w = Worker(db, store, processor=proc, max_attempts=1)
    stats = w.run_queue(QueueKind.HEAVY, [1], max_jobs=1)
    assert stats.failed >= 1
    after = assess_file(db, fid)
    assert after.ai_final
    for art in HEAVY_ARTIFACTS:
        assert after.ready(art)
    assert after.get(Artifact.OCR) == ArtifactStatus.INVALID
    assert not after.ready(Artifact.OCR)
    assert plan_jobs_for_file(after, Mode.GENERAL_AI, ocr_enabled=True) == []
    row = db.get_file_by_id(fid) or {}
    assert str(row.get("error_msg") or "") == ""
    assert str(row.get("status") or "") != "error"
    assert str(row.get("heavy_status") or "") != "failed"
    assert int(row.get("ocr_processed") or 0) == 0
    assert str(row.get("ocr_error") or "").startswith("ocr_failed:")
    pools = count_v3_ssot(db, [1])
    assert pools["ai_final"] == 1
    assert pools["ocr"] == 0
    assert store.job_state(fid, Artifact.OCR) in ("failed", "failed_permanent", "pending")
    assert store.count_failed_permanent_files(source_ids=[1]) == 0

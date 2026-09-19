"""count_v3_ssot must not run between GA stages on the index worker thread."""

from __future__ import annotations

import statistics
import time
from pathlib import Path

from core.db import Database
from core.index_v3.live_contract import worker_progress_should_compute_ssot
from core.index_v3.queues import JobStore
from core.index_v3.types import Artifact, Job, QueueKind
from core.index_v3.ui_bridge import count_v3_ssot
from core.index_v3.worker import ArtifactProcessor, Worker


GA_LINE = (
    Artifact.HASH,
    Artifact.DINO,
    Artifact.CLIP,
    Artifact.TEXTURE,
    Artifact.SEMANTIC,
    Artifact.DNA,
)

GAPS = (
    ("hash", "dino"),
    ("dino", "clip"),
    ("clip", "texture"),
    ("texture", "semantic"),
    ("semantic", "dna"),
)


def test_worker_progress_policy_never_ssot_on_v3_progress():
    assert worker_progress_should_compute_ssot("v3_progress") is False
    assert worker_progress_should_compute_ssot("index_ready") is True
    assert worker_progress_should_compute_ssot("index_done") is True
    assert worker_progress_should_compute_ssot("index_starting") is True


def _seed_ga_files(db: Database, tmp_path: Path, n: int) -> list[int]:
    preview = tmp_path / "prev.webp"
    preview.write_bytes(b"prev")
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(tmp_path)),
        )
    ids: list[int] = []
    for i in range(n):
        fid = db.upsert_file(
            {
                "path": str(tmp_path / f"f{i:04d}.jpg"),
                "filename": f"f{i:04d}.jpg",
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
        ids.append(int(fid))
    return ids


def _gaps_ms(store: JobStore, n: int) -> dict[str, list[float]]:
    out = {f"{a}_to_{b}": [] for a, b in GAPS}
    with store._connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM index_v3_file_timings
            WHERE hash_ended_at IS NOT NULL AND dna_ended_at IS NOT NULL
            """
        ).fetchall()
    for row in rows:
        d = dict(row)
        for a, b in GAPS:
            ea, sb = d.get(f"{a}_ended_at"), d.get(f"{b}_started_at")
            if ea is not None and sb is not None:
                out[f"{a}_to_{b}"].append(max(0.0, (float(sb) - float(ea)) * 1000.0))
    return out


def test_count_v3_ssot_not_called_between_ga_stages(tmp_path: Path, monkeypatch):
    db = Database(tmp_path / "g.db")
    ids = _seed_ga_files(db, tmp_path, 8)
    store = JobStore(tmp_path / "j.db")
    jobs = [
        Job(fid, art, QueueKind.HEAVY, 1, f"f{i:04d}.jpg")
        for i, fid in enumerate(ids)
        for art in GA_LINE
    ]
    store.enqueue(jobs)
    ssot_calls = {"n": 0}
    real = count_v3_ssot

    def spy(db_, source_ids=None):
        ssot_calls["n"] += 1
        return real(db_, source_ids)

    monkeypatch.setattr("core.index_v3.ui_bridge.count_v3_ssot", spy)

    def cb(info=None):
        if worker_progress_should_compute_ssot("v3_progress"):
            spy(db, [1])

    w = Worker(db, store, processor=ArtifactProcessor(), max_attempts=2)
    w.set_progress_callback(cb)
    w.run_queue(QueueKind.HEAVY, [1], max_jobs=len(jobs) + 8)
    assert ssot_calls["n"] == 0


def test_ga_stage_gaps_without_ssot_on_100_files(tmp_path: Path):
    """100 dosya stub GA hattı — SSOT yok; aşama boşluğu milisaniye olmalı."""
    n = 100
    db = Database(tmp_path / "g100.db")
    ids = _seed_ga_files(db, tmp_path, n)
    store = JobStore(tmp_path / "j100.db")
    jobs = [
        Job(fid, art, QueueKind.HEAVY, 1, f"f{i:04d}.jpg")
        for i, fid in enumerate(ids)
        for art in GA_LINE
    ]
    store.enqueue(jobs)
    ssot_n = {"n": 0}

    def cb(info=None):
        if worker_progress_should_compute_ssot("v3_progress"):
            ssot_n["n"] += 1
            count_v3_ssot(db, [1])

    t0 = time.perf_counter()
    w = Worker(db, store, processor=ArtifactProcessor(), max_attempts=2)
    w.set_progress_callback(cb)
    w.run_queue(QueueKind.HEAVY, [1], max_jobs=len(jobs) + 20)
    wall = time.perf_counter() - t0
    assert ssot_n["n"] == 0
    gaps = _gaps_ms(store, n)
    for key in ("hash_to_dino", "dino_to_clip"):
        vals = gaps[key]
        assert len(vals) >= 90, key
        med = statistics.median(vals)
        p90 = sorted(vals)[int(0.9 * (len(vals) - 1))]
        # Üretim SSOT bloğu: medyan ~8200 ms. Bu yolda < 500 ms.
        assert med < 500.0, f"{key} median {med}"
        assert p90 < 1500.0, f"{key} p90 {p90}"
    files_per_hour = (n / wall) * 3600.0 if wall > 0 else 0.0
    # Rapor için: tmp stub hızı (gerçek DINO değil)
    assert files_per_hour > 200.0

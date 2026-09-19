"""Index Lifecycle Contract — acceptance (tmp DB; production/NAS yok).

Doğrular:
  Fast / General AI / Complete / Repair
  Stop / Pause / Resume
  Kaynak kaldır / Sil+index temizle (orijinal silinmez)
  Fiziksel silme → reconcile → missing → aktif sayı düşer, kuyruk düşer
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from PIL import Image

from core.db import Database
from core.index_v3 import IndexEngineV3, Mode, Artifact
from core.index_v3.progress import count_progress
from core.index_v3.scope import resolve_index_scope
from core.index_v3.ui_bridge import count_v3_ssot, v3_status_dict
from core.index_v3.types import QueueKind


def _imgs(folder: Path, n: int, prefix: str = "f") -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for i in range(n):
        p = folder / f"{prefix}_{i:03d}.jpg"
        Image.new("RGB", (36, 36), color=(i % 180, 30, 80)).save(p, "JPEG")
        out.append(p)
    return out


@pytest.fixture
def life(tmp_path: Path):
    db = Database(tmp_path / "life.db")
    root = tmp_path / "src"
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("life", str(root)),
        )
    paths = _imgs(root, 20)
    eng = IndexEngineV3(db, job_db_path=tmp_path / "jobs.db")
    sources = [{"id": 1, "root_path": str(root)}]
    return db, eng, sources, root, paths, tmp_path


def test_fast_only_light_no_heavy(life):
    db, eng, sources, root, paths, tmp = life
    eng.run(mode=Mode.FAST, sources=sources, walk_disk=True)
    c = count_v3_ssot(db, [1])
    assert c["total"] == 20
    assert c["preview"] == 20
    assert c["dino"] == 0
    assert c["texture"] == 0
    assert c["ai_final"] == 0


def test_general_ai_only_heavy_after_preview(life):
    db, eng, sources, root, paths, tmp = life
    eng.run(mode=Mode.FAST, sources=sources, walk_disk=True)
    before_prev = count_v3_ssot(db, [1])["preview"]
    eng.run(mode=Mode.GENERAL_AI, sources=sources, walk_disk=False)
    c = count_v3_ssot(db, [1])
    assert c["preview"] == before_prev == 20
    assert c["dino"] == 20
    assert c["ai_final"] == 20


def test_complete_light_then_heavy_independent_lanes(life):
    db, eng, sources, root, paths, tmp = life
    eng.run(mode=Mode.COMPLETE, sources=sources, walk_disk=True)
    c = count_v3_ssot(db, [1])
    assert c["preview"] == 20
    assert c["ai_final"] == 20
    # ikinci geçiş idempotent
    eng.run(mode=Mode.COMPLETE, sources=sources, walk_disk=False)
    assert count_v3_ssot(db, [1])["ai_final"] == 20


def test_repair_fills_fast_then_heavy(life):
    db, eng, sources, root, paths, tmp = life
    eng.run(mode=Mode.COMPLETE, sources=sources, walk_disk=True)
    with db.connect() as conn:
        fid = int(conn.execute("SELECT id FROM files ORDER BY id LIMIT 1").fetchone()[0])
        conn.execute(
            "UPDATE files SET feature_preview_path='', physical_preview_ready=0 WHERE id=?",
            (fid,),
        )
        conn.execute(
            "UPDATE features SET dino_embedding=NULL WHERE file_id=?",
            (fid,),
        )
    assert count_v3_ssot(db, [1])["preview"] == 19
    eng.run(mode=Mode.REPAIR, sources=sources, walk_disk=False)
    assert count_v3_ssot(db, [1])["preview"] == 20
    assert count_v3_ssot(db, [1])["dino"] == 20
    assert count_v3_ssot(db, [1])["ai_final"] == 20


def test_stop_halts_processing(life):
    db, eng, sources, root, paths, tmp = life
    # max_jobs ile kısmi; stop API çalışır
    t = threading.Thread(
        target=lambda: eng.run(
            mode=Mode.COMPLETE, sources=sources, walk_disk=True, max_jobs_per_queue=2
        ),
        daemon=True,
    )
    t.start()
    time.sleep(0.05)
    eng.request_stop()
    t.join(timeout=30)
    assert not t.is_alive()


def test_pause_resume_preserves_progress(life):
    db, eng, sources, root, paths, tmp = life
    eng.run(mode=Mode.FAST, sources=sources, walk_disk=True, max_jobs_per_queue=5)
    mid = count_v3_ssot(db, [1])["preview"]
    assert mid >= 1
    eng.request_pause()
    time.sleep(0.05)
    eng.request_resume()
    eng.run(mode=Mode.FAST, sources=sources, walk_disk=False)
    assert count_v3_ssot(db, [1])["preview"] == 20
    eng.run(mode=Mode.COMPLETE, sources=sources, walk_disk=False)
    assert count_v3_ssot(db, [1])["ai_final"] == 20


def test_purge_index_keeps_original_files(life):
    db, eng, sources, root, paths, tmp = life
    eng.run(mode=Mode.FAST, sources=sources, walk_disk=True)
    for p in paths:
        assert p.is_file()
    # index kayıtlarını sil (sources.purge benzeri)
    removed = db.purge_source_records(1, cache_dir=tmp / "cache")
    assert removed == 20
    for p in paths:
        assert p.is_file(), "orijinal silinmemeli"
    assert count_v3_ssot(db, [1])["total"] == 0


def test_remove_source_keep_index_out_of_archive(life):
    db, eng, sources, root, paths, tmp = life
    eng.run(mode=Mode.FAST, sources=sources, walk_disk=True)
    db.delete_source(1)
    scope = resolve_index_scope(db, [])
    assert scope.source_ids == []
    assert scope.orphan_file_total == 20
    assert count_v3_ssot(db, scope.source_ids)["total"] == 0
    st = v3_status_dict(db, scope.source_ids, scope=scope)
    assert st["total"] == 0
    assert st["orphan_file_total"] == 20


def test_physical_delete_reconcile_marks_missing_drops_active(life):
    """Fiziksel silme → walk reconcile → missing; aktif sayı düşer."""
    db, eng, sources, root, paths, tmp = life
    eng.run(mode=Mode.COMPLETE, sources=sources, walk_disk=True)
    assert count_v3_ssot(db, [1])["total"] == 20

    deleted = paths[:5]
    for p in deleted:
        p.unlink()
        assert not p.is_file()

    report = eng.run(mode=Mode.FAST, sources=sources, walk_disk=True)
    src_disc = (report.discovery.get("sources") or [{}])[0]
    assert int(src_disc.get("missing") or 0) == 5

    active = count_v3_ssot(db, [1])["total"]
    assert active == 15
    with db.connect() as conn:
        miss = int(
            conn.execute(
                "SELECT COUNT(*) FROM files WHERE source_id=1 AND status='missing'"
            ).fetchone()[0]
        )
        mids = [
            int(r[0])
            for r in conn.execute(
                "SELECT id FROM files WHERE status='missing'"
            ).fetchall()
        ]
    assert miss == 5
    # missing için pending/claimed job kalmamalı
    bad = 0
    if mids:
        with eng.store._connect() as conn:
            ph = ",".join("?" * len(mids))
            bad = int(
                conn.execute(
                    f"""
                    SELECT COUNT(*) AS n FROM index_v3_jobs
                    WHERE state IN ('pending','claimed') AND file_id IN ({ph})
                    """,
                    mids,
                ).fetchone()["n"]
                or 0
            )
    assert bad == 0
    assert sum(1 for p in paths if p.is_file()) == 15
    assert active == 15


def test_active_count_equals_physically_existing(life):
    db, eng, sources, root, paths, tmp = life
    eng.run(mode=Mode.FAST, sources=sources, walk_disk=True)
    for p in paths[10:]:
        p.unlink()
    eng.run(mode=Mode.REPAIR, sources=sources, walk_disk=True)
    active = count_v3_ssot(db, [1])["total"]
    physical = sum(1 for p in root.glob("*.jpg") if p.is_file())
    assert active == physical == 10

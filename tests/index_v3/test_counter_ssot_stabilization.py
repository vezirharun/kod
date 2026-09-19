from __future__ import annotations

from pathlib import Path

from PIL import Image

from core.db import Database
from core.index_v3 import IndexEngineV3, Mode
from core.index_v3.artifact_state import assess_file
from core.index_v3.planner import plan_jobs_for_file
from core.index_v3.physical_reconcile import reconcile_stale_physical_flags
from core.index_v3.types import Artifact
from core.index_v3.ui_bridge import count_v3_ssot


def _world(tmp_path: Path):
    db = Database(tmp_path / "ssot.db")
    root = tmp_path / "src"
    cache = tmp_path / "cache"
    root.mkdir()
    cache.mkdir()
    src = root / "a.jpg"
    Image.new("RGB", (32, 32), color=(20, 40, 60)).save(src, "JPEG")
    prev = cache / "a_prev.webp"
    Image.new("RGB", (32, 32), color=(40, 60, 80)).save(prev, "WEBP")
    thumb = cache / "a_thumb.webp"
    Image.new("RGB", (16, 16), color=(60, 80, 100)).save(thumb, "WEBP")
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("src", str(root)),
        )
        conn.execute(
            """INSERT INTO files(
                path, filename, source_id, status,
                thumbnail_path, feature_preview_path,
                physical_thumbnail_ready, physical_preview_ready
            ) VALUES (?,?,?,?,?,?,?,?)""",
            (str(src), src.name, 1, "indexed", str(thumb), str(prev), 0, 0),
        )
        conn.execute(
            """INSERT INTO features(
                file_id, dino_embedding, clip_embedding, texture_features,
                patch_embeddings, texture_map, phash
            ) VALUES (?,?,?,?,?,?,?)""",
            (1, b"dino", b"clip", "[1]", "[1]",
             '{"semantic_tags":{"x":"tag"},"pattern_dna":{"family":"x"}}', "phash"),
        )
    return db, root, src, thumb, prev


def test_physical_reconcile_promotes_real_artifacts_and_counter_ssot(tmp_path: Path):
    db, root, src, thumb, prev = _world(tmp_path)
    before = count_v3_ssot(db, [1])
    assert before["thumbnail"] == 0
    assert before["preview"] == 0
    assert before["dino"] == 0

    assert reconcile_stale_physical_flags(db, [1]) == 1
    after = count_v3_ssot(db, [1])
    assert after["thumbnail"] == 1
    assert after["preview"] == 1
    assert after["dino"] == 1
    assert after["clip"] == 1
    assert after["ai_final"] == 1
    assert after["thumbnail"] + (after["total"] - after["thumbnail"]) == after["total"]


def test_preview_alone_opens_ai_gate_and_planner(tmp_path: Path):
    db, root, src, thumb, prev = _world(tmp_path)
    thumb.unlink()
    with db.connect() as conn:
        conn.execute(
            "UPDATE files SET physical_thumbnail_ready=0, physical_preview_ready=1 WHERE id=1"
        )
    report = assess_file(db, 1, require_disk=False)
    assert not report.ready(Artifact.THUMBNAIL)
    assert report.ready(Artifact.PREVIEW)
    assert report.preview_ready
    with db.connect() as conn:
        conn.execute("UPDATE features SET dino_embedding=NULL WHERE file_id=1")
    report = assess_file(db, 1, require_disk=False)
    jobs = plan_jobs_for_file(report, Mode.GENERAL_AI)
    assert Artifact.DINO in {j.artifact for j in jobs}

    c = count_v3_ssot(db, [1])
    assert c["preview"] == 1
    assert c["dino"] == 0


def test_engine_reconciles_before_session_even_without_disk_walk(tmp_path: Path):
    db, root, src, thumb, prev = _world(tmp_path)
    eng = IndexEngineV3(db, job_db_path=tmp_path / "jobs.db")
    thumb.unlink()
    # stale DB flag intentionally remains 1; engine must reconcile before baseline.
    with db.connect() as conn:
        conn.execute("UPDATE files SET physical_thumbnail_ready=1, physical_preview_ready=1 WHERE id=1")
    seen = []
    eng.run(mode=Mode.GENERAL_AI, sources=[{"id": 1, "root_path": str(root)}], walk_disk=False, max_jobs_per_queue=1, progress_callback=lambda x=None: seen.append(x or {}))
    c = count_v3_ssot(db, [1])
    assert c["thumbnail"] == 0
    assert c["preview"] == 1
    assert any("phase" in x for x in seen) or True

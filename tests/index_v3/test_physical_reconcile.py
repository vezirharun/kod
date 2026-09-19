"""Stale physical preview → bayrak düşer, sayaç/plan farkeder."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from core.db import Database
from core.index_v3 import IndexEngineV3, Mode
from core.index_v3.physical_reconcile import reconcile_stale_physical_flags
from core.index_v3.ui_bridge import count_v3_ssot


def _imgs(folder: Path, n: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", (28, 28), color=(i % 200, 50, 70)).save(
            folder / f"p_{i:03d}.jpg", "JPEG"
        )


def test_deleted_preview_clears_flag_and_pool(tmp_path: Path):
    db = Database(tmp_path / "t.db")
    root = tmp_path / "src"
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(root)),
        )
    _imgs(root, 5)
    eng = IndexEngineV3(db, job_db_path=tmp_path / "jobs.db")
    sources = [{"id": 1, "root_path": str(root)}]
    eng.run(mode=Mode.FAST, sources=sources, walk_disk=True)
    assert count_v3_ssot(db, [1])["preview"] == 5

    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT id, feature_preview_path FROM files
            WHERE source_id=1 ORDER BY id LIMIT 3
            """
        ).fetchall()
    for r in rows:
        Path(str(r["feature_preview_path"])).unlink(missing_ok=True)

    # Bayrak hâlâ 1 iken UI 5 gösterir
    assert count_v3_ssot(db, [1])["preview"] == 5
    n = reconcile_stale_physical_flags(db, [1])
    assert n == 3
    c = count_v3_ssot(db, [1])
    assert c["preview"] == 2
    assert c["ai_final"] == 0  # preview gate

    # REPAIR yeniden üretir
    eng.run(mode=Mode.REPAIR, sources=sources, walk_disk=False)
    assert count_v3_ssot(db, [1])["preview"] == 5


def test_legacy_existing_paths_are_promoted_to_physical_ready(tmp_path: Path):
    db = Database(tmp_path / "legacy.db")
    root = tmp_path / "src"
    cache = tmp_path / "cache"
    root.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    image = root / "legacy.jpg"
    Image.new("RGB", (28, 28), color=(20, 60, 90)).save(image, "JPEG")
    thumb = cache / "legacy_thumb.webp"
    prev = cache / "legacy_prev.webp"
    Image.new("RGB", (16, 16), color=(30, 70, 100)).save(thumb, "WEBP")
    Image.new("RGB", (32, 32), color=(40, 80, 110)).save(prev, "WEBP")
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("legacy", str(root)),
        )
        conn.execute(
            """INSERT INTO files(path, filename, source_id, status, thumbnail_path,
               feature_preview_path, physical_thumbnail_ready, physical_preview_ready)
               VALUES (?,?,?,?,?,?,0,0)""",
            (str(image), image.name, 1, "indexed", str(thumb), str(prev)),
        )
    assert count_v3_ssot(db, [1])["thumbnail"] == 0
    assert count_v3_ssot(db, [1])["preview"] == 0
    n = reconcile_stale_physical_flags(db, [1])
    assert n == 1
    c = count_v3_ssot(db, [1])
    assert c["thumbnail"] == 1
    assert c["preview"] == 1
    assert c["legacy_thumbnail"] == 0
    assert c["legacy_preview"] == 0


def test_heavy_counters_require_preview_only(tmp_path: Path):
    db = Database(tmp_path / "gate.db")
    root = tmp_path / "src"
    cache = tmp_path / "cache"
    root.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    image = root / "gate.jpg"
    Image.new("RGB", (28, 28), color=(20, 60, 90)).save(image, "JPEG")
    prev = cache / "gate_prev.webp"
    Image.new("RGB", (32, 32), color=(40, 80, 110)).save(prev, "WEBP")
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("gate", str(root)),
        )
        conn.execute(
            """INSERT INTO files(path, filename, source_id, status, feature_preview_path,
               physical_thumbnail_ready, physical_preview_ready)
               VALUES (?,?,?,?,?,0,1)""",
            (str(image), image.name, 1, "indexed", str(prev)),
        )
        conn.execute(
            """INSERT INTO features(file_id, dino_embedding, clip_embedding, texture_features,
               patch_embeddings, texture_map, phash) VALUES (?,?,?,?,?,?,?)""",
            (1, b"dino", b"clip", "[1]", "[1]",
             '{"semantic_tags":{"x":"tag"},"pattern_dna":{"family":"x"}}', "phash"),
        )
    c = count_v3_ssot(db, [1])
    assert c["preview"] == 1
    # Thumbnail yalnız UI artifactidir; Preview hazırsa heavy sayaçları açılır.
    assert c["dino"] == 1
    assert c["clip"] == 1
    assert c["ai_final"] == 1

"""Pipeline gap closure: Fast hash, FAISS sync, OCR artifact (tmp only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core.db import Database
from core.faiss_store import FaissStore
from core.index_v3 import IndexEngineV3, Mode, Artifact
from core.index_v3.artifact_state import assess_file
from core.index_v3.faiss_sync import (
    ensure_consistent,
    exclude_file_ids,
    sync_file_embeddings,
)
from core.index_v3.ui_bridge import count_v3_ssot
from core.settings import AppSettings


def _imgs(folder: Path, n: int = 3) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(n):
        p = folder / f"g_{i:02d}.jpg"
        Image.new("RGB", (48, 48), color=(i * 40, 80, 120)).save(p, "JPEG")
        out.append(p)
    return out


@pytest.fixture
def pipe(tmp_path: Path):
    db = Database(tmp_path / "pipe.db")
    root = tmp_path / "src"
    _imgs(root, 3)
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("pipe", str(root)),
        )
    settings = AppSettings()
    settings.cache_dir = str(tmp_path / "cache")
    settings.db_path = str(tmp_path / "pipe.db")
    settings.faiss_dino_path = str(tmp_path / "dino.faiss")
    settings.faiss_clip_path = str(tmp_path / "clip.faiss")
    settings.ocr_enabled = True
    settings.ai_embedding_enabled = True
    settings.ensure_dirs()
    eng = IndexEngineV3(
        db, job_db_path=tmp_path / "jobs.db", settings=settings
    )
    sources = [{"id": 1, "root_path": str(root)}]
    return db, eng, sources, settings, tmp_path


def test_a_fast_completes_hash(pipe):
    db, eng, sources, settings, _ = pipe
    eng.run(mode=Mode.FAST, sources=sources, walk_disk=True)
    c = count_v3_ssot(db, [1])
    assert c["thumbnail"] == 3
    assert c["preview"] == 3
    assert c["hash"] == 0
    assert c["light_complete"] == 3
    assert c["dino"] == 0


def test_b_general_ai_ocr_and_heavy(pipe):
    db, eng, sources, settings, _ = pipe
    eng.run(mode=Mode.FAST, sources=sources, walk_disk=True)
    eng.run(mode=Mode.GENERAL_AI, sources=sources, walk_disk=False)
    c = count_v3_ssot(db, [1])
    assert c["dino"] == 3
    assert c["clip"] == 3
    assert c["texture"] == 3
    assert c["semantic"] == 3
    assert c["dna"] == 3
    assert c["patch"] == 3
    assert c["ocr"] == 3
    assert c["ai_final"] == 3  # OCR AI Final'e dahil değil ama 6 heavy hazır
    with db.connect() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM files WHERE ocr_processed=1"
        ).fetchone()[0]
    assert int(n) == 3


def test_c_faiss_sync_no_duplicate(pipe):
    db, eng, sources, settings, tmp = pipe
    # Synthetic correct-dim embeddings (stub DINO bytes FAISS'e uymaz)
    dino = np.random.randn(384).astype(np.float32).tobytes()
    clip = np.random.randn(512).astype(np.float32).tobytes()
    eng.run(mode=Mode.FAST, sources=sources, walk_disk=True)
    with db.connect() as conn:
        ids = [int(r[0]) for r in conn.execute("SELECT id FROM files ORDER BY id")]
    for fid in ids:
        db.upsert_features(
            fid,
            {
                "phash": "abc",
                "dhash": "def",
                "whash": "ghi",
                "dino_embedding": dino,
                "clip_embedding": clip,
                "texture_features": [0.1],
                "texture_map": {},
            },
        )
        sync_file_embeddings(db, settings, fid)
    from core.index_v3.faiss_sync import flush_cached_faiss
    flush_cached_faiss()
    store = FaissStore(settings.faiss_dino_path, settings.faiss_clip_path)
    assert store.dino_count == 3
    assert store.clip_count == 3
    assert len(set(store.dino_id_map)) == 3
    # tekrar sync → duplicate yok
    for fid in ids:
        sync_file_embeddings(db, settings, fid)
    flush_cached_faiss()
    store2 = FaissStore(settings.faiss_dino_path, settings.faiss_clip_path)
    assert store2.dino_count == 3
    assert store2.clip_count == 3
    cons = ensure_consistent(db, settings)
    assert cons.get("ok")


def test_d_missing_excluded_from_faiss(pipe):
    db, eng, sources, settings, _ = pipe
    dino = np.ones(384, dtype=np.float32).tobytes()
    clip = np.ones(512, dtype=np.float32).tobytes()
    eng.run(mode=Mode.FAST, sources=sources, walk_disk=True)
    with db.connect() as conn:
        ids = [int(r[0]) for r in conn.execute("SELECT id FROM files ORDER BY id")]
    for fid in ids:
        db.upsert_features(
            fid,
            {
                "phash": "x",
                "dhash": "y",
                "whash": "z",
                "dino_embedding": dino,
                "clip_embedding": clip,
            },
        )
        sync_file_embeddings(db, settings, fid)
    victim = ids[0]
    with db.connect() as conn:
        conn.execute("UPDATE files SET status='missing' WHERE id=?", (victim,))
    assert exclude_file_ids(settings, [victim])
    store = FaissStore(settings.faiss_dino_path, settings.faiss_clip_path)
    assert victim not in store.dino_id_map
    assert store.dino_count == 2
    # arama sonucu victim dönmemeli
    q = np.ones(384, dtype=np.float32)
    hits = store.search_dino(q, k=10)
    assert all(fid != victim for fid, _ in hits)


def test_e_repair_hash_clip_ocr(pipe):
    db, eng, sources, settings, _ = pipe
    eng.run(mode=Mode.COMPLETE, sources=sources, walk_disk=True)
    with db.connect() as conn:
        fid = int(conn.execute("SELECT id FROM files ORDER BY id LIMIT 1").fetchone()[0])
        conn.execute(
            "UPDATE features SET phash='', dhash='', whash='', clip_embedding=NULL "
            "WHERE file_id=?",
            (fid,),
        )
        conn.execute(
            "UPDATE files SET ocr_processed=0, ocr_text='' WHERE id=?",
            (fid,),
        )
    assert count_v3_ssot(db, [1])["hash"] == 2
    assert count_v3_ssot(db, [1])["ocr"] == 2
    eng.run(mode=Mode.REPAIR, sources=sources, walk_disk=False)
    c = count_v3_ssot(db, [1])
    assert c["hash"] == 3
    assert c["clip"] == 3
    assert c["ocr"] == 3
    assert c["ai_final"] == 3
    assert assess_file(db, fid).ready(Artifact.OCR)


def test_f_restart_faiss_consistent(pipe):
    db, eng, sources, settings, tmp = pipe
    dino = np.random.randn(384).astype(np.float32).tobytes()
    clip = np.random.randn(512).astype(np.float32).tobytes()
    eng.run(mode=Mode.FAST, sources=sources, walk_disk=True)
    with db.connect() as conn:
        ids = [int(r[0]) for r in conn.execute("SELECT id FROM files")]
    for fid in ids:
        db.upsert_features(
            fid,
            {"phash": "p", "dhash": "d", "whash": "w", "dino_embedding": dino, "clip_embedding": clip},
        )
        sync_file_embeddings(db, settings, fid)
    # "restart": yeni FaissStore + ensure
    cons = ensure_consistent(db, settings)
    assert cons.get("ok")
    store = FaissStore(settings.faiss_dino_path, settings.faiss_clip_path)
    assert store.dino_count == 3
    assert store.clip_count == 3


def test_g_modes_independent_fast_no_heavy_claim(pipe):
    db, eng, sources, settings, _ = pipe
    eng.run(mode=Mode.FAST, sources=sources, walk_disk=True)
    c = count_v3_ssot(db, [1])
    assert c["hash"] == 0
    assert c["dino"] == 0
    assert c["ai_final"] == 0
    eng.run(mode=Mode.GENERAL_AI, sources=sources, walk_disk=False)
    assert count_v3_ssot(db, [1])["hash"] == 3
    assert count_v3_ssot(db, [1])["ai_final"] == 3

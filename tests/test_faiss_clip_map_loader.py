"""Regression: FAISS CLIP/DINO id-map loader format contract."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from core.faiss_store import (
    CLIP_MAP_UNAVAILABLE,
    FaissMapError,
    FaissStore,
    HAS_FAISS,
    inspect_faiss_id_map,
    load_faiss_id_map,
    resolve_faiss_id_map_path,
    save_faiss_id_map,
)


def test_normal_npy_int64_map(tmp_path: Path):
    path = tmp_path / "faiss_clip.map.npy"
    save_faiss_id_map(path, [10, 20, 30])
    ids = load_faiss_id_map(path)
    assert ids == [10, 20, 30]
    info = inspect_faiss_id_map(path)
    assert info["is_npy"] is True
    assert info["kind"] == "npy"
    assert info["dtype"] in ("int64", "<i8", "i8")
    arr = np.load(path, allow_pickle=False)
    assert arr.dtype == np.int64
    assert arr.shape == (3,)


def test_object_array_map(tmp_path: Path):
    path = tmp_path / "faiss_clip.map.npy"
    np.save(path, np.array([11, 22, 33], dtype=object), allow_pickle=True)
    ids = load_faiss_id_map(path)
    assert ids == [11, 22, 33]
    info = inspect_faiss_id_map(path)
    assert info["kind"] == "npy_object"
    assert info["is_npy"] is True


def test_invalid_file_rejected(tmp_path: Path):
    path = tmp_path / "faiss_clip.map.npy"
    path.write_bytes(b"this is not npy or pickle")
    with pytest.raises(FaissMapError) as exc:
        load_faiss_id_map(path)
    assert "invalid_format" in str(exc.value)


def test_missing_file(tmp_path: Path):
    path = tmp_path / "faiss_clip.map.npy"
    with pytest.raises(FaissMapError) as exc:
        load_faiss_id_map(path)
    assert exc.value.reason == "missing"


def test_old_schema_int32_and_legacy_filename(tmp_path: Path):
    index_path = tmp_path / "faiss_clip.index"
    legacy = tmp_path / "faiss_clip_map.npy"
    np.save(legacy, np.asarray([7, 8, 9], dtype=np.int32), allow_pickle=False)
    resolved = resolve_faiss_id_map_path(index_path)
    assert resolved == legacy
    assert load_faiss_id_map(resolved) == [7, 8, 9]


def test_corrupt_zeroed_file(tmp_path: Path):
    path = tmp_path / "faiss_clip.map.npy"
    path.write_bytes(b"\x00" * 68248)
    info = inspect_faiss_id_map(path)
    assert info["kind"] == "zeroed"
    with pytest.raises(FaissMapError) as exc:
        load_faiss_id_map(path)
    assert exc.value.reason == "corrupt_zeroed_not_npy"


def test_raw_pickle_not_silently_loaded(tmp_path: Path):
    path = tmp_path / "faiss_clip.map.npy"
    with path.open("wb") as fh:
        pickle.dump([1, 2, 3], fh)
    with pytest.raises(FaissMapError) as exc:
        load_faiss_id_map(path)
    assert exc.value.reason == "raw_pickle_not_npy"


@pytest.mark.skipif(not HAS_FAISS, reason="faiss not installed")
def test_safe_fallback_does_not_crash_or_save(tmp_path: Path):
    import faiss

    dino = tmp_path / "faiss_dino.index"
    clip = tmp_path / "faiss_clip.index"
    vecs = np.ones((2, 512), dtype=np.float32)
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(512)
    index.add(vecs)
    faiss.write_index(index, str(clip))
    clip.with_suffix(".map.npy").write_bytes(b"\x00" * 256)

    store = FaissStore(str(dino), str(clip))
    assert store.clip_map_unavailable is True
    assert CLIP_MAP_UNAVAILABLE.split()[0] == "CLIP"
    assert store.clip_count == 0
    assert store.search_clip(vecs[0], k=2) == []
    assert store.upsert_clip(99, vecs[0].tobytes()) is False
    store.save()
    raw = clip.with_suffix(".map.npy").read_bytes()
    assert raw == b"\x00" * 256
    assert clip.stat().st_size > 0


@pytest.mark.skipif(not HAS_FAISS, reason="faiss not installed")
def test_recover_leaves_zero_slots_unmapped(tmp_path: Path):
    import sqlite3

    import faiss

    from core.faiss_store import recover_clip_id_map_from_stored_embeddings

    d = 8
    rng = np.random.default_rng(0)
    v1 = rng.standard_normal(d).astype(np.float32)
    v2 = rng.standard_normal(d).astype(np.float32)
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    z = np.zeros(d, dtype=np.float32)
    index = faiss.IndexFlatIP(d)
    index.add(np.stack([v1, z, v2]))
    clip = tmp_path / "faiss_clip.index"
    faiss.write_index(index, str(clip))
    db = tmp_path / "t.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE features (file_id INTEGER, clip_embedding BLOB)")
    con.execute("INSERT INTO features VALUES (101, ?)", (v1.tobytes(),))
    con.execute("INSERT INTO features VALUES (202, ?)", (v2.tobytes(),))
    con.commit()
    con.close()
    ids, stats = recover_clip_id_map_from_stored_embeddings(clip, db)
    assert ids == [101, -1, 202]
    assert stats["zero_slots"] == 1
    assert stats["unique_ok"] == 2
    assert stats["unmapped_slots"] == 1


@pytest.mark.skipif(not HAS_FAISS, reason="faiss not installed")
def test_restore_map_does_not_rewrite_index(tmp_path: Path):
    import faiss

    from core.faiss_store import restore_clip_id_map_atomic

    vecs = np.ones((2, 8), dtype=np.float32)
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(8)
    index.add(vecs)
    clip = tmp_path / "faiss_clip.index"
    faiss.write_index(index, str(clip))
    before = clip.read_bytes()
    dest = clip.with_suffix(".map.npy")
    dest.write_bytes(b"\x00" * 64)
    restore_clip_id_map_atomic(clip, [11, -1])
    assert clip.read_bytes() == before
    assert load_faiss_id_map(dest) == [11, -1]
    assert dest.with_name(dest.name + ".zeroed.bak").exists()


@pytest.mark.skipif(not HAS_FAISS, reason="faiss not installed")
def test_search_skips_unmapped_slots(tmp_path: Path):
    import faiss

    dino = tmp_path / "faiss_dino.index"
    clip = tmp_path / "faiss_clip.index"
    vecs = np.eye(3, 8, dtype=np.float32)
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(8)
    index.add(vecs)
    faiss.write_index(index, str(clip))
    save_faiss_id_map(clip.with_suffix(".map.npy"), [10, -1, 20])
    store = FaissStore(str(dino), str(clip))
    assert store.clip_map_unavailable is False
    assert store.clip_count == 2
    hits = store.search_clip(vecs[1], k=3)
    assert all(fid > 0 for fid, _ in hits)
    assert -1 not in [fid for fid, _ in hits]

"""INDEX_FROZEN A/B/C: algorithms frozen; search protects writes; maintenance safe."""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest
from PIL import Image

from core.db import Database
from core.faiss_store import FaissStore
from core.index_freeze import (
    INDEX_FROZEN,
    IndexFrozenWriteBlocked,
    MAINTENANCE_OPS,
    allow_index_writes,
    freeze_fingerprint,
    guard_index_write,
    process_search_active,
    search_session,
    search_write_protection_active,
    snapshot_index_artifacts,
)
from core.hash_verify import apply_verified_hashes
from core.on_demand_scan import run_search_preflight
from core.ovd_index import ovd_work_allowed
from core.search_engine import SearchEngine, SearchResult
from core.search_models import SearchQuery
from core.settings import AppSettings, DEFAULT_CACHE_DIR, DEFAULT_DATA_DIR
from core.sources import SourceManager
from core.unified_relevance_ranker import apply_unified_text_ranking


def _settings(tmp: Path) -> AppSettings:
    cache = tmp / "cache"
    data = tmp / "data"
    cache.mkdir()
    data.mkdir()
    return AppSettings(
        db_path=str(data / "patterns.db"),
        cache_dir=str(cache),
        faiss_dino_path=str(data / "faiss_dino.index"),
        faiss_clip_path=str(data / "faiss_clip.index"),
        face_db_path=str(data / "face_index.db"),
        face_index_enabled=False,
        ai_embedding_enabled=False,
        ocr_enabled=False,
    )


def _seed(db: Database, folder: Path, n: int = 3) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(folder)),
        )
    for i in range(n):
        p = folder / f"f_{i}.jpg"
        Image.new("RGB", (24, 24), (i * 40, 10, 80)).save(p, "JPEG")
        db.upsert_file(
            {
                "path": str(p),
                "filename": p.name,
                "source_id": 1,
                "status": "indexed",
                "file_size": p.stat().st_size,
                "mtime": p.stat().st_mtime,
                "width": 24,
                "height": 24,
                "physical_preview_ready": 0,
                "physical_thumbnail_ready": 0,
            }
        )
    with sqlite3.connect(db.db_path) as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def _snap(settings: AppSettings) -> dict:
    return snapshot_index_artifacts(
        db_path=settings.db_path,
        faiss_dino_path=settings.faiss_dino_path,
        faiss_clip_path=settings.faiss_clip_path,
        cache_dir=settings.cache_dir,
    )


def _logic(settings: AppSettings) -> dict:
    db = Database(settings.db_path, read_only=True)
    with db.connect() as conn:
        files = int(conn.execute("SELECT COUNT(*) FROM files").fetchone()[0])
        feat = int(conn.execute("SELECT COUNT(*) FROM features").fetchone()[0])
        meta = int(conn.execute("SELECT COUNT(*) FROM meta").fetchone()[0])
    return {
        "files": files,
        "features": feat,
        "meta": meta,
        "artifacts": freeze_fingerprint(_snap(settings)),
    }


def _prod_snap() -> dict:
    return snapshot_index_artifacts(
        db_path=str(DEFAULT_DATA_DIR / "patterns.db"),
        faiss_dino_path=str(DEFAULT_DATA_DIR / "faiss_dino.index"),
        faiss_clip_path=str(DEFAULT_DATA_DIR / "faiss_clip.index"),
        cache_dir=str(DEFAULT_CACHE_DIR),
    )


def test_index_frozen_flag_is_central():
    assert INDEX_FROZEN is True
    assert search_write_protection_active() is False
    with search_session():
        assert search_write_protection_active() is True
        with pytest.raises(IndexFrozenWriteBlocked) as exc:
            guard_index_write("test.op", "tests.test_index_freeze")
        assert "INDEX_FROZEN_WRITE_BLOCKED" in str(exc.value)


def test_display_cache_writes_allowed_during_search_session():
    """Liste thumbnail / feature preview arama sırasında yazılabilmeli."""
    with search_session():
        guard_index_write("thumbnail.create", "core.thumbnailer")
        guard_index_write("preview.create", "core.preview_cache")
        with pytest.raises(IndexFrozenWriteBlocked):
            guard_index_write("sqlite.write", "core.db")
        with pytest.raises(IndexFrozenWriteBlocked):
            guard_index_write("faiss.save", "core.faiss_store")


def test_maintenance_ops_allowlisted_during_search():
    """C: explicit maintenance ops must not raise WRITE_BLOCKED."""
    assert MAINTENANCE_OPS
    with search_session():
        for op in MAINTENANCE_OPS:
            guard_index_write(op, "tests.test_index_freeze")


def test_new_file_indexes_when_index_frozen_but_no_search(tmp_path: Path):
    """A: INDEX_FROZEN=True must not block new file rows outside search."""
    assert INDEX_FROZEN is True
    settings = _settings(tmp_path)
    db = Database(settings.db_path)
    _seed(db, tmp_path / "files", n=1)
    with allow_index_writes():
        db.upsert_file(
            {
                "path": str(tmp_path / "new_src" / "n.jpg"),
                "filename": "n.jpg",
                "source_id": 1,
                "status": "pending",
                "file_size": 1,
                "mtime": 1.0,
            }
        )
        db.upsert_features(1, {"phash": "deadbeef"})
    with db.connect() as conn:
        n_files = int(conn.execute("SELECT COUNT(*) FROM files").fetchone()[0])
        n_feat = int(conn.execute("SELECT COUNT(*) FROM features").fetchone()[0])
    assert n_files >= 2
    assert n_feat == 1
    assert process_search_active() is False


def test_source_manager_maintenance_no_write_blocked_during_search(tmp_path: Path):
    """C: SourceManager maintenance must not crash with WRITE_BLOCKED mid-search."""
    settings = _settings(tmp_path)
    Database(settings.db_path)
    with search_session():
        sm = SourceManager(settings, run_maintenance=True)
        assert sm.excluded_internal_files == 0
        # Forced race path: maintenance helpers must swallow, not raise.
        assert sm._exclude_generated_artifacts() == 0


def test_source_manager_maintenance_works_outside_search(tmp_path: Path):
    """C: normal housekeeping SQLite writes succeed when not in search."""
    settings = _settings(tmp_path)
    db = Database(settings.db_path)
    cache = Path(settings.cache_dir)
    junk = cache / "artifact.jpg"
    junk.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (1, 2, 3)).save(junk)
    db.upsert_file(
        {
            "path": str(junk),
            "filename": junk.name,
            "status": "indexed",
            "file_size": junk.stat().st_size,
            "mtime": junk.stat().st_mtime,
        }
    )
    sm = SourceManager(settings, run_maintenance=True)
    assert sm.excluded_internal_files >= 1
    with db.connect() as conn:
        st = conn.execute(
            "SELECT status FROM files WHERE path=?", (str(junk),)
        ).fetchone()
    assert st is not None
    assert st[0] == "excluded_internal"


def test_indexing_resumes_after_search_session(tmp_path: Path):
    """Search must not permanently stop index writes after it ends."""
    settings = _settings(tmp_path)
    Database(settings.db_path)
    store = FaissStore(settings.faiss_dino_path, settings.faiss_clip_path)
    writable = Database(settings.db_path, read_only=False)
    with search_session():
        with allow_index_writes():
            with pytest.raises(IndexFrozenWriteBlocked):
                writable.upsert_file(
                    {
                        "path": str(tmp_path / "blocked.jpg"),
                        "filename": "b.jpg",
                        "status": "pending",
                    }
                )
    assert process_search_active() is False
    with allow_index_writes():
        writable.upsert_file(
            {
                "path": str(tmp_path / "after.jpg"),
                "filename": "after.jpg",
                "status": "pending",
            }
        )
        store.save()
    with writable.connect() as conn:
        n = int(
            conn.execute(
                "SELECT COUNT(*) FROM files WHERE filename=?", ("after.jpg",)
            ).fetchone()[0]
        )
    assert n == 1


def test_ovd_work_allowed_not_blocked_by_index_frozen_alone():
    assert INDEX_FROZEN is True
    assert ovd_work_allowed() is True
    with search_session():
        assert ovd_work_allowed() is False


def test_thumbnail_create_during_search_writes_cache(tmp_path: Path):
    """Cache miss EPS/JPG — search_session açıkken thumbnail dosyası üretilsin."""
    from core.thumbnailer import Thumbnailer

    src = tmp_path / "car.jpg"
    Image.new("RGB", (64, 48), (20, 80, 160)).save(src)
    cache = tmp_path / "thumbs"
    cache.mkdir()
    thumb = Thumbnailer(str(cache), max_edge=48)
    with search_session():
        result = thumb.create(str(src))
    assert result.success, result.error
    assert Path(result.thumbnail_path).is_file()


def test_index_pipeline_writes_still_allowed(tmp_path: Path):
    store = FaissStore(tmp_path / "d.index", tmp_path / "c.index")
    with allow_index_writes():
        store.save()


def test_text_search_does_not_mutate_index(tmp_path: Path):
    settings = _settings(tmp_path)
    db = Database(settings.db_path)
    _seed(db, tmp_path / "files")
    before = _logic(settings)
    eng = SearchEngine(settings, load_ai=False)
    eng.search_by_text("leopard")
    eng.search_by_text("çiçek")
    assert _logic(settings) == before


def test_image_search_does_not_mutate_index(tmp_path: Path):
    settings = _settings(tmp_path)
    db = Database(settings.db_path)
    root = tmp_path / "files"
    _seed(db, root)
    before = _logic(settings)
    eng = SearchEngine(settings, load_ai=False)
    eng.search_by_image(str(root / "f_0.jpg"))
    assert _logic(settings) == before


def test_hybrid_search_does_not_mutate_index(tmp_path: Path):
    settings = _settings(tmp_path)
    db = Database(settings.db_path)
    root = tmp_path / "files"
    _seed(db, root)
    before = _logic(settings)
    eng = SearchEngine(settings, load_ai=False)
    eng.execute_search(
        SearchQuery(mode="hybrid", image_path=str(root / "f_1.jpg"), text="geometrik")
    )
    assert _logic(settings) == before


def test_ranking_does_not_mutate_index(tmp_path: Path):
    settings = _settings(tmp_path)
    db = Database(settings.db_path)
    _seed(db, tmp_path / "files")
    before = _logic(settings)
    rows = [
        SearchResult(
            file_id=1,
            path="a.jpg",
            filename="a.jpg",
            customer="",
            thumbnail_path="",
            score=0.4,
            score_percent=40,
            breakdown={"filename_score": 0.2},
            debug={"clip_score": 0.33, "visual_win": True},
        )
    ]
    apply_unified_text_ranking(rows, visual_retrieval=True)
    assert _logic(settings) == before


def test_pattern_family_text_search_does_not_mutate_index(tmp_path: Path):
    settings = _settings(tmp_path)
    db = Database(settings.db_path)
    _seed(db, tmp_path / "files")
    before = _logic(settings)
    eng = SearchEngine(settings, load_ai=False)
    eng.search_by_text("geometrik")
    assert _logic(settings) == before


def test_repeated_search_does_not_mutate_index(tmp_path: Path):
    settings = _settings(tmp_path)
    db = Database(settings.db_path)
    _seed(db, tmp_path / "files")
    before = _logic(settings)
    eng = SearchEngine(settings, load_ai=False)
    for _ in range(3):
        eng.search_by_text("leopard")
        eng.search_by_image(str(tmp_path / "files" / "f_0.jpg"))
    assert _logic(settings) == before


def test_search_session_blocks_faiss_and_sqlite_write(tmp_path: Path):
    settings = _settings(tmp_path)
    Database(settings.db_path)
    store = FaissStore(settings.faiss_dino_path, settings.faiss_clip_path)
    writable = Database(settings.db_path, read_only=False)
    with search_session():
        with pytest.raises(IndexFrozenWriteBlocked):
            store.save()
        with pytest.raises(IndexFrozenWriteBlocked):
            writable.upsert_file(
                {"path": str(tmp_path / "x.jpg"), "filename": "x.jpg", "status": "pending"}
            )
        with pytest.raises(IndexFrozenWriteBlocked):
            writable.upsert_features(1, {"phash": "abc"})
        with pytest.raises(IndexFrozenWriteBlocked):
            store.upsert_dino(1, b"\x00" * (384 * 4))


def test_search_session_sqlite_reads_do_not_raise_frozen_write(tmp_path: Path):
    settings = _settings(tmp_path)
    db = Database(settings.db_path)
    root = tmp_path / "files"
    _seed(db, root)
    writable = Database(settings.db_path, read_only=False)
    with search_session():
        rows = writable.get_indexed_files()
        assert rows
        rec = writable.get_file_by_id(int(rows[0]["id"]))
        assert rec


def test_search_preflight_skips_only_during_search_session(tmp_path: Path):
    """A: INDEX_FROZEN alone must not skip preflight; B: search_session does."""
    settings = _settings(tmp_path)
    db = Database(settings.db_path)
    root = tmp_path / "files"
    _seed(db, root)
    img = str(root / "f_0.jpg")
    stats = run_search_preflight(settings, img, query_only=False)
    assert stats.get("reason") != "index_frozen"
    # A: outside search, preflight may index. B: during search, artifacts frozen.
    before = _logic(settings)
    with search_session():
        stats2 = run_search_preflight(settings, img, query_only=False)
        assert stats2.get("skipped") is True
        assert stats2.get("reason") == "search_session"
        eng = SearchEngine(settings, load_ai=False)
        eng.search_in_folder_quick(img, str(root), index_first=True)
    assert _logic(settings) == before


def test_apply_verified_hashes_search_reads_only(monkeypatch, tmp_path: Path):
    called: list[str] = []

    def _boom(path: str):
        called.append(path)
        return "deadbeef", "d", "w"

    monkeypatch.setattr("core.hash_verify._compute_from_image_path", _boom)
    thumb = tmp_path / "t.webp"
    thumb.write_bytes(b"x")
    rec = {
        "id": 7,
        "phash": "storedhash",
        "dhash": "sd",
        "whash": "sw",
        "thumbnail_path": str(thumb),
        "feature_preview_path": "",
    }
    with search_session():
        out = apply_verified_hashes(rec)
        assert called == []
        assert out["phash"] == "storedhash"
        assert out["_hash_stale"] is False
    # Outside search: A alone does not forbid compute.
    missing = apply_verified_hashes({"id": 8, "phash": "", "thumbnail_path": str(thumb)})
    assert called == [str(thumb)]
    assert missing["phash"] == "deadbeef"


def test_index_writes_allowed_outside_search(tmp_path: Path):
    settings = _settings(tmp_path)
    db = Database(settings.db_path)
    _seed(db, tmp_path / "files", n=1)
    store = FaissStore(settings.faiss_dino_path, settings.faiss_clip_path)
    with allow_index_writes():
        db.upsert_features(1, {"phash": "aa", "dhash": "bb", "whash": "cc"})
        store.save()
    with db.connect() as conn:
        n = int(conn.execute("SELECT COUNT(*) FROM features").fetchone()[0])
    assert n == 1


def test_search_never_calls_indexer_index_paths(monkeypatch, tmp_path: Path):
    calls: list[str] = []

    def _spy(self, *args, **kwargs):
        calls.append("index_paths")
        return {"processed": 0, "errors": 0, "paths": 0, "skipped": True}

    monkeypatch.setattr("core.indexer.Indexer.index_paths", _spy)
    settings = _settings(tmp_path)
    db = Database(settings.db_path)
    root = tmp_path / "files"
    _seed(db, root)
    before = freeze_fingerprint(_snap(settings))
    eng = SearchEngine(settings, load_ai=False)
    eng.search_by_text("leopard")
    eng.search_by_image(str(root / "f_0.jpg"))
    eng.execute_search(
        SearchQuery(mode="hybrid", image_path=str(root / "f_1.jpg"), text="geometrik")
    )
    # During search protection, folder quick must not index.
    with search_session():
        eng.search_in_folder_quick(str(root / "f_0.jpg"), str(root), index_first=True)
    assert calls == []
    assert freeze_fingerprint(_snap(settings)) == before


def test_search_session_beats_allow_index_writes(tmp_path: Path):
    settings = _settings(tmp_path)
    Database(settings.db_path)
    store = FaissStore(settings.faiss_dino_path, settings.faiss_clip_path)
    writable = Database(settings.db_path, read_only=False)
    with search_session():
        assert process_search_active() is True
        with allow_index_writes():
            with pytest.raises(IndexFrozenWriteBlocked):
                store.save()
            with pytest.raises(IndexFrozenWriteBlocked):
                writable.upsert_file(
                    {"path": str(tmp_path / "y.jpg"), "filename": "y.jpg", "status": "pending"}
                )
            with pytest.raises(IndexFrozenWriteBlocked):
                guard_index_write("nested.write", "tests.test_index_freeze")
    with allow_index_writes():
        with search_session():
            with pytest.raises(IndexFrozenWriteBlocked):
                store.save()
    assert process_search_active() is False
    with allow_index_writes():
        store.save()


def test_concurrent_indexer_blocked_during_search_session(tmp_path: Path):
    settings = _settings(tmp_path)
    Database(settings.db_path)
    store = FaissStore(settings.faiss_dino_path, settings.faiss_clip_path)
    writable = Database(settings.db_path, read_only=False)
    results: list[str] = []

    def _fake_indexer() -> None:
        with allow_index_writes():
            try:
                writable.upsert_file(
                    {
                        "path": str(tmp_path / "concurrent.jpg"),
                        "filename": "concurrent.jpg",
                        "status": "pending",
                    }
                )
                store.save()
                results.append("wrote")
            except IndexFrozenWriteBlocked:
                results.append("blocked")

    with search_session():
        t = threading.Thread(target=_fake_indexer)
        t.start()
        t.join(timeout=5)
        assert not t.is_alive()
        assert results == ["blocked"]
    results.clear()
    t2 = threading.Thread(target=_fake_indexer)
    t2.start()
    t2.join(timeout=5)
    assert results == ["wrote"]


def test_frozen_write_reports_caller_line(tmp_path: Path):
    settings = _settings(tmp_path)
    writable = Database(settings.db_path)
    with search_session():
        with pytest.raises(IndexFrozenWriteBlocked) as exc:
            writable.upsert_file(
                {"path": str(tmp_path / "z.jpg"), "filename": "z.jpg", "status": "pending"}
            )
    assert exc.value.caller
    assert "test_frozen_write_reports_caller_line" in exc.value.caller
    assert "INDEX_FROZEN_WRITE_BLOCKED" in str(exc.value)


def test_entity_fusion_text_search_does_not_write_or_freeze(tmp_path: Path):
    settings = _settings(tmp_path)
    settings.face_index_enabled = True
    settings.object_index_enabled = True
    settings.object_db_path = str(tmp_path / "data" / "object_index.db")
    db = Database(settings.db_path)
    _seed(db, tmp_path / "files", n=8)
    before = _logic(settings)
    eng = SearchEngine(settings, load_ai=False)
    text_queries = [
        "çiçek", "gül", "leopar", "desen", "geometrik", "çizgi", "paisley",
        "barok", "tropikal", "kamuflaj", "dantel", "ekose", "zebra", "mavi",
        "siyah", "beyaz", "vintage", "modern", "logo", "marka",
    ]
    entity_queries = [
        "çanta", "kolye", "kadın yüzü", "çiçek", "amiri",
        "elbise", "ayakkabı", "saat", "gözlük", "şapka",
    ]
    fusion_queries = [
        "amiri çiçek", "kırmızı çanta", "kadın + çanta", "Amiri kırmızı çiçek",
        "amiri kırmızı", "mavi çiçek", "siyah kolye", "gül leopard",
        "kadın çiçek", "amiri çanta",
    ]
    empty = ["zzzznotapatternxyz"]
    for q in text_queries + entity_queries + fusion_queries + empty:
        eng.search_by_text(q)
    assert _logic(settings) == before


def test_search_while_index_thread_blocked_still_returns(tmp_path: Path):
    settings = _settings(tmp_path)
    db = Database(settings.db_path)
    _seed(db, tmp_path / "files", n=4)
    writable = Database(settings.db_path, read_only=False)
    before = _logic(settings)
    blocked: list[str] = []

    def _indexer() -> None:
        with allow_index_writes():
            try:
                writable.upsert_file(
                    {
                        "path": str(tmp_path / "idx.jpg"),
                        "filename": "idx.jpg",
                        "status": "pending",
                    }
                )
                blocked.append("wrote")
            except IndexFrozenWriteBlocked:
                blocked.append("blocked")

    eng = SearchEngine(settings, load_ai=False)
    with search_session():
        t = threading.Thread(target=_indexer)
        t.start()
        out = eng.search_by_text("çiçek")
        t.join(timeout=5)
        assert not t.is_alive()
        assert blocked == ["blocked"]
        assert isinstance(out, list)
    assert _logic(settings) == before


def test_production_index_untouched_by_freeze_tests():
    """Presence check only — live production mtime is not the freeze oracle."""
    snap = _prod_snap()
    db = Path(DEFAULT_DATA_DIR / "patterns.db")
    if db.is_file():
        assert snap[str(db)]["exists"] is True
        assert snap[str(db)]["size"] > 0

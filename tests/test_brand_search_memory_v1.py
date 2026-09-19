"""Brand + search memory V1: aliases, overlay, undo, freeze. No Pattern Index writes."""
from __future__ import annotations

from pathlib import Path

from core.brand_aliases import (
    register_brand_alias,
    resolve_brand_alias,
    BRAND_ALIASES,
)
from core.brand_evidence import extract_brand_evidence
from core.db import Database
from core.fuzzy_correct import correct_query
from core.index_freeze import freeze_fingerprint, snapshot_index_artifacts
from core.ocr_engine import OCREngine, ocr_to_brand_entities
from core.search_engine import SearchEngine
from core.search_memory import (
    apply_query_memory,
    memory_db_path,
    ocr_cache_put,
    ocr_cache_get,
    overlay_adjustments,
    record_feedback_overlay,
    remember_query_rewrite,
    undo_last,
)
from core.settings import AppSettings
from core.user_feedback import UserFeedbackStore


def _fp(db_path: str) -> dict:
    return freeze_fingerprint(
        snapshot_index_artifacts(
            db_path=db_path,
            faiss_dino_path=str(Path(db_path).with_name("faiss_dino.index")),
            faiss_clip_path=str(Path(db_path).with_name("faiss_clip.index")),
        )
    )


def test_existing_brand_aliases_unchanged():
    assert BRAND_ALIASES["dolce"] == "dolce gabbana"
    assert BRAND_ALIASES["amiri"] == "amiri"
    assert BRAND_ALIASES["lv"] == "louis vuitton"
    assert resolve_brand_alias("dolce") == "dolce gabbana"
    assert resolve_brand_alias("amiri") == "amiri"
    assert resolve_brand_alias("lv") == "louis vuitton"


def test_louise_resolves_to_louis_vuitton():
    assert resolve_brand_alias("louise") == "louis vuitton"


def test_leopard_and_leopar_typos():
    assert correct_query("leopard").corrected.lower() in {"leopard", "leopar"}
    leo = correct_query("leopar")
    assert leo.was_corrected is False or leo.corrected.lower() in {"leopar", "leopard"}
    fixed = correct_query("leoapr")
    assert fixed.was_corrected is True
    assert "leopard" in fixed.corrected.lower()


def test_unknown_brand_is_not_forced():
    assert resolve_brand_alias("unknownbrandxyz") is None


def test_user_added_brand_resolves_from_search_memory_not_index(tmp_path):
    db_path = str(tmp_path / "patterns.db")
    Database(db_path)
    before = _fp(db_path)
    assert register_brand_alias(db_path, "polo", "Polo Ralph Lauren") is True
    assert resolve_brand_alias("polo", db_path) == "Polo Ralph Lauren"
    mem = Path(memory_db_path(db_path))
    assert mem.is_file()
    assert _fp(db_path) == before
    rec = {"ocr_text": "POLO RALPH LAUREN", "filename": "x.jpg"}
    ev = extract_brand_evidence(rec, db_path)
    assert any("polo" in x.lower() or "ralph" in x.lower() for x in ev)


def test_typo_learn_persists_to_next_search(tmp_path):
    db_path = str(tmp_path / "patterns.db")
    Database(db_path)
    before = _fp(db_path)
    assert remember_query_rewrite(db_path, "leoapr", "leopard") is True
    rewritten, meta = apply_query_memory(db_path, "leoapr")
    assert rewritten.lower() == "leopard"
    assert meta.get("memory_rewrite")
    # new process / new helper instance
    rewritten2, _ = apply_query_memory(db_path, "leoapr")
    assert rewritten2.lower() == "leopard"
    assert _fp(db_path) == before


def test_louise_and_amiri_memory_persist(tmp_path):
    db_path = str(tmp_path / "patterns.db")
    Database(db_path)
    remember_query_rewrite(db_path, "louise", "louis vuitton", kind="brand")
    register_brand_alias(db_path, "amiri", "AMIRI")
    q, meta = apply_query_memory(db_path, "louise")
    assert "louis" in q.lower() or meta.get("memory_brand")
    assert resolve_brand_alias("amiri", db_path) == "Amiri"


def test_feedback_overlay_ranks_next_search_and_undo(tmp_path):
    db_path = str(tmp_path / "patterns.db")
    Database(db_path)
    before = _fp(db_path)
    record_feedback_overlay(db_path, "leopard", 7, "correct")
    adj = overlay_adjustments(db_path, "leopard")
    assert adj.get(7, 0) > 0
    record_feedback_overlay(db_path, "leopard", 8, "wrong")
    store = UserFeedbackStore(Database(db_path, read_only=True))
    assert 8 in store.wrong_result_ids("leopard")
    undone = undo_last(db_path)
    assert undone.get("undone") is True
    undone2 = undo_last(db_path)
    assert undone2.get("undone") is True
    assert 7 not in overlay_adjustments(db_path, "leopard")
    assert _fp(db_path) == before


def test_undo_brand_restore(tmp_path):
    db_path = str(tmp_path / "patterns.db")
    Database(db_path)
    register_brand_alias(db_path, "zzbrand", "First")
    register_brand_alias(db_path, "zzbrand", "Second")
    assert resolve_brand_alias("zzbrand", db_path) == "Second"
    undo_last(db_path)
    assert resolve_brand_alias("zzbrand", db_path) == "First"


def test_ocr_default_disabled_and_brand_entity():
    assert AppSettings().ocr_enabled is False
    ocr = OCREngine(enabled=False)
    assert ocr.extract_text("nope.jpg") == ""
    brands = ocr_to_brand_entities("LOUISE LV AMIRI DOLCE")
    joined = " ".join(brands).lower()
    assert "louis" in joined
    assert "amiri" in joined
    assert "dolce" in joined


def test_search_time_ocr_cache_is_memory_db_only(tmp_path):
    db_path = str(tmp_path / "patterns.db")
    Database(db_path)
    before = _fp(db_path)
    ocr_cache_put(db_path, "file:1", "GUCCI", confidence=0.9, brands=["gucci"])
    got = ocr_cache_get(db_path, "file:1")
    assert got and got["text"] == "GUCCI"
    assert _fp(db_path) == before


def test_search_engine_uses_memory_rewrite_without_index_write(tmp_path):
    db_path = str(tmp_path / "patterns.db")
    db = Database(db_path)
    img = tmp_path / "a.jpg"
    img.write_bytes(b"\xff\xd8\xff")
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(tmp_path)),
        )
        conn.execute(
            "INSERT INTO files(path, filename, status, ocr_text) VALUES (?,?,?,?)",
            (str(img), "a.jpg", "indexed", "leopard print"),
        )
        conn.commit()
    settings = AppSettings(
        db_path=db_path,
        cache_dir=str(tmp_path / "cache"),
        faiss_dino_path=str(tmp_path / "faiss_dino.index"),
        faiss_clip_path=str(tmp_path / "faiss_clip.index"),
        ai_embedding_enabled=False,
        ocr_enabled=False,
        face_index_enabled=False,
    )
    remember_query_rewrite(db_path, "leoapr", "leopard")
    before = _fp(db_path)
    eng = SearchEngine(settings, load_ai=False)
    eng.search_by_text("leoapr")
    eng.search_by_text("louise")
    eng.search_by_text("dolce")
    eng.search_by_text("amiri")
    eng.search_by_text("lv")
    assert _fp(db_path) == before


def test_isolated_sentinel_fingerprint_unchanged(tmp_path):
    isolated = tmp_path / "data"
    isolated.mkdir()
    sentinel = isolated / "patterns.db"
    sentinel.write_bytes(b"index-frozen-sentinel")
    faiss_d = isolated / "faiss_dino.index"
    faiss_c = isolated / "faiss_clip.index"
    faiss_d.write_bytes(b"FAISS0")
    faiss_c.write_bytes(b"FAISS0")
    before = freeze_fingerprint(
        snapshot_index_artifacts(
            db_path=str(sentinel),
            faiss_dino_path=str(faiss_d),
            faiss_clip_path=str(faiss_c),
        )
    )
    remember_query_rewrite(str(sentinel), "leoapr", "leopard")
    register_brand_alias(str(sentinel), "louise", "Louis Vuitton")
    record_feedback_overlay(str(sentinel), "leopard", 1, "correct")
    undo_last(str(sentinel))
    after = freeze_fingerprint(
        snapshot_index_artifacts(
            db_path=str(sentinel),
            faiss_dino_path=str(faiss_d),
            faiss_clip_path=str(faiss_c),
        )
    )
    assert after == before
    assert sentinel.read_bytes() == b"index-frozen-sentinel"
    assert faiss_d.read_bytes() == b"FAISS0"

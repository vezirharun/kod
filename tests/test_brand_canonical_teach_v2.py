"""Marka canonicalization + öğretilen marka arama v2."""
from __future__ import annotations

from pathlib import Path

from core.brand_aliases import brand_names, canonical_brand_display, normalize_brand_key
from core.category_memory import dynamic_children, register_category
from core.db import Database
from core.search_engine import SearchEngine
from core.search_memory import file_ids_for_taught_brand, teach_file_brand
from core.settings import AppSettings


def _insert_file(db: Database, tmp: Path, name: str) -> int:
    img = tmp / name
    img.write_bytes(b"\xff\xd8\xff")
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO files(path, filename, status, ocr_text) VALUES (?,?,?,?)",
            (str(img), name, "indexed", ""),
        )
        conn.commit()
        fid = int(conn.execute("SELECT id FROM files WHERE filename=?", (name,)).fetchone()[0])
    return fid


def _engine(db_path: str, tmp: Path) -> SearchEngine:
    settings = AppSettings(
        db_path=db_path,
        cache_dir=str(tmp / "cache"),
        faiss_dino_path=str(tmp / "faiss_dino.index"),
        faiss_clip_path=str(tmp / "faiss_clip.index"),
        ai_embedding_enabled=False,
        ocr_enabled=False,
        face_index_enabled=False,
        semantic_pattern_intel_enabled=False,
        semantic_text_search_enabled=False,
        pattern_intelligence_v2_enabled=False,
        universal_visual_intel_enabled=False,
    )
    return SearchEngine(settings, load_ai=False)


def test_canonical_display_collapses_case():
    assert canonical_brand_display("amiri") == "Amiri"
    assert canonical_brand_display("AMIRI") == "Amiri"
    assert canonical_brand_display("AmIrI") == "Amiri"
    assert canonical_brand_display("LOEWE") == canonical_brand_display("loewe")
    assert canonical_brand_display("adidas") == canonical_brand_display("Adidas")


def test_dictionary_case_variants_collapse(tmp_path):
    db_path = str(tmp_path / "patterns.db")
    Database(db_path)
    register_category(db_path, "Marka", "amiri")
    register_category(db_path, "Marka", "Amiri")
    register_category(db_path, "Marka", "AMIRI")
    kids = dynamic_children(db_path, "Marka")
    keys = [normalize_brand_key(x) for x in kids]
    assert keys.count("amiri") == 1
    assert "Amiri" in kids
    names = brand_names(db_path)
    assert sum(1 for n in names if normalize_brand_key(n) == "amiri") == 1


def test_teach_search_counts_and_case_and_idempotent(tmp_path):
    db_path = str(tmp_path / "patterns.db")
    db = Database(db_path)
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(tmp_path)),
        )
        conn.commit()
    # One filename hit (legacy 1-result) + taught files without Amiri in the name.
    named = _insert_file(db, tmp_path, "Amiri-runway.jpg")
    taught_ids = [named]
    for n in (1, 5, 10):
        while len(taught_ids) < n:
            taught_ids.append(_insert_file(db, tmp_path, f"floral_{len(taught_ids)}.jpg"))
        for fid in taught_ids:
            assert teach_file_brand(db_path, fid, "Amiri")
        assert set(file_ids_for_taught_brand(db_path, "Amiri")) >= set(taught_ids)
        teach_file_brand(db_path, taught_ids[0], "amiri")
        teach_file_brand(db_path, taught_ids[0], "AMIRI")
        assert file_ids_for_taught_brand(db_path, "Amiri").count(taught_ids[0]) == 1

        eng = _engine(db_path, tmp_path)
        sets = []
        for q in ("Amiri", "amiri", "AMIRI"):
            hits = eng.search_by_text(q, limit=0)
            ids = [r.file_id for r in hits]
            assert len(ids) == len(set(ids))
            sets.append(set(ids))
        assert sets[0] == sets[1] == sets[2]
        assert set(taught_ids).issubset(sets[0])
    assert set(taught_ids).issubset(set(file_ids_for_taught_brand(db_path, "amiri")))

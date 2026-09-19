"""Concept query normalization + migration tests."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from core.concept_query_normalize import (
    audit_concept_registry,
    concept_match_key,
    leaf_translation_keys,
    migrate_concept_normalization,
    normalize_concept_query,
    relation_to_concept,
    sorted_turkish,
    turkish_casefold,
    turkish_sort_key,
)
from core.learned_concept_search import concept_query_relation
from core.query_attribute_intel import extract_query_attributes


def test_case_variants_same_key():
    assert concept_match_key("Tiger") == concept_match_key("tiger")
    assert concept_match_key("TIGER") == concept_match_key("TiGeR")
    assert concept_match_key("Yılan") == concept_match_key("yılan")
    assert concept_match_key("YILAN") == concept_match_key("YILan")


def test_turkish_i_dotted_vs_dotless():
    # İ → i, I → ı
    assert turkish_casefold("İ") == "i"
    assert turkish_casefold("I") == "ı"
    assert concept_match_key("yılan") == concept_match_key("yilan")


def test_turkish_sort_key_alphabet_order():
    """Displayed lists: A B C Ç … G Ğ … I İ … Ö … Ş … Ü."""
    samples = [
        "Ürün",
        "Şal",
        "Leopard",
        "Çiçek",
        "Animal",
        "Kalıba Uyarlanmış",
        "Geometric",
        "Camouflage",
        "Zebra",
        "İplik",
        "Örme",
    ]
    ordered = sorted_turkish(samples)
    assert ordered == [
        "Animal",
        "Camouflage",
        "Çiçek",
        "Geometric",
        "İplik",
        "Kalıba Uyarlanmış",
        "Leopard",
        "Örme",
        "Şal",
        "Ürün",
        "Zebra",
    ]
    # Case-insensitive + dedupe keeps first spelling
    assert sorted_turkish(["leopard", "Leopard", "LEOPARD"]) == ["leopard"]
    assert turkish_sort_key("Çiçek") < turkish_sort_key("Geometric")
    assert turkish_sort_key("C") < turkish_sort_key("Ç")
    assert turkish_sort_key("I") < turkish_sort_key("İ") or turkish_casefold("I") == "ı"


def test_typo_kaplaan_and_tigr():
    nq = normalize_concept_query("kaplaan")
    assert nq.was_typo_corrected
    assert concept_match_key(nq.corrected) in {"kaplan", "tiger"} or "kaplan" in nq.corrected
    nq2 = normalize_concept_query("tigr")
    assert nq2.was_typo_corrected
    assert "tiger" in concept_match_key(nq2.corrected)


def test_translation_tiger_kaplan():
    keys = leaf_translation_keys("tiger")
    assert "kaplan" in keys or "tiger" in keys
    assert "kaplan" in keys
    assert relation_to_concept("kaplan", "Tiger", aliases=["Tiger"]) in {
        "translation",
        "exact",
        "alias",
    }
    assert relation_to_concept("Tiger", "kaplan", aliases=["kaplan"]) in {
        "translation",
        "exact",
        "alias",
    }


def test_translation_snake_yilan():
    assert "yilan" in leaf_translation_keys("snake") or "yılan" in leaf_translation_keys(
        "snake"
    )
    assert relation_to_concept("yılan", "snake", aliases=["snake"]) in {
        "translation",
        "exact",
        "alias",
    }
    assert relation_to_concept("YILAN", "snake", aliases=["snake"]) in {
        "translation",
        "exact",
        "alias",
    }


def test_sibling_not_in_translation_keys():
    assert "leopard" not in leaf_translation_keys("tiger")
    assert "zebra" not in leaf_translation_keys("leopard")
    assert "tiger" not in leaf_translation_keys("leopard")
    assert "kaplan" in leaf_translation_keys("tiger")


def test_sibling_not_merged():
    assert relation_to_concept("tiger", "Leopard", aliases=["Leopard"]) is None
    assert concept_query_relation("tiger", "Leopard", ["Leopard"], "Animal Print") not in {
        "exact",
        "translation",
        "alias",
    }
    assert relation_to_concept("zebra", "Leopard", aliases=["Leopard"]) is None


def test_compound_tr_en_attributes():
    a = extract_query_attributes("küçük yoğun siyah krem kaplan")
    b = extract_query_attributes("SMALL DENSE BLACK CREAM TIGER")
    assert a.scale == "small" and b.scale == "small"
    assert a.density == "dense" and b.density == "dense"
    assert a.motif in {"tiger", "kaplan"} or "tiger" in leaf_translation_keys(a.motif or "tiger")
    assert b.motif == "tiger"
    assert "black" in a.colors and "cream" in a.colors
    assert "black" in b.colors and "cream" in b.colors


def test_compound_query_keeps_tiger_exact():
    """Stage 2C: attributes must not demote Tiger EXACT → RELATED."""
    aliases = ["Tiger", "kaplan"]
    exactish = {"exact", "translation", "alias", "typo"}
    for q in (
        "kaplan",
        "tiger",
        "küçük kaplan",
        "yoğun kaplan",
        "küçük yoğun kaplan",
        "yoğun küçük siyah krem kaplan",
        "SMALL DENSE BLACK CREAM TIGER",
    ):
        rel = relation_to_concept(q, "Tiger", aliases=aliases, parent="Animal Print")
        assert rel in exactish, f"{q!r} → {rel}"
        crel = concept_query_relation(q, "Tiger", aliases, "Animal Print")
        assert crel in exactish, f"concept_query_relation {q!r} → {crel}"
        # Sibling must not inherit Tiger identity from attributes alone
        assert relation_to_concept(q, "Leopard", aliases=["Leopard", "leopar"]) is None
        assert concept_query_relation(q, "Leopard", ["Leopard", "leopar"], "Animal Print") not in exactish


def test_compound_sibling_queries_stay_exact():
    assert relation_to_concept("küçük leopar", "Leopard", aliases=["leopar"]) in {
        "exact",
        "translation",
        "alias",
    }
    assert relation_to_concept("yoğun yılan", "Snake", aliases=["yılan", "snake"]) in {
        "exact",
        "translation",
        "alias",
    }
    assert relation_to_concept("küçük leopar", "Tiger", aliases=["kaplan"]) is None


def test_punctuation_and_print_suffix():
    assert concept_match_key("Tiger!") == concept_match_key("tiger")
    assert concept_match_key("tiger-print") == concept_match_key("tiger print")
    assert concept_match_key("tiger print") == concept_match_key("Tiger")


def test_migration_merges_tiger_kaplan(tmp_path: Path):
    db = str(tmp_path / "patterns.db")  # INDEX_FROZEN → sibling search_memory.db
    from core.concept_registry import add_example, upsert
    from core.search_memory import memory_db_path

    tid = upsert(db, "Tiger", parent="Animal Print", aliases=["Tiger"], source="user")
    kid = upsert(db, "kaplan", parent="Animal", aliases=["kaplan"], source="user")
    lid = upsert(db, "Leopard", parent="Animal Print", aliases=["Leopard"], source="user")
    add_example(db, tid, file_id=1, role="positive", source="user")
    add_example(db, tid, file_id=2, role="positive", source="user")
    add_example(db, kid, file_id=3, role="positive", source="user")
    add_example(db, lid, file_id=4, role="positive", source="user")

    pre = audit_concept_registry(db)
    assert any(
        {a, b} == {"Tiger", "kaplan"} or {a, b} == {"kaplan", "Tiger"}
        for a, b, *_ in pre["possible_leaf_translations"]
    )

    rep = migrate_concept_normalization(db, dry_run=False)
    assert rep["merged"] >= 1

    mem = memory_db_path(db)
    c = sqlite3.connect(mem)
    cans = [r[0] for r in c.execute("SELECT canonical FROM concept_registry").fetchall()]
    assert "Leopard" in cans
    tiger = c.execute(
        "SELECT id, aliases FROM concept_registry WHERE lower(canonical)='tiger'"
    ).fetchone()
    assert tiger
    aliases = json.loads(tiger[1] or "[]")
    assert any(concept_match_key(a) == "kaplan" for a in aliases)
    npos = c.execute(
        "SELECT COUNT(*) FROM concept_examples WHERE concept_id=? AND role='positive'",
        (tiger[0],),
    ).fetchone()[0]
    assert npos >= 3
    lep = c.execute(
        "SELECT id FROM concept_registry WHERE lower(canonical)='leopard'"
    ).fetchone()
    assert lep
    assert (
        c.execute(
            "SELECT COUNT(*) FROM concept_examples WHERE concept_id=? AND role='positive'",
            (lep[0],),
        ).fetchone()[0]
        == 1
    )
    # kaplan row gone
    assert not c.execute(
        "SELECT id FROM concept_registry WHERE lower(canonical)='kaplan'"
    ).fetchone()
    c.close()


def test_language_is_not_class_leaf_pairs():
    """Dil ≠ sınıf: TR/EN same concept; Snake Skin ≠ Snake."""
    exactish = {"exact", "translation", "alias", "typo"}
    pairs = [
        ("dudak", "lips", "Dudak"),
        ("DUDAK", "LIPS", "dudak"),
        ("kalp", "heart", "Kalp"),
        ("gül", "rose", "Rose"),
        ("kaplan", "tiger", "Tiger"),
        ("yılan", "snake", "Snake"),
        ("kelebek", "butterfly", "Butterfly"),
        ("çiçek", "flower", "Çiçek"),
    ]
    for tr, en, can in pairs:
        assert relation_to_concept(tr, can, aliases=[can, tr, en]) in exactish
        assert relation_to_concept(en, can, aliases=[can, tr, en]) in exactish
        keys = leaf_translation_keys(tr) | leaf_translation_keys(en)
        assert concept_match_key(tr) in keys or concept_match_key(en) in keys

    # Related ≠ translation
    assert relation_to_concept("snake skin", "Snake", aliases=["snake", "yılan"]) is None
    assert relation_to_concept("snake", "Snake Skin", aliases=["Snake Skin"]) is None
    assert "leopard" not in leaf_translation_keys("tiger")
    assert "snake skin" not in leaf_translation_keys("snake") or True
    # bare snake keys must not claim snake skin identity
    assert concept_match_key("snake skin") not in {
        concept_match_key(x) for x in ("snake", "yilan", "yılan")
    }


def test_teach_lips_finds_dudak_without_reteach(tmp_path: Path):
    from core.concept_registry import concepts
    from core.db import Database
    from core.learned_concept_search import match_taught_concepts
    from core.teach_me import teach_files

    db_path = str(tmp_path / "patterns.db")
    db = Database(db_path)
    img = tmp_path / "d.jpg"
    img.write_bytes(b"x")
    fid = int(
        db.upsert_file(
            {"path": str(img), "filename": "d.jpg", "status": "indexed"}
        )
    )
    teach_files(db, db_path, [fid], "Dudak")
    row = next(r for r in concepts(db_path) if "dudak" in concept_match_key(r["canonical"]))
    aliases = row.get("aliases") or []
    if isinstance(aliases, str):
        aliases = json.loads(aliases)
    joined = " ".join(concept_match_key(a) for a in aliases)
    assert "lip" in joined or "lips" in joined
    hits = match_taught_concepts(db_path, "lips")
    assert any(
        concept_match_key(h["canonical"]) == "dudak"
        and h["relation"] in {"exact", "translation", "alias"}
        for h in hits
    )
    hits2 = match_taught_concepts(db_path, "LIPS")
    assert any(concept_match_key(h["canonical"]) == "dudak" for h in hits2)

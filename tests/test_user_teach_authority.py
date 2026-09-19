"""User teach outranks Object AI; learned hits enter ranking and persist."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from core.category_predictions import (
    category_predictions_from_result,
    prediction_label,
)
from core.concept_registry import concepts
from core.db import Database
from core.geometric_concepts import dna_concept_alignment, resolve_geometric_query
from core.learned_concept_search import (
    apply_learned_to_results,
    collect_learned_hits,
    resolve_learned_concept,
)
from core.search_engine import SearchResult
from core.teach_me import apply_metadata_edit_to_files, teach_files
from core.user_feedback import apply_metadata_overlay_to_result


def _db(tmp_path):
    db = Database(tmp_path / "patterns.db")
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(tmp_path)),
        )
    return db


def _add(db: Database, tmp_path, name: str, *, clip=None, family: str = "", tm=None) -> int:
    img = tmp_path / name
    img.write_bytes(b"x")
    fid = int(
        db.upsert_file(
            {
                "path": str(img),
                "filename": name,
                "source_id": 1,
                "status": "indexed",
                "pattern_family": family,
            }
        )
    )
    payload = {}
    if clip is not None:
        payload["clip_embedding"] = clip
    if tm is not None:
        payload["texture_map"] = tm
    if payload:
        db.upsert_features(fid, payload)
    return fid


def _goi_apple_result(**kwargs) -> SearchResult:
    data = dict(
        file_id=1,
        path="C:/tmp/cat.jpg",
        filename="cat.jpg",
        customer="",
        thumbnail_path="",
        score=0.8,
        score_percent=80,
        color_family="",
        pattern_family="unknown",
        animal_print_type="",
        debug={
            "texture_map": {
                "global_object_intelligence": {
                    "objects": [{"label": "elma", "label_tr": "elma", "confidence": 0.64}],
                }
            },
            "global_object_intelligence": {
                "objects": [{"label": "elma", "label_tr": "elma", "confidence": 0.64}],
            },
        },
    )
    data.update(kwargs)
    return SearchResult(**data)


def test_kedi_teach_is_not_crushed_by_object_ai():
    result = _goi_apple_result()
    apply_metadata_overlay_to_result(
        result,
        {"parent": "Animal", "child": "kedi", "category_path": "Animal/kedi"},
    )
    preds = category_predictions_from_result(result)
    assert preds
    assert preds[0]["source"] == "admin_label"
    assert "elma" not in str(preds[0].get("label") or "").lower()
    assert "kedi" in str(preds[0].get("label") or "").lower()
    assert "elma" not in prediction_label(preds).lower()


def test_object_ai_wins_only_when_user_did_not_teach():
    result = _goi_apple_result()
    preds = category_predictions_from_result(result)
    assert preds
    assert preds[0]["source"] == "global_object_ai"
    assert "elma" in str(preds[0].get("label") or "").lower()


def test_ananas_taught_examples_are_search_candidates(tmp_path):
    db = _db(tmp_path)
    path = str(tmp_path / "patterns.db")
    ids = [_add(db, tmp_path, f"an{i}.jpg") for i in range(6)]
    teach_files(db, path, ids, "Ananas", overlay={"parent": "Fruit", "child": "Ananas"})
    pack = collect_learned_hits(path, "ananas", db=db)
    assert pack
    for fid in ids:
        assert fid in pack["exact_ids"]
        assert pack["file_scores"][fid] >= 0.96


def test_zincir_neighbors_beyond_ten_examples(tmp_path):
    db = _db(tmp_path)
    path = str(tmp_path / "patterns.db")
    proto = np.array([1, 0, 0, 0], dtype=np.float32).tobytes()
    near = np.array([0.97, 0.03, 0, 0], dtype=np.float32).tobytes()
    far = np.array([0, 1, 0, 0], dtype=np.float32).tobytes()
    taught = [_add(db, tmp_path, f"z{i}.jpg", clip=proto) for i in range(10)]
    extra = _add(db, tmp_path, "lookalike_chain.jpg", clip=near)
    noise = _add(db, tmp_path, "unrelated.jpg", clip=far)
    teach_files(db, path, taught, "Zincir")
    pack = collect_learned_hits(path, "zincir", db=db)
    assert len(pack["exact_ids"]) == 10
    assert extra in pack["file_scores"]
    assert extra not in pack["exact_ids"]
    assert noise not in pack["file_scores"]
    weak = SimpleNamespace(
        file_id=taught[0], score=0.2, score_percent=20, debug={}, match_explanations=[]
    )
    missing_neighbor = SimpleNamespace(
        file_id=extra, score=0.1, score_percent=10, debug={}, match_explanations=[]
    )
    out = apply_learned_to_results([weak], pack, extra_rows=[missing_neighbor])
    ids = [int(r.file_id) for r in out]
    assert extra in ids
    assert out[0].debug.get("learned_concept_exact") is True


def test_kare_is_not_puantiye():
    assert resolve_geometric_query("kare").concept_id == "square"
    assert resolve_geometric_query("puantiye").concept_id == "polka_dot"
    square_rec = {"pattern_family": "geometric", "pattern_type": "square"}
    polka_rec = {"pattern_family": "polka_dot", "pattern_type": "polka_dot"}
    sq, _ = dna_concept_alignment("kare", square_rec, {"pattern_dna": {"motif": "square"}})
    pk, _ = dna_concept_alignment("kare", polka_rec, {"pattern_dna": {"motif": "polka_dot"}})
    assert sq >= 0.86
    assert pk < 0.5


def test_cizgi_is_not_puantiye():
    assert resolve_geometric_query("çizgi").concept_id == "stripe"
    stripe_rec = {"pattern_family": "stripe", "pattern_type": "stripe"}
    polka_rec = {"pattern_family": "polka_dot", "pattern_type": "polka_dot"}
    st, _ = dna_concept_alignment("çizgi", stripe_rec, {"pattern_dna": {"family": "stripe"}})
    pk, _ = dna_concept_alignment("çizgi", polka_rec, {"pattern_dna": {"motif": "polka_dot"}})
    assert st >= 0.86
    assert pk < 0.5


def test_ucgen_is_not_puantiye():
    assert resolve_geometric_query("üçgen").concept_id == "triangle"
    tri_rec = {"pattern_family": "geometric", "pattern_type": "triangle"}
    polka_rec = {"pattern_family": "polka_dot", "pattern_type": "polka_dot"}
    tr, _ = dna_concept_alignment("üçgen", tri_rec, {"pattern_dna": {"motif": "triangle"}})
    pk, _ = dna_concept_alignment("üçgen", polka_rec, {"pattern_dna": {"motif": "polka_dot"}})
    assert tr >= 0.86
    assert pk < 0.5


def test_akrep_uses_learned_concept_not_generic_animal(tmp_path):
    db = _db(tmp_path)
    path = str(tmp_path / "patterns.db")
    fid = _add(db, tmp_path, "scorpion.jpg", family="animal_print")
    teach_files(db, path, [fid], "Akrep", overlay={"parent": "Animal", "child": "Akrep"})
    hit = resolve_learned_concept(path, "akrep")
    assert hit
    assert str(hit.get("canonical") or "").lower() == "akrep"
    pack = collect_learned_hits(path, "akrep", db=db)
    assert fid in pack["exact_ids"]


def test_learned_concepts_survive_reopen(tmp_path):
    db = _db(tmp_path)
    path = str(tmp_path / "patterns.db")
    fid = _add(db, tmp_path, "cat.jpg")
    teach_files(db, path, [fid], "kedi", overlay={"parent": "Animal", "child": "kedi"})
    again = Database(path)
    assert any("kedi" in str(r["canonical"]).lower() for r in concepts(path))
    assert resolve_learned_concept(path, "kedi")
    pack = collect_learned_hits(path, "kedi", db=again)
    assert fid in pack["exact_ids"]


def test_taught_city_replaces_conflicting_ekose_on_card():
    from ui.designer_labels import (
        family_badge_for_result,
        family_badge_text,
        independent_feature_badge,
        taught_concept_label,
    )

    result = SearchResult(
        file_id=11,
        path="city.jpg",
        filename="city.jpg",
        customer="",
        thumbnail_path="",
        score=0.99,
        score_percent=99,
        color_family="black_white",
        pattern_family="plaid_check",
        animal_print_type="",
        debug={
            "learned_concept": True,
            "learned_concept_exact": True,
            "learned_canonical": "şehir",
            "pattern_dna": {"family": "plaid_check", "confidence": 0.8},
        },
    )
    assert taught_concept_label(result).lower().startswith("şehir") or "ehir" in taught_concept_label(result).lower()
    assert family_badge_for_result(result).casefold().startswith("şehir") or "ehir" in family_badge_for_result(result).casefold()
    assert family_badge_text(result.pattern_family) == "Ekose"
    assert independent_feature_badge(result) == ""
    preds = category_predictions_from_result(result)
    assert preds[0]["source"] == "learned_concept"
    assert "şehir" in str(preds[0]["label"]).casefold()
    assert "ekose" not in str(preds[0]["label"]).casefold()
    assert "plaid" not in str(preds[0]["label"]).casefold()
    from ui.designer_labels import color_badge_text

    assert "siyah" in color_badge_text(result.color_family).casefold()


def test_taught_keeps_independent_stripe_and_untaught_keeps_ekose():
    from ui.designer_labels import family_badge_for_result, independent_feature_badge

    taught = SearchResult(
        file_id=12,
        path="a.jpg",
        filename="a.jpg",
        customer="",
        thumbnail_path="",
        score=0.9,
        score_percent=90,
        color_family="black_white",
        pattern_family="stripe",
        debug={"learned_concept_exact": True, "learned_canonical": "şehir"},
    )
    assert "ehir" in family_badge_for_result(taught).casefold()
    assert independent_feature_badge(taught) == "Çizgi"
    plain = SearchResult(
        file_id=13,
        path="b.jpg",
        filename="b.jpg",
        customer="",
        thumbnail_path="",
        score=0.4,
        score_percent=40,
        pattern_family="plaid_check",
        debug={},
    )
    assert family_badge_for_result(plain) == "Ekose"
    preds = category_predictions_from_result(plain)
    assert not preds or preds[0].get("source") != "learned_concept"


def test_apply_learned_stamps_canonical_for_cards():
    row = SimpleNamespace(
        file_id=7, score=0.2, score_percent=20, debug={}, match_explanations=[]
    )
    out = apply_learned_to_results(
        [row],
        {"canonical": "şehir", "exact_ids": [7], "neighbor_scores": {}},
    )
    assert out[0].debug.get("learned_canonical") == "şehir"
    from ui.designer_labels import family_badge_for_result

    out[0].pattern_family = "plaid_check"
    assert "ehir" in family_badge_for_result(out[0]).casefold()


def test_edit_zebra_to_leopard_enters_same_memory(tmp_path):
    db = _db(tmp_path)
    path = str(tmp_path / "patterns.db")
    fid = _add(db, tmp_path, "z.jpg", family="zebra")
    stats = apply_metadata_edit_to_files(
        db,
        path,
        [fid],
        {
            "parent": "Animal Print",
            "child": "Leopard",
            "category_path": "Animal Print/Leopard",
            "pattern_family": "animal_print",
            "animal_print_type": "leopard",
        },
    )
    assert stats["taught"] == 1
    assert resolve_learned_concept(path, "leopard")
    pack = collect_learned_hits(path, "leopard", db=db)
    assert fid in pack["exact_ids"]


def test_unlearned_query_does_not_force_learned_pack():
    assert resolve_learned_concept("", "ananas") is None
    assert collect_learned_hits("missing.db", "ananas") == {}
    assert collect_learned_hits("missing.db", "kare") == {}

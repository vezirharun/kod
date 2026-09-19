"""Lane A (exact brand) vs lane B (conceptual) ranking + Güven."""
from __future__ import annotations

from types import SimpleNamespace

from core.confidence_engine import (
    LEVEL_LOW,
    LEVEL_VERY_HIGH,
    MATCH_BRAND,
    MATCH_EXACT,
    compute_confidence,
)
from core.fusion_query_v2 import apply_fusion_query_v2, parse_fusion_query
from core.visual_pattern_query import apply_visual_pattern_priority, parse_visual_pattern_concepts


def _row(**kwargs):
    defaults = dict(
        filename="x.jpg",
        path="",
        color_family="",
        pattern_family="",
        score=0.50,
        score_percent=50.0,
        breakdown={},
        debug={},
        file_id=0,
        is_self_match=False,
    )
    defaults.update(kwargs)
    defaults["breakdown"] = dict(defaults["breakdown"])
    defaults["debug"] = dict(defaults["debug"])
    return SimpleNamespace(**defaults)


def test_lane_split_brand_vs_conceptual():
    assert parse_fusion_query("LV").ranking_lane == "A"
    assert parse_fusion_query("Louis Vuitton").ranking_lane == "A"
    assert parse_fusion_query("Dolce").ranking_lane == "A"
    assert parse_fusion_query("Dior").ranking_lane == "A"
    assert parse_fusion_query("ekose").ranking_lane == "B"
    assert parse_fusion_query("kadın").ranking_lane == "B"
    assert parse_fusion_query("yılan").ranking_lane == "B"
    assert parse_fusion_query("çiçek").ranking_lane == "B"
    assert parse_fusion_query("Amiri ekose").ranking_lane == "hybrid"


def test_lv_brand_stays_on_top():
    lv = _row(
        filename="LV_logo.jpg",
        score=1.0,
        score_percent=100.0,
        breakdown={"brand_alias_score": 0.95, "brand_evidence_hit": 1.0},
        debug={"brand_evidence": True},
    )
    print_hit = _row(
        filename="random_print.jpg",
        pattern_family="floral",
        score=0.74,
        breakdown={"clip": 0.74},
        debug={"clip_score": 0.74},
    )
    out = apply_fusion_query_v2([print_hit, lv], "Louis Vuitton")
    assert out[0] is lv
    assert out[0].score >= 0.90
    assert print_hit.score <= 0.48


def test_dolce_generic_prints_below_brand_floor():
    dg = _row(
        filename="DG_logo.jpg",
        score=1.0,
        breakdown={"brand_alias_score": 0.94, "brand_evidence_hit": 1.0},
        debug={"brand_evidence": True},
    )
    junk = _row(
        filename="baroque_lookalike.jpg",
        pattern_family="paisley",
        score=0.72,
        breakdown={"clip": 0.72, "filename_score": 0.10},
        debug={"clip_score": 0.72},
    )
    out = apply_fusion_query_v2([junk, dg], "Dolce")
    assert out[0] is dg
    assert junk.score < 0.90
    assert junk.score <= 0.48


def test_kadin_fabric_only_below_person():
    fabric = _row(
        filename="kadin_kumas_ss24.jpg",
        pattern_family="floral",
        score=0.96,
        breakdown={"filename_score": 0.99, "ocr_score": 0.40},
        debug={"clip_score": 0.22, "text_mode": True},
    )
    person = _row(
        filename="runway_model.jpg",
        score=0.41,
        breakdown={"filename_score": 0.05},
        debug={
            "face_gender_match": True,
            "gender_visual_score": 0.88,
            "human_semantic_score": 0.88,
            "text_mode": True,
        },
    )
    out = apply_fusion_query_v2([fabric, person], "kadın")
    assert out[0] is person
    assert fabric not in out


def test_erkek_object_person_box_not_boosted_to_93():
    """COCO 'person' on a jean/texture is not erkek evidence (UI ~%93 bug)."""
    fabric = _row(
        filename="jean_texture.jpg",
        pattern_family="denim",
        score=0.26,
        breakdown={"filename_score": 0.0, "ocr_score": 0.0},
        debug={
            "clip_score": 0.27,
            "entity_evidence_kind": "object_detector",
            "entity_evidence_score": 0.97,
            "entity_query": ["person"],
            "text_mode": True,
        },
    )
    person = _row(
        filename="portrait.jpg",
        score=0.41,
        debug={
            "face_gender_match": True,
            "gender_visual_score": 0.88,
            "human_semantic_score": 1.0,
            "text_mode": True,
        },
    )
    out = apply_fusion_query_v2([fabric, person], "erkek")
    assert out[0] is person
    assert fabric.score <= 0.52
    assert fabric.score_percent < 90.0


def test_erkek_clip_fabric_not_boosted_to_95():
    fabric = _row(
        filename="bw_geo_print.jpg",
        pattern_family="geometric",
        score=0.96,
        breakdown={"clip": 0.96},
        debug={
            "clip_score": 0.96,
            "clip_only": True,
            "gender_visual_score": 0.32,
            "human_semantic_score": 0.32,
            "text_mode": True,
        },
    )
    person = _row(
        filename="portrait.jpg",
        score=0.41,
        debug={
            "face_gender_match": True,
            "gender_visual_score": 0.88,
            "human_semantic_score": 1.0,
            "text_mode": True,
        },
    )
    out = apply_fusion_query_v2([fabric, person], "erkek")
    assert out[0] is person
    assert fabric.score <= 0.52


def test_kadin_yuzu_clip_gender_score_still_counts():
    person = _row(
        filename="face.jpg",
        score=0.40,
        debug={"gender_visual_score": 0.80, "human_semantic_score": 0.80},
    )
    fabric = _row(
        filename="print.jpg",
        pattern_family="floral",
        score=0.82,
        breakdown={"clip": 0.82},
        debug={"clip_score": 0.82, "entity_evidence_kind": "openclip"},
    )
    out = apply_fusion_query_v2([fabric, person], "kadın yüzü")
    assert out[0] is person
    person = _row(
        filename="face.jpg",
        score=0.40,
        debug={"face_gender_match": True, "gender_visual_score": 0.80},
    )
    fabrics = [
        _row(
            filename=f"print_{i}.jpg",
            pattern_family="floral",
            score=0.82,
            breakdown={"clip": 0.82},
            debug={"clip_score": 0.82, "entity_evidence_kind": "openclip"},
        )
        for i in range(60)
    ]
    out = apply_fusion_query_v2(fabrics + [person], "kadın")
    assert person in out[:50]
    assert all((r.debug or {}).get("evidence_ok") for r in out[:50])
    assert all(f not in out[:50] for f in fabrics)


def test_ekose_non_plaid_cannot_outrank_or_fill_page():
    plaid = _row(
        filename="scan_04.jpg",
        pattern_family="plaid_check",
        score=0.40,
        breakdown={"family_score": 0.90, "dna_score": 0.86, "texture_score": 0.70},
        debug={"texture_map": {"pattern_dna": {"family": "plaid_check", "motif": "tartan"}}},
    )
    junk = _row(
        filename="AMIRI_lookbook.jpg",
        pattern_family="floral",
        score=0.32,
        breakdown={
            "clip": 0.32,
            "brand_alias_score": 0.90,
            "filename_score": 0.20,
        },
        debug={
            "clip_score": 0.32,
            "brand_evidence": True,
            "texture_map": {"pattern_dna": {"family": "floral"}},
        },
    )
    weak = _row(
        filename="geo.jpg",
        pattern_family="geometric",
        score=0.28,
        breakdown={"clip": 0.28},
        debug={"clip_score": 0.28, "texture_map": {"pattern_dna": {"family": "geometric"}}},
    )
    out = apply_fusion_query_v2([junk, weak, plaid], "ekose")
    assert out[0] is plaid
    assert junk not in out
    assert weak not in out
    assert all(
        (getattr(r, "pattern_family", "") == "plaid_check")
        or (r.debug or {}).get("pattern_visual_positive")
        for r in out
    )


def test_snake_not_leopard_lookalike():
    snake = _row(
        filename="python_print.jpg",
        pattern_family="animal_print",
        score=0.50,
        breakdown={"family_score": 0.88, "dna_score": 0.84},
        debug={"texture_map": {"pattern_dna": {"family": "animal_print", "animal_print_type": "snake"}},
               "animal_print_type": "snake"},
    )
    leo = _row(
        filename="leopard_spot.jpg",
        pattern_family="animal_print",
        score=0.80,
        breakdown={"family_score": 0.90, "dna_score": 0.88, "clip": 0.70},
        debug={"texture_map": {"pattern_dna": {"family": "animal_print", "animal_print_type": "leopard"}},
               "animal_print_type": "leopard"},
    )
    cons = parse_visual_pattern_concepts("yılan")
    out = apply_visual_pattern_priority([leo, snake], "yılan", concepts=cons, reject_unmatched=True)
    assert out[0] is snake
    assert leo not in out or out[0] is snake


def test_mislabeled_leopard_dna_kept_when_clip_says_snake():
    skin = _row(
        filename="print_2044.jpg",
        pattern_family="animal_print",
        score=0.44,
        breakdown={"family_score": 0.80, "dna_score": 0.70, "clip": 0.30},
        debug={
            "texture_map": {"pattern_dna": {"family": "animal_print", "animal_print_type": "leopard"}},
            "animal_print_type": "leopard",
            "clip_score": 0.30,
            "rival_clip": {"snake": 0.30, "leopard": 0.21, "zebra": 0.18},
        },
    )
    leo = _row(
        filename="true_leopard.jpg",
        pattern_family="animal_print",
        score=0.80,
        breakdown={"family_score": 0.90, "dna_score": 0.88, "clip": 0.32},
        debug={
            "texture_map": {"pattern_dna": {"family": "animal_print", "animal_print_type": "leopard"}},
            "animal_print_type": "leopard",
            "clip_score": 0.21,
            "rival_clip": {"snake": 0.21, "leopard": 0.33},
        },
    )
    cons = parse_visual_pattern_concepts("yılan")
    out = apply_visual_pattern_priority([leo, skin], "yılan", concepts=cons, reject_unmatched=True)
    assert skin in out
    assert out[0] is skin
    assert leo not in out


def test_brand_100_confidence_is_very_high():
    lv = _row(
        filename="louis.jpg",
        score=1.0,
        score_percent=100.0,
        breakdown={"brand_alias_score": 0.95, "brand_evidence_hit": 1.0, "filename_score": 0.1, "ocr_score": 0.0},
        debug={"brand_evidence": True, "text_mode": True, "fusion_query_v2": {"ranking_lane": "A", "brand": 1.0}},
    )
    conf = compute_confidence(lv, mode="text")
    assert conf.match_type == MATCH_BRAND
    assert conf.level == LEVEL_VERY_HIGH
    assert conf.level != LEVEL_LOW

    exact = _row(
        filename="same.jpg",
        score=1.0,
        score_percent=100.0,
        breakdown={"filename_score": 1.0},
        debug={"protected_exact": True, "text_mode": True},
        is_self_match=True,
    )
    c2 = compute_confidence(exact, mode="text")
    assert c2.match_type == MATCH_EXACT
    assert c2.level == LEVEL_VERY_HIGH


def test_weak_clip_stays_low_confidence():
    weak = _row(
        filename="x.jpg",
        score=0.23,
        score_percent=23.0,
        breakdown={"clip": 0.23},
        debug={"clip_score": 0.23, "text_mode": True},
    )
    conf = compute_confidence(weak, mode="text")
    assert conf.level == LEVEL_LOW
    assert conf.score_percent < 90

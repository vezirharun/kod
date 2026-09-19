"""Semantic Pattern Intelligence v1 — test-first (no index/FAISS/lifecycle)."""

from __future__ import annotations

from core.brand_aliases import brand_match_score
from core.fuzzy_correct import correct_query
from core.natural_language_query import parse_natural_query
from core.semantic_pattern_intel import (
    TIER_EXACT,
    TIER_SAME_FAMILY,
    TIER_SEMANTIC,
    TIER_VARIANT,
    TIER_VERY_SIMILAR,
    BLOCKED_STATUSES,
    collect_evidence,
    gate_result,
    hybrid_score,
    parse_pattern_intent,
    record_is_searchable,
)
from core.textile_terms import fuzzy_correct_query, query_family_hints


def test_ontology_tr_en_aliases():
    rose = parse_pattern_intent("gül")
    daisy = parse_pattern_intent("papatya")
    tulip = parse_pattern_intent("lale")
    polka = parse_pattern_intent("puantiye")
    assert rose.family == "floral" and rose.motif == "rose"
    assert daisy.family == "floral" and daisy.motif == "daisy"
    assert tulip.family == "floral" and tulip.motif == "tulip"
    assert parse_pattern_intent("daisy").motif == "daisy"
    assert parse_pattern_intent("polka dot").motif == "polka_dot"
    assert parse_pattern_intent("noktalı").motif == "polka_dot"
    assert parse_pattern_intent("dot").motif == "polka_dot"
    assert polka.motif == "polka_dot"
    assert parse_pattern_intent("çiçek").motif == "floral"
    assert parse_pattern_intent("çiçek").specificity == "parent"
    assert daisy.specificity == "leaf"
    assert rose.motif != daisy.motif


def test_lale_not_normalized_to_lace():
    corrected, was = fuzzy_correct_query("lale")
    assert "lace" not in corrected.lower()
    assert correct_query("lale").corrected.lower() == "lale"
    hints = query_family_hints("lale")
    assert hints.get("pattern_type") == "tulip"
    assert hints.get("pattern_family") == "floral"


def test_gül_and_papatya_intents_differ():
    a = parse_pattern_intent("gül")
    b = parse_pattern_intent("papatya")
    assert a.as_key() != b.as_key()
    assert a.clip_prompts[0] != b.clip_prompts[0]


def test_natural_language_components():
    q1 = parse_pattern_intent("küçük çiçekli kumaş")
    assert q1.family == "floral" and q1.scale == "small"
    q2 = parse_pattern_intent("küçük papatyalı floral")
    assert q2.motif == "daisy" and q2.scale == "small"
    q3 = parse_pattern_intent("beyaz zemin üzerine siyah puantiye")
    assert q3.motif == "polka_dot"
    assert q3.ground == "white"
    assert q3.color == "black_white" or q3.color == "black"
    q4 = parse_pattern_intent("kahverengi leopar desen")
    assert q4.motif == "leopard" and q4.family == "animal_print"
    q5 = parse_pattern_intent("çizgili geometrik desen")
    assert q5.family == "geometric"
    assert q5.structure in ("stripe", "striped", "line")
    q6 = parse_pattern_intent("lacivert zemin üzerine küçük beyaz çiçek")
    assert q6.family == "floral"
    assert q6.scale == "small"
    assert q6.ground in ("navy", "blue")


def test_nl_parser_distinct_flower_motifs():
    assert parse_natural_query("papatya").motif == "daisy"
    assert parse_natural_query("gül").motif == "rose"
    assert parse_natural_query("lale").motif == "tulip"


def test_short_brand_alias_is_weak():
    strong = brand_match_score("Louis Vuitton", ["louis vuitton scarf"])
    weak = brand_match_score("LV", ["versace baroque logo"])
    lv_file = brand_match_score("LV", ["LV-101.jpg"])
    assert strong >= 0.85
    assert weak < 0.5
    assert 0.0 < lv_file < 0.55


def test_hybrid_clip_cannot_outrank_filename():
    text_win = hybrid_score(0.92, 0.22)
    clip_only = hybrid_score(0.0, 0.34)
    assert text_win > clip_only
    assert 0.45 <= clip_only <= 0.72
    blended = hybrid_score(0.70, 0.40)
    assert blended <= 0.70 + 0.25
    assert blended >= 0.70


def test_daisy_gate_drops_generic_floral():
    intent = parse_pattern_intent("papatya")
    floral_only = {
        "id": 1,
        "status": "pending",
        "filename": "4913 AB.jpg",
        "ocr_text": "",
        "pattern_family": "floral",
        "pattern_type": "",
        "texture_map": {"pattern_family": "floral"},
        "physical_preview_ready": 1,
    }
    daisy = dict(floral_only)
    daisy["filename"] = "papatya-2.jpg"
    daisy["texture_map"] = {"pattern_family": "floral", "pattern_type": "daisy"}
    floral_named = dict(floral_only)
    floral_named["filename"] = "seamless-floral-pattern.jpg"
    ev_generic = collect_evidence(intent, floral_only, clip_score=0.0)
    ev_daisy = collect_evidence(intent, daisy, clip_score=0.0)
    ev_named_floral = collect_evidence(intent, floral_named, clip_score=0.0)
    kept_generic, tier_g, _ = gate_result(intent, ev_generic, text_score=0.78)
    kept_daisy, tier_d, _ = gate_result(intent, ev_daisy, text_score=0.86)
    kept_named, _, _ = gate_result(intent, ev_named_floral, text_score=0.86)
    assert kept_generic is False
    assert kept_named is False
    assert kept_daisy is True
    assert tier_d in (TIER_EXACT, TIER_VERY_SIMILAR)


def test_parent_flower_keeps_family():
    intent = parse_pattern_intent("çiçek")
    rec = {
        "filename": "159GI.jpg",
        "status": "pending",
        "pattern_family": "floral",
        "texture_map": {"pattern_family": "floral"},
        "physical_preview_ready": 1,
    }
    ev = collect_evidence(intent, rec, clip_score=0.0)
    kept, tier, _ = gate_result(intent, ev, text_score=0.78)
    assert kept is True
    assert tier == TIER_SAME_FAMILY


def test_polka_rejects_leopard_blob():
    intent = parse_pattern_intent("puantiye")
    leopard = {
        "filename": "spot-animal.jpg",
        "status": "indexed",
        "pattern_family": "animal_print",
        "texture_map": {
            "pattern_family": "animal_print",
            "animal_print_type": "leopard",
            "animal_score": 0.9,
        },
    }
    polka = {
        "filename": "dots.eps",
        "status": "indexed",
        "pattern_family": "polka_dot",
        "texture_map": {"pattern_family": "polka_dot", "pattern_type": "polka_dot"},
    }
    drop, _, _ = gate_result(
        intent, collect_evidence(intent, leopard, clip_score=0.12), text_score=0.70
    )
    keep, _, _ = gate_result(
        intent, collect_evidence(intent, polka, clip_score=0.0), text_score=0.80
    )
    assert drop is False
    assert keep is True


def test_leopard_not_same_as_zebra():
    leopard_q = parse_pattern_intent("leopar")
    zebra = {
        "filename": "z1.jpg",
        "status": "indexed",
        "pattern_family": "animal_print",
        "texture_map": {
            "pattern_family": "animal_print",
            "animal_print_type": "zebra",
        },
    }
    kept, _, _ = gate_result(
        leopard_q, collect_evidence(leopard_q, zebra, clip_score=0.0), text_score=0.84
    )
    assert kept is False


def test_lv_clip_only_is_semantic_similar():
    intent = parse_pattern_intent("Louis Vuitton")
    rec = {
        "filename": "4913 AB.jpg",
        "status": "pending",
        "ocr_text": "",
        "pattern_family": "monogram_logo",
        "texture_map": {"pattern_family": "monogram_logo"},
        "physical_preview_ready": 1,
    }
    ev = collect_evidence(intent, rec, clip_score=0.33)
    kept, tier, score = gate_result(intent, ev, text_score=0.0)
    assert kept is True
    assert tier == TIER_SEMANTIC
    assert score < 0.85


def test_missing_and_excluded_are_blocked():
    rec_ok = {"status": "pending", "physical_preview_ready": 1, "clip_embedding": b"x"}
    rec_miss = {"status": "missing", "physical_preview_ready": 1, "clip_embedding": b"x"}
    rec_ex = {"status": "excluded_internal", "clip_embedding": b"x"}
    assert record_is_searchable(rec_ok) is True
    assert record_is_searchable(rec_miss) is False
    assert record_is_searchable(rec_ex) is False
    assert "missing" in BLOCKED_STATUSES
    assert "excluded_internal" in BLOCKED_STATUSES


def test_low_confidence_not_padded():
    intent = parse_pattern_intent("papatya")
    scores = []
    for i in range(40):
        rec = {
            "filename": f"{i}.jpg",
            "status": "indexed",
            "pattern_family": "floral",
            "texture_map": {"pattern_family": "floral"},
        }
        kept, _, sc = gate_result(
            intent, collect_evidence(intent, rec, clip_score=0.05), text_score=0.78
        )
        if kept:
            scores.append(sc)
    assert scores == []


def test_leaf_semantic_padding_is_capped():
    from core.semantic_pattern_intel import LEAF_SEMANTIC_CAP, cap_confidence_padding

    intent = parse_pattern_intent("papatya")
    rows = [{"semantic_tier": "Semantic Similar", "clip_score": 0.40 - i * 0.001} for i in range(40)]
    rows.append({"semantic_tier": "Variant", "clip_score": 0.0})
    out = cap_confidence_padding(intent, rows)
    assert len(out) == LEAF_SEMANTIC_CAP + 1


def _animal_rec(name: str, animal_type: str = "", family: str = "animal_print") -> dict:
    return {
        "filename": name,
        "status": "indexed",
        "pattern_family": family,
        "texture_map": {
            "pattern_family": family,
            "animal_print_type": animal_type,
        },
    }


def test_filename_snake_is_supported_not_dropped():
    intent = parse_pattern_intent("yılan")
    rec = _animal_rec("snake.jpg", animal_type="")
    kept, tier, _ = gate_result(
        intent, collect_evidence(intent, rec, clip_score=0.0), text_score=0.90
    )
    assert kept is True
    assert tier == TIER_VARIANT


def test_indexed_leopard_with_snake_filename_is_conflict_keep():
    intent = parse_pattern_intent("yılan derisi")
    rec = _animal_rec("snake.jpg", animal_type="leopard")
    ev = collect_evidence(intent, rec, clip_score=0.0)
    kept, tier, _ = gate_result(intent, ev, text_score=0.84)
    assert kept is True
    assert ev.filename_hit is True
    assert ev.animal_type == "leopard"
    assert tier == TIER_VARIANT


def test_snake_clip_win_beats_leopard_rival():
    intent = parse_pattern_intent("yılan")
    rec = _animal_rec("4913.jpg", animal_type="")
    ev = collect_evidence(
        intent,
        rec,
        clip_score=0.33,
        rival_clip={"snake": 0.33, "leopard": 0.21, "zebra": 0.18, "tiger": 0.17},
    )
    kept, tier, _ = gate_result(intent, ev, text_score=0.4)
    assert kept is True
    assert ev.visual_win is True
    assert tier in (TIER_VERY_SIMILAR, TIER_EXACT, TIER_SEMANTIC)


def test_leopard_rival_rejects_snake_ranking():
    intent = parse_pattern_intent("yılan")
    rec = _animal_rec("spots.jpg", animal_type="")
    ev = collect_evidence(
        intent,
        rec,
        clip_score=0.28,
        rival_clip={"snake": 0.28, "leopard": 0.32, "zebra": 0.19, "tiger": 0.18},
    )
    kept, _, _ = gate_result(intent, ev, text_score=0.78)
    assert kept is False
    assert "leopard" in ev.negative_motifs


def test_visual_snake_win_keeps_leopard_metadata():
    intent = parse_pattern_intent("yılan")
    rec = _animal_rec("4913.jpg", animal_type="leopard")
    ev = collect_evidence(
        intent,
        rec,
        clip_score=0.33,
        rival_clip={"snake": 0.33, "leopard": 0.21, "zebra": 0.18, "tiger": 0.17},
    )
    kept, _, _ = gate_result(intent, ev, text_score=0.12)
    assert kept is True
    assert ev.visual_win is True
    assert ev.animal_type == "leopard"
    assert ev.filename_hit is False


def test_animal_print_parent_keeps_all_subtypes():
    intent = parse_pattern_intent("animal print")
    assert intent.specificity == "parent"
    leopard = _animal_rec("a.jpg", animal_type="leopard")
    snake = _animal_rec("b.jpg", animal_type="snake")
    k1, _, _ = gate_result(intent, collect_evidence(intent, leopard, clip_score=0.0), text_score=0.78)
    k2, _, _ = gate_result(intent, collect_evidence(intent, snake, clip_score=0.0), text_score=0.78)
    assert k1 is True and k2 is True


def test_floral_rival_infrastructure_exists():
    from core.semantic_pattern_intel import competing_subtype_prompts

    intent = parse_pattern_intent("gül")
    prompts = competing_subtype_prompts(intent)
    assert "rose" in prompts and "daisy" in prompts and "tulip" in prompts
    ev = collect_evidence(
        intent,
        {"filename": "x.jpg", "pattern_family": "floral", "texture_map": {"pattern_family": "floral"}},
        clip_score=0.21,
        rival_clip={"rose": 0.21, "daisy": 0.31},
    )
    kept, _, _ = gate_result(intent, ev, text_score=0.5)
    assert kept is False


def test_yaprak_is_leaf_not_parent_floral():
    intent = parse_pattern_intent("yaprak")
    assert intent.motif == "leaf"
    assert intent.family == "floral"
    assert intent.specificity == "leaf"


def test_yaprak_keeps_leaf_clip_when_indexed_as_floral():
    intent = parse_pattern_intent("yaprak")
    rec = {
        "id": 1,
        "status": "pending",
        "filename": "tropical-print.jpg",
        "ocr_text": "",
        "pattern_family": "floral",
        "pattern_type": "floral",
        "texture_map": {"pattern_family": "floral"},
        "physical_preview_ready": 1,
    }
    ev_keep = collect_evidence(
        intent,
        rec,
        clip_score=0.28,
        rival_clip={"leaf": 0.28, "rose": 0.27, "floral": 0.26, "geometric": 0.18},
    )
    keep, _, sc = gate_result(intent, ev_keep, text_score=0.12)
    assert keep is True
    assert sc >= 0.60
    ev_geo = collect_evidence(
        intent,
        rec,
        clip_score=0.28,
        rival_clip={"leaf": 0.28, "rose": 0.27, "floral": 0.26, "geometric": 0.31},
    )
    keep_geo, _, sc_geo = gate_result(intent, ev_geo, text_score=0.12)
    assert keep_geo is True
    assert sc_geo >= 0.60
    ev_mid = collect_evidence(
        intent,
        rec,
        clip_score=0.28,
        rival_clip={"leaf": 0.28, "rose": 0.32, "floral": 0.31},
    )
    keep_mid, _, sc_mid = gate_result(intent, ev_mid, text_score=0.12)
    assert keep_mid is True
    assert sc_mid >= 0.60
    ev_drop = collect_evidence(
        intent,
        rec,
        clip_score=0.24,
        rival_clip={"leaf": 0.24, "rose": 0.34, "floral": 0.33, "geometric": 0.18},
    )
    drop, _, _ = gate_result(intent, ev_drop, text_score=0.12)
    assert drop is False

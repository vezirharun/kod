"""Metin Arama Doğruluk Paketi — lanes A/B/C, CLIP ≠ detector."""
from __future__ import annotations

from types import SimpleNamespace

from core.confidence_engine import MATCH_VISUAL, compute_confidence
from core.fusion_query_v2 import apply_fusion_query_v2, last_evidence_gate_meta, parse_fusion_query
from core.index_freeze import INDEX_FROZEN
from core.query_intent_router import classify_query
from core.search_evidence_gate import (
    EMPTY_BRAND,
    classify_accuracy_lane,
    has_hard_person_evidence,
    kanit_yok_label,
)
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


def test_index_stays_frozen():
    assert INDEX_FROZEN is True


def test_lanes_pattern_object_brand():
    assert classify_accuracy_lane("leopar") == "pattern"
    assert classify_accuracy_lane("ekose") == "pattern"
    assert classify_accuracy_lane("yılan") == "pattern"
    assert classify_accuracy_lane("kadın") == "object"
    assert classify_accuracy_lane("kaplan") == "object"
    assert classify_accuracy_lane("tavşan") == "object"
    assert classify_accuracy_lane("kuş") == "object"
    assert classify_accuracy_lane("bisiklet") == "object"
    assert classify_accuracy_lane("barok") == "pattern"
    assert classify_accuracy_lane("suluboya") == "pattern"
    assert classify_accuracy_lane("etnik") == "pattern"
    assert classify_accuracy_lane("takı") == "object"
    assert classify_accuracy_lane("karga") == "object"
    assert classify_accuracy_lane("marka") == "brand"
    assert classify_accuracy_lane("LV") == "brand"
    assert parse_fusion_query("marka").brand == ""
    intent = classify_query("kaplan")
    assert intent.channel == "object"
    assert intent.value == "tiger"


def test_lane_a_leopard_dna_outranks_filename():
    dna = _row(
        filename="scan.jpg",
        pattern_family="animal_print",
        score=0.40,
        breakdown={"family_score": 0.90, "dna_score": 0.88, "filename_score": 0.0},
        debug={"texture_map": {"pattern_dna": {"family": "animal_print", "animal_print_type": "leopard"}},
               "animal_print_type": "leopard"},
    )
    name = _row(
        filename="leopar_lookbook.jpg",
        pattern_family="floral",
        score=0.92,
        breakdown={"filename_score": 0.95, "clip": 0.30},
        debug={"clip_score": 0.30, "texture_map": {"pattern_dna": {"family": "floral"}}},
    )
    out = apply_fusion_query_v2([name, dna], "leopar")
    assert out[0] is dna


def test_lane_a_ekose_plaid_outranks_filename():
    plaid = _row(
        filename="d042.jpg",
        pattern_family="plaid_check",
        score=0.45,
        breakdown={"family_score": 0.91, "dna_score": 0.86, "filename_score": 0.0},
        debug={"texture_map": {"pattern_dna": {"family": "plaid_check", "motif": "tartan"}}},
    )
    named = _row(
        filename="ekose_ss24.jpg",
        pattern_family="geometric",
        score=0.93,
        breakdown={"filename_score": 0.96},
        debug={"texture_map": {"pattern_dna": {"family": "geometric"}}},
    )
    out = apply_fusion_query_v2([named, plaid], "ekose")
    assert out[0] is plaid


def test_yilan_generic_animal_print_not_accepted():
    snake = _row(
        filename="python.jpg",
        pattern_family="animal_print",
        score=0.50,
        breakdown={"family_score": 0.88, "dna_score": 0.84},
        debug={"texture_map": {"pattern_dna": {"family": "animal_print", "animal_print_type": "snake"}},
               "animal_print_type": "snake"},
    )
    generic = _row(
        filename="texture.jpg",
        pattern_family="animal_print",
        score=0.80,
        breakdown={"family_score": 0.90, "dna_score": 0.80, "clip": 0.70},
        debug={"texture_map": {"pattern_dna": {"family": "animal_print"}}, "clip_score": 0.70},
    )
    cons = parse_visual_pattern_concepts("yılan")
    out = apply_visual_pattern_priority(
        [generic, snake], "yılan", concepts=cons, reject_unmatched=True,
    )
    assert out[0] is snake
    assert generic not in out


def test_lane_b_kadin_clip_not_in_top50():
    person = _row(
        filename="face.jpg",
        score=0.35,
        debug={"face_gender_match": True, "gender_visual_score": 0.9},
    )
    clip_rows = [
        _row(
            filename=f"kumas_{i}.jpg",
            pattern_family="floral",
            score=0.82,
            breakdown={"clip": 0.82},
            debug={"clip_score": 0.82, "entity_evidence_kind": "openclip"},
        )
        for i in range(55)
    ]
    out = apply_fusion_query_v2(clip_rows + [person], "kadın")
    assert person in out[:50]
    assert has_hard_person_evidence(person)
    assert all(c not in out[:50] for c in clip_rows)
    lone = apply_fusion_query_v2(clip_rows[:1], "kadın")
    assert clip_rows[0] not in lone
    meta = last_evidence_gate_meta()
    assert "Kanıt yok" in (meta.get("empty_state") or kanit_yok_label("kadın"))


def test_lane_b_kaplan_leopard_dna_loses_tiger_wins():
    leo = _row(
        filename="leo.jpg",
        pattern_family="animal_print",
        score=0.88,
        breakdown={"family_score": 0.92, "dna_score": 0.90, "clip": 0.80},
        debug={
            "clip_score": 0.80,
            "animal_print_type": "leopard",
            "texture_map": {"pattern_dna": {"family": "animal_print", "animal_print_type": "leopard"}},
        },
    )
    tiger = _row(
        filename="tiger_det.jpg",
        pattern_family="animal_print",
        score=0.40,
        breakdown={"clip": 0.20},
        debug={
            "animal_print_type": "tiger",
            "entity_query": ["tiger"],
            "entity_evidence_kind": "object_detector",
            "entity_evidence_score": 0.91,
            "texture_map": {"pattern_dna": {"family": "animal_print", "animal_print_type": "tiger"}},
        },
    )
    out = apply_fusion_query_v2([leo, tiger], "kaplan")
    assert tiger in out
    assert leo not in out
    assert out[0] is tiger


def test_lane_b_clip_only_objects_cannot_fill_top50():
    for q in ("tavşan", "kuş", "bisiklet"):
        clips = [
            _row(
                filename=f"{q}_{i}.jpg",
                score=0.80,
                breakdown={"clip": 0.80},
                debug={"clip_score": 0.80, "entity_evidence_kind": "openclip"},
            )
            for i in range(50)
        ]
        out = apply_fusion_query_v2(clips, q)
        assert clips[0] not in out[:50]
        assert out == []


def test_lane_c_brand_without_evidence_does_not_inject_prints():
    junk = _row(
        filename="random_floral.jpg",
        pattern_family="floral",
        score=0.77,
        breakdown={"clip": 0.77},
        debug={"clip_score": 0.77},
    )
    out = apply_fusion_query_v2([junk], "Dior")
    assert junk not in out
    assert last_evidence_gate_meta().get("empty_state") == EMPTY_BRAND

    out_m = apply_fusion_query_v2([junk], "marka")
    assert junk not in out_m
    assert last_evidence_gate_meta().get("empty_state") == EMPTY_BRAND


def test_lane_c_lv_evidence_still_tops():
    lv = _row(
        filename="LV_logo.jpg",
        score=1.0,
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


def test_kanit_yok_confidence_flags():
    fabric = _row(
        filename="print.jpg",
        score=0.82,
        score_percent=82.0,
        breakdown={"clip": 0.82, "clip_score": 0.82},
        debug={
            "clip_score": 0.82,
            "kanit_yok": True,
            "kanit_yok_label": "Kadın — Kanıt yok",
            "evidence_ok": False,
            "accuracy_lane": "object",
            "text_mode": True,
            "fusion_query_v2": {"ranking_lane": "B", "accuracy_lane": "object"},
        },
    )
    conf = compute_confidence(fabric, mode="text")
    assert conf.match_type == MATCH_VISUAL
    assert any("Kanıt yok" in w for w in conf.warnings)
    assert any("doğruluk" in r for r in conf.reasons)
    person = _row(
        filename="face.jpg",
        score=0.70,
        score_percent=70.0,
        breakdown={},
        debug={
            "face_gender_match": True,
            "gender_visual_score": 0.9,
            "evidence_ok": True,
            "fusion_query_v2": {"ranking_lane": "B", "accuracy_lane": "object", "person": 0.95},
        },
    )
    c2 = compute_confidence(person, mode="text")
    assert not any("Kanıt yok" in w for w in c2.warnings)

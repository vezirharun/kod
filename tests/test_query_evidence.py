"""Query Evidence layer — general visual slots, not per-object detectors."""

from __future__ import annotations

from core.pattern_intelligence_v2 import parse_pattern_query_v2
from core.query_evidence import (
    STATE_CONTRADICTED,
    STATE_SUPPORTED,
    STATE_UNKNOWN,
    apply_query_evidence,
    build_search_plan,
    collect_candidate_evidence,
    plan_retrieval_needles,
)
from core.semantic_pattern_intel import collect_evidence, gate_result, parse_pattern_intent
from core.text_index import text_search_score


def test_plan_red_rose_is_object_and_color():
    plan = build_search_plan("kırmızı gül")
    ids = [c.concept_id for c in plan.objects]
    colors = [c.concept_id for c in plan.attributes if c.kind == "color"]
    assert "rose" in ids
    assert any(c in colors for c in ("red_pink", "red"))
    assert any("rose" in g and any(x in g for x in colors) for g in plan.and_groups)


def test_plan_leopard_floral_is_and():
    v2q = parse_pattern_query_v2("leopar çiçek")
    plan = build_search_plan("leopar çiçek", v2q=v2q)
    ids = {c.concept_id for c in plan.objects}
    assert "leopard" in ids
    assert "floral" in ids
    assert any("leopard" in g and "floral" in g for g in plan.and_groups)


def test_retrieval_needles_include_snake_aliases():
    plan = build_search_plan("yılan")
    needles = plan_retrieval_needles(plan)
    assert "snake" in needles


def test_unknown_label_is_not_false():
    plan = build_search_plan("yılan")
    rec = {
        "filename": "snake.jpg",
        "ocr_text": "",
        "pattern_family": "",
        "texture_map": {},
    }
    report = collect_candidate_evidence(plan, rec, texture_map={})
    snake = report.concepts.get("snake")
    assert snake is not None
    assert snake.state == STATE_SUPPORTED
    assert "filename" in snake.channels


def test_filename_snake_ai_leopard_is_conflict_supported():
    plan = build_search_plan("yılan")
    rec = {
        "filename": "snake-skin.jpg",
        "ocr_text": "",
        "pattern_family": "animal_print",
        "texture_map": {
            "pattern_family": "animal_print",
            "animal_print_type": "leopard",
        },
    }
    report = collect_candidate_evidence(plan, rec, texture_map=rec["texture_map"])
    snake = report.concepts["snake"]
    assert snake.state == STATE_SUPPORTED
    assert snake.conflict_with == "leopard"
    assert report.conflict is True


def test_red_rose_outranks_non_red_rose():
    plan = build_search_plan("kırmızı gül")
    red = collect_candidate_evidence(
        plan,
        {"filename": "rose.jpg", "ocr_text": "", "texture_map": {"color_family": "red_pink"}},
        texture_map={"color_family": "red_pink"},
        breakdown={"nl_color": 0.92, "filename_score": 0.76},
    )
    blue = collect_candidate_evidence(
        plan,
        {"filename": "rose.jpg", "ocr_text": "", "texture_map": {"color_family": "blue"}},
        texture_map={"color_family": "blue"},
        breakdown={"filename_score": 0.76},
    )
    unknown = collect_candidate_evidence(
        plan,
        {"filename": "rose.jpg", "ocr_text": "", "texture_map": {}},
        texture_map={},
        breakdown={"filename_score": 0.76},
    )
    s_red, red = apply_query_evidence(0.76, plan, red)
    s_blue, blue = apply_query_evidence(0.76, plan, blue)
    s_unk, unknown = apply_query_evidence(0.76, plan, unknown)
    assert red.concepts["red_pink"].state == STATE_SUPPORTED
    assert blue.concepts["red_pink"].state == STATE_CONTRADICTED
    assert unknown.concepts["red_pink"].state == STATE_UNKNOWN
    assert s_red > s_unk > s_blue


def test_ocr_gucci_not_dropped():
    plan = build_search_plan("gucci")
    rec = {
        "filename": "5807664_x.jpg",
        "ocr_text": "GUCCIFICATION?",
        "texture_map": {},
    }
    report = collect_candidate_evidence(plan, rec, texture_map={})
    gucci = next(v for k, v in report.concepts.items() if "gucci" in k)
    assert gucci.state == STATE_SUPPORTED
    assert "ocr" in gucci.channels
    base, report = apply_query_evidence(0.70, plan, report)
    assert base >= 0.70


def test_text_search_score_applies_color_delta():
    red = {
        "filename": "stock-photo-flower-rose.jpg",
        "path": "x/stock-photo-flower-rose.jpg",
        "ocr_text": "",
        "texture_map": {"color_family": "red_pink", "pattern_family": "floral"},
        "pattern_family": "floral",
        "status": "indexed",
    }
    other = dict(red)
    other["texture_map"] = {"color_family": "blue", "pattern_family": "floral"}
    s_red, b_red, _ = text_search_score("kırmızı gül", red, semantic_enabled=True)
    s_blue, b_blue, _ = text_search_score("kırmızı gül", other, semantic_enabled=True)
    assert s_red > s_blue
    assert b_red.get("query_evidence_report")


def test_competing_zebra_without_support_still_drops():
    intent = parse_pattern_intent("leopar")
    rec = {
        "filename": "z1.jpg",
        "status": "indexed",
        "pattern_family": "animal_print",
        "texture_map": {"pattern_family": "animal_print", "animal_print_type": "zebra"},
    }
    kept, _, _ = gate_result(
        intent, collect_evidence(intent, rec, clip_score=0.0), text_score=0.84
    )
    assert kept is False


def test_visual_only_snake_metadata_leopard_is_conflict_supported():
    plan = build_search_plan("yılan")
    rec = {
        "filename": "4913.jpg",
        "ocr_text": "",
        "pattern_family": "animal_print",
        "texture_map": {
            "pattern_family": "animal_print",
            "animal_print_type": "leopard",
        },
    }
    report = collect_candidate_evidence(
        plan,
        rec,
        texture_map=rec["texture_map"],
        clip_scores={"snake": 0.33, "_similarity": 0.33, "leopard": 0.21},
    )
    assert report.concepts["snake"].state == STATE_SUPPORTED
    assert "visual" in report.concepts["snake"].channels
    assert report.visual_grade in ("visual_exact", "visual_strong")
    assert report.visual_conflict is False
    assert report.conflict is True
    assert report.visual_object >= 0.20
    base, report = apply_query_evidence(0.40, plan, report)
    assert base > 0.40


def test_filename_snake_visual_leopard_keeps_conflict():
    plan = build_search_plan("yılan")
    rec = {
        "filename": "snake-skin.jpg",
        "ocr_text": "",
        "pattern_family": "animal_print",
        "texture_map": {
            "pattern_family": "animal_print",
            "animal_print_type": "leopard",
        },
    }
    report = collect_candidate_evidence(
        plan,
        rec,
        texture_map=rec["texture_map"],
        clip_scores={"snake": 0.16, "leopard": 0.34, "_similarity": 0.16},
    )
    snake = report.concepts["snake"]
    assert snake.state == STATE_SUPPORTED
    assert snake.conflict_with == "leopard"
    assert report.visual_grade == "visual_conflict"
    assert report.visual_conflict is True
    base, report = apply_query_evidence(0.76, plan, report)
    assert report.query_evidence < 0.08
    assert base < 0.76


def test_visual_lifts_object_without_filename():
    plan = build_search_plan("yılan")
    rec = {"filename": "4913.jpg", "ocr_text": "", "texture_map": {}}
    none = collect_candidate_evidence(plan, rec, texture_map={})
    vis = collect_candidate_evidence(
        plan, rec, texture_map={}, clip_scores={"snake": 0.31, "_similarity": 0.31}
    )
    s0, none = apply_query_evidence(0.40, plan, none)
    s1, vis = apply_query_evidence(0.40, plan, vis)
    assert vis.concepts["snake"].state == STATE_SUPPORTED
    assert "visual" in vis.concepts["snake"].channels
    assert vis.visual_grade in ("visual_exact", "visual_strong")
    assert s1 > s0


def test_car_alias_uses_clip_car_key():
    plan = build_search_plan("araba")
    rec = {"filename": "4913.jpg", "ocr_text": "", "texture_map": {}}
    none = collect_candidate_evidence(plan, rec, texture_map={})
    vis = collect_candidate_evidence(
        plan, rec, texture_map={}, clip_scores={"car": 0.32, "denim": 0.18, "_similarity": 0.32}
    )
    s0, none = apply_query_evidence(0.40, plan, none)
    s1, vis = apply_query_evidence(0.40, plan, vis)
    assert "araba" in vis.concepts
    assert vis.concepts["araba"].state == STATE_SUPPORTED
    assert "visual" in vis.concepts["araba"].channels
    assert vis.visual_conflict is False
    assert s1 > s0


def test_plan_blue_togg_suv_keeps_parts_together():
    plan = build_search_plan("mavi Togg SUV")
    ids = {c.concept_id for c in plan.objects}
    colors = {c.concept_id for c in plan.attributes if c.kind == "color"}
    assert "togg" in ids
    assert "suv" in ids
    assert any(c in colors for c in ("blue", "navy"))
    assert any("togg" in g and any(x in g for x in colors) for g in plan.and_groups)

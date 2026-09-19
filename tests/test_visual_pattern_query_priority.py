"""Visual Pattern Query Priority v1 — ranking + Fusion Query token types."""
from __future__ import annotations

from types import SimpleNamespace

from core.fusion_query_v2 import apply_fusion_query_v2, parse_fusion_query
from core.query_intent_router import classify_query
from core.visual_pattern_query import (
    apply_visual_pattern_priority,
    is_visual_pattern_query,
    parse_visual_pattern_concepts,
)


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
    )
    defaults.update(kwargs)
    defaults["breakdown"] = dict(defaults["breakdown"])
    defaults["debug"] = dict(defaults["debug"])
    return SimpleNamespace(**defaults)


def test_fusion_token_type_table():
    cases = {
        "Amiri": ["brand"],
        "kırmızı": ["color"],
        "ekose": ["pattern"],
        "çiçek": ["motif"],
        "çanta": ["object"],
        "kadın": ["person"],
        "Amiri ekose": ["brand", "pattern"],
        "kırmızı çanta": ["color", "object"],
        "kadın çanta çiçek": ["person", "object", "motif"],
    }
    for q, types in cases.items():
        plan = parse_fusion_query(q)
        assert plan.token_types == types, f"{q}: {plan.token_types} != {types}"


def test_pattern_lexicon_queries_classified():
    for q in (
        "ekose", "kareli", "çizgili", "çiçek", "leopar", "zebra",
        "puantiye", "geometrik", "yılan", "Amiri ekose", "kırmızı ekose", "ekose çanta",
    ):
        assert is_visual_pattern_query(q), q
    assert not is_visual_pattern_query("Amiri")
    intent = classify_query("ekose")
    assert intent.channel == "pattern"
    assert any(c.value == "tartan" for c in intent.channels)


def test_visual_dna_outranks_filename_ekose():
    visual = _row(
        filename="design_042.jpg",
        pattern_family="plaid_check",
        score=0.55,
        breakdown={"family_score": 0.91, "dna_score": 0.88, "filename_score": 0.0, "dino": 0.70, "clip": 0.48},
        debug={
            "texture_map": {"pattern_dna": {"family": "plaid_check", "motif": "tartan"}},
            "dino_score": 0.70,
            "clip_score": 0.48,
        },
    )
    text_only = _row(
        filename="LOEWE_ekose_ss24.jpg",
        pattern_family="floral",
        score=0.94,
        breakdown={
            "filename_score": 0.95,
            "ocr_score": 0.92,
            "brand_alias_score": 0.90,
            "family_score": 0.12,
            "dna_score": 0.10,
        },
        debug={"texture_map": {"pattern_dna": {"family": "floral", "motif": "flower"}}},
    )
    ranked = apply_fusion_query_v2([text_only, visual], "ekose")
    assert ranked[0] is visual
    assert text_only not in ranked or ranked[0].debug.get("pattern_visual_positive") is True
    if text_only in ranked:
        assert ranked[1].debug.get("ekose_visual_negative") is True
        assert ranked[1].score <= 0.42
    else:
        assert text_only.debug.get("pattern_visual_accepted") is False


def test_visual_negative_cap_standalone():
    cons = parse_visual_pattern_concepts("ekose")
    miss = _row(
        filename="ekose_folder/loewe.jpg",
        pattern_family="floral",
        score=0.99,
        breakdown={"filename_score": 0.99, "ocr_score": 0.95},
        debug={"texture_map": {"pattern_dna": {"family": "floral"}}},
    )
    hit = _row(
        filename="plain_scan.jpg",
        pattern_family="plaid_check",
        score=0.40,
        breakdown={"family_score": 0.86, "dna_score": 0.80, "texture_score": 0.70},
        debug={"texture_map": {"pattern_dna": {"family": "plaid_check"}}},
    )
    out = apply_visual_pattern_priority(
        [miss, hit], "ekose", concepts=cons, reject_unmatched=True
    )
    assert out[0] is hit
    assert miss not in out
    assert miss.debug["pattern_visual_negative"] is True
    assert miss.debug["pattern_visual_accepted"] is False


def test_amiri_brand_only_still_boosts():
    a = _row(filename="amiri_1.jpg", score=0.91, breakdown={"brand_alias_score": 0.95})
    b = _row(filename="other.jpg", score=0.40)
    out = apply_fusion_query_v2([a, b], "Amiri")
    assert out[0] is a
    assert out[0].score == 0.91
    assert parse_fusion_query("Amiri").token_types == ["brand"]
    assert not parse_fusion_query("Amiri").patterns


def test_amiri_ekose_additive_both_signals_win():
    both = _row(
        filename="amiri_runway.jpg",
        pattern_family="plaid_check",
        score=0.60,
        breakdown={"brand_alias_score": 0.96, "family_score": 0.90, "dna_score": 0.85, "clip": 0.5},
        debug={
            "brand_evidence": True,
            "texture_map": {"pattern_dna": {"family": "plaid_check", "motif": "tartan"}},
        },
    )
    brand_only = _row(
        filename="amiri_ekose_lookbook.jpg",
        pattern_family="floral",
        score=0.88,
        breakdown={"brand_alias_score": 0.96, "filename_score": 0.94},
        debug={
            "brand_evidence": True,
            "texture_map": {"pattern_dna": {"family": "floral"}},
        },
    )
    pattern_only = _row(
        filename="unknown_plaid.jpg",
        pattern_family="plaid_check",
        score=0.58,
        breakdown={"family_score": 0.88, "dna_score": 0.84},
        debug={"texture_map": {"pattern_dna": {"family": "plaid_check"}}},
    )
    ranked = apply_fusion_query_v2([brand_only, pattern_only, both], "Amiri ekose")
    assert ranked[0] is both
    assert brand_only in ranked and pattern_only in ranked
    plan = parse_fusion_query("Amiri ekose")
    assert plan.brand and plan.patterns
    assert plan.multi_channel


def test_color_and_object_do_not_replace_pattern_channel():
    q = parse_fusion_query("kırmızı ekose")
    assert "red" in q.colors and "ekose" in q.patterns
    q2 = parse_fusion_query("ekose çanta")
    assert "ekose" in q2.patterns
    assert q2.entities or "object" in q2.token_types
    assert q2.token_types[0] == "pattern"


def test_index_frozen_not_swallowed_in_visual_module():
    src = __import__("inspect").getsource(
        __import__("core.visual_pattern_query", fromlist=["x"])
    )
    assert "except Exception" not in src or "INDEX_FROZEN" not in src
    from core.index_freeze import INDEX_FROZEN

    assert INDEX_FROZEN is True

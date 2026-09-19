"""Fusion Query Intelligence v2 — kanal ayrıştırma ve kontrollü rerank."""
from __future__ import annotations

from types import SimpleNamespace

from core.fusion_query_v2 import apply_fusion_query_v2, parse_fusion_query, score_fusion_result
from core.index_freeze import INDEX_FROZEN


def test_parse_example_queries_not_hardcoded_seven():
    amiri = parse_fusion_query("Amiri")
    assert amiri.brand
    assert "amiri" in amiri.brand.lower()
    assert not amiri.entities
    assert not amiri.multi_channel

    q = parse_fusion_query("Amiri çiçek")
    assert q.brand and "flower" in q.motifs
    assert "amiri" not in q.entities
    assert "amiri" not in q.motifs
    assert q.multi_channel

    q2 = parse_fusion_query("Amiri kırmızı çiçek")
    assert q2.brand and "red" in q2.colors and "flower" in q2.motifs

    q3 = parse_fusion_query("kırmızı çanta")
    assert "red" in q3.colors and "handbag" in q3.entities
    assert not q3.brand

    q4 = parse_fusion_query("kadın çanta")
    assert "person" in q4.entities and "handbag" in q4.entities

    q5 = parse_fusion_query("kadın yüzü kolye")
    ids = set(q5.entities)
    assert "necklace" in ids
    assert "face" in ids or "person" in ids

    q6 = parse_fusion_query("Amiri çantalı kadın çiçek")
    assert q6.brand
    assert "handbag" in q6.entities and "person" in q6.entities
    assert "flower" in q6.motifs
    assert "amiri" not in q6.entities


def test_color_channel_works_even_if_global_color_ignore():
    plan = parse_fusion_query("kırmızı çanta")
    hit = SimpleNamespace(
        filename="red_bag.jpg",
        color_family="red",
        pattern_family="",
        score=0.40,
        breakdown={"clip": 0.40, "dino": 0.30},
        debug={"entity_query": ["handbag"], "entity_evidence_kind": "object_detector",
               "entity_evidence_score": 0.94, "texture_map": {"pattern_dna": {"color_family": "red"}}},
    )
    miss = SimpleNamespace(
        filename="blue.jpg",
        color_family="blue",
        pattern_family="",
        score=0.40,
        breakdown={"clip": 0.40},
        debug={"entity_query": ["handbag"], "entity_evidence_kind": "openclip",
               "entity_semantic_score": 0.50, "texture_map": {}},
    )
    feat = score_fusion_result(hit, plan)
    feat2 = score_fusion_result(miss, plan)
    assert feat["color"] >= 0.5
    assert feat["entity_kind"] == "object_detector"
    assert feat2["clip_as_detector"] is True
    ranked = apply_fusion_query_v2([miss, hit], "kırmızı çanta")
    assert ranked[0] is hit


def test_brand_only_does_not_rerank():
    a = SimpleNamespace(filename="amiri_1.jpg", score=0.91, color_family="", pattern_family="",
                        breakdown={"brand_alias_score": 0.95}, debug={})
    b = SimpleNamespace(filename="other.jpg", score=0.40, color_family="", pattern_family="",
                        breakdown={}, debug={})
    out = apply_fusion_query_v2([a, b], "Amiri")
    assert out[0] is a
    assert out[0].score == 0.91


def test_and_coverage_and_second_tier():
    plan = parse_fusion_query("Amiri kırmızı çiçek")
    full = SimpleNamespace(
        filename="AMIRI floral red.jpg",
        color_family="red",
        pattern_family="floral",
        score=0.70,
        breakdown={"brand_alias_score": 0.96, "clip": 0.5, "dino": 0.4},
        debug={"texture_map": {"pattern_dna": {"family": "floral", "color_family": "red"}}},
    )
    feat = score_fusion_result(full, plan)
    assert feat["brand"] >= 0.5
    assert feat["color"] >= 0.5
    assert feat["motif"] >= 0.5
    assert feat["full_and"] is True
    weak = SimpleNamespace(
        filename="random.jpg",
        color_family="",
        pattern_family="",
        score=0.70,
        breakdown={"clip": 0.7},
        debug={"texture_map": {}},
    )
    ranked = apply_fusion_query_v2([weak, full], "Amiri kırmızı çiçek")
    assert ranked[0] is full


def test_missing_channel_does_not_drop_result():
    rows = [
        SimpleNamespace(
            filename="bag.jpg",
            color_family="",
            pattern_family="",
            score=0.55,
            breakdown={"clip": 0.55},
            debug={"entity_query": ["handbag", "person"], "entity_evidence_kind": "openclip",
                   "entity_semantic_score": 0.4, "texture_map": {}},
        )
    ]
    out = apply_fusion_query_v2(rows, "kadın çanta")
    assert len(out) == 1
    assert out[0].debug["fusion_query_v2"]["second_tier"] is True


def test_red_car_does_not_and_on_rival_object_scene():
    mixed = SimpleNamespace(
        filename="scene.jpg",
        color_family="red_pink",
        pattern_family="",
        score=0.80,
        breakdown={"clip": 0.8},
        debug={
            "entity_query": ["car"],
            "entity_evidence_kind": "object_detector",
            "entity_evidence_score": 0.9,
            "object_index_hit": True,
            "texture_map": {
                "global_object_intelligence": {
                    "objects": [{"label": "car"}, {"label": "flower"}],
                },
                "color_index": {"dominant": "red"},
            },
        },
    )
    feat = score_fusion_result(mixed, parse_fusion_query("kırmızı araba"))
    from core.fusion_query_v2 import _color_object_aligned

    plan = parse_fusion_query("kırmızı araba")
    assert plan.colors and "car" in plan.entities
    assert _color_object_aligned(plan, mixed, {**feat, "color": 0.9, "entity_kind": "object_detector"}) is False


def test_search_does_not_unfreeze_index():
    assert INDEX_FROZEN is True
    src = __import__("inspect").getsource(__import__("core.fusion_query_v2", fromlist=["x"]))
    assert "upsert" not in src
    assert "FaissStore" not in src

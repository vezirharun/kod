"""Pattern Intelligence v2 fixtures — query-time cards, no index writes."""

from __future__ import annotations

from core.pattern_intelligence_v2 import (
    REP_PHOTO,
    REP_TEXTILE,
    build_pattern_card,
    composite_search_score,
    parse_pattern_query_v2,
)
from core.semantic_pattern_intel import TIER_VERY_SIMILAR, collect_evidence, gate_result, parse_pattern_intent


def test_nl_parse_components():
    q = parse_pattern_query_v2("küçük papatyalı kumaş")
    assert "daisy" in q.motifs and q.scale == "small" and q.representation == REP_TEXTILE
    q2 = parse_pattern_query_v2("leopar üzerine gül")
    assert "leopard" in q2.required and "rose" in q2.required
    assert q2.composition == "composite"
    assert q2.relationship == "overlay"
    assert q2.base_hint == "leopard"
    assert q2.overlay_hint == "rose"
    q3 = parse_pattern_query_v2("fareli kumaş")
    assert "mouse" in q3.objects or "mouse" in q3.required
    q4 = parse_pattern_query_v2("leopar ve yılan karışık desen")
    assert q4.composition == "composite"
    assert "leopard" in q4.required and "snake" in q4.required
    q5 = parse_pattern_query_v2("lacivert zeminde küçük beyaz çiçek")
    assert q5.ground in ("navy", "blue") and q5.scale == "small"


def test_filename_is_not_object_evidence():
    q = parse_pattern_query_v2("fare")
    card = build_pattern_card(q, clip_scores={}, filename_hits=["mouse"])
    assert card.objects == []
    assert card.filename_only is True
    keep, _, _, reason = composite_search_score(q, card, clip_scores={}, v11_score=0.0)
    assert keep is False
    assert "no_visual" in reason or "missing" in reason


def test_pure_animal_fixtures():
    for qtext, motif in (("leopar", "leopard"), ("yılan", "snake"), ("zebra", "zebra")):
        q = parse_pattern_query_v2(qtext)
        scores = {motif: 0.33, "floral": 0.12, "snake": 0.11 if motif != "snake" else 0.33}
        if motif != "leopard":
            scores["leopard"] = 0.14
        card = build_pattern_card(
            q,
            clip_scores=scores,
            rep_scores={REP_TEXTILE: 0.31, REP_PHOTO: 0.18},
        )
        keep, score, bucket, _ = composite_search_score(q, card, clip_scores=scores, v11_score=0.8)
        assert keep is True
        assert bucket == 0
        assert card.textile_pattern is True
        assert card.representation == REP_TEXTILE
        assert card.composition == "single"


def test_photo_not_textile():
    q = parse_pattern_query_v2("leopar")
    scores = {"leopard": 0.34}
    card = build_pattern_card(q, clip_scores=scores, rep_scores={REP_PHOTO: 0.36, REP_TEXTILE: 0.16})
    keep, _, _, reason = composite_search_score(q, card, clip_scores=scores, v11_score=0.7)
    assert keep is False
    assert "photo" in reason


def test_query_real_photo_vs_textile():
    q = parse_pattern_query_v2("gerçek leopar fotoğrafı")
    assert q.representation == REP_PHOTO
    card_ok = build_pattern_card(
        q, clip_scores={"leopard": 0.33}, rep_scores={REP_PHOTO: 0.35, REP_TEXTILE: 0.12}
    )
    keep, _, _, _ = composite_search_score(q, card_ok, clip_scores={"leopard": 0.33})
    assert keep is True
    card_bad = build_pattern_card(
        q, clip_scores={"leopard": 0.33}, rep_scores={REP_TEXTILE: 0.34, REP_PHOTO: 0.12}
    )
    keep2, _, _, _ = composite_search_score(q, card_bad, clip_scores={"leopard": 0.33})
    assert keep2 is False


def test_composite_fixtures():
    cases = [
        ("leopar gül", {"leopard": 0.32, "rose": 0.30}, ["leopard", "rose"]),
        ("leopar çiçek", {"leopard": 0.32, "floral": 0.29}, ["leopard", "floral"]),
        ("leopar yılan", {"leopard": 0.31, "snake": 0.30}, ["leopard", "snake"]),
        ("kuş çiçek", {"bird": 0.30, "floral": 0.31}, ["bird", "floral"]),
        ("fare çiçek", {"mouse": 0.31, "floral": 0.28}, ["mouse", "floral"]),
        ("kelebek çiçek", {"butterfly": 0.30, "floral": 0.29}, ["butterfly", "floral"]),
        ("yılan çiçek", {"snake": 0.30, "floral": 0.28}, ["snake", "floral"]),
    ]
    for text, scores, need in cases:
        q = parse_pattern_query_v2(text)
        for n in need:
            assert n in q.required, (text, q.required)
        card = build_pattern_card(q, clip_scores=scores, rep_scores={REP_TEXTILE: 0.3})
        keep, score, bucket, reason = composite_search_score(q, card, clip_scores=scores)
        assert keep is True, (text, reason)
        assert bucket == 0
        assert card.composition == "composite"


def test_pure_leopard_ranks_below_leopard_floral_composite():
    q = parse_pattern_query_v2("leopar çiçek")
    card = build_pattern_card(q, clip_scores={"leopard": 0.34, "floral": 0.10}, rep_scores={REP_TEXTILE: 0.3})
    keep, _, bucket, reason = composite_search_score(
        q, card, clip_scores={"leopard": 0.34, "floral": 0.10}
    )
    assert keep is False
    assert bucket == 9
    assert "missing_component" in reason
    card_full = build_pattern_card(
        q, clip_scores={"leopard": 0.32, "floral": 0.29}, rep_scores={REP_TEXTILE: 0.3}
    )
    keep2, _, bucket2, _ = composite_search_score(
        q, card_full, clip_scores={"leopard": 0.32, "floral": 0.29}
    )
    assert keep2 is True
    assert bucket2 == 0
    assert bucket2 < bucket


def test_rose_not_inferred_from_floral_family():
    q = parse_pattern_query_v2("gül")
    card = build_pattern_card(q, clip_scores={"floral": 0.33, "rose": 0.11}, rep_scores={REP_TEXTILE: 0.3})
    keep, _, _, reason = composite_search_score(q, card, clip_scores={"floral": 0.33, "rose": 0.11})
    assert keep is False or (card.motifs and card.motifs[0]["id"] != "rose")
    rose_hits = [m for m in card.motifs if m["id"] == "rose"]
    assert rose_hits == []


def test_geometric_floral_logo_composites():
    q = parse_pattern_query_v2("çiçek geometrik")
    card = build_pattern_card(
        q, clip_scores={"floral": 0.3, "geometric": 0.29}, rep_scores={REP_TEXTILE: 0.28}
    )
    keep, _, bucket, _ = composite_search_score(q, card, clip_scores={"floral": 0.3, "geometric": 0.29})
    assert keep and bucket == 0
    q2 = parse_pattern_query_v2("logo çiçek")
    card2 = build_pattern_card(q2, clip_scores={"logo": 0.3, "floral": 0.28})
    keep2, _, _, _ = composite_search_score(q2, card2, clip_scores={"logo": 0.3, "floral": 0.28})
    assert keep2 is True


def test_v11_snake_gate_still_rejects_leopard():
    intent = parse_pattern_intent("yılan")
    rec = {
        "filename": "x.jpg",
        "status": "indexed",
        "pattern_family": "animal_print",
        "texture_map": {"pattern_family": "animal_print"},
    }
    ev = collect_evidence(
        intent,
        rec,
        clip_score=0.28,
        rival_clip={"snake": 0.28, "leopard": 0.33, "zebra": 0.1, "tiger": 0.1},
    )
    kept, _, _ = gate_result(intent, ev, text_score=0.7)
    assert kept is False


def test_overlay_not_invented_without_language():
    q = parse_pattern_query_v2("leopar gül")
    card = build_pattern_card(q, clip_scores={"leopard": 0.32, "rose": 0.30})
    assert card.relationship == "unknown"
    assert card.composition == "composite"
    q2 = parse_pattern_query_v2("leopar üzerine gül")
    card2 = build_pattern_card(q2, clip_scores={"leopard": 0.32, "rose": 0.30})
    assert card2.relationship == "overlay"
    assert card2.base_motif == "leopard"
    assert "rose" in card2.overlay_motifs


def test_v11_very_similar_snake_still_kept_through_v2():
    q = parse_pattern_query_v2("yılan")
    card = build_pattern_card(q, clip_scores={"snake": 0.31, "leopard": 0.18})
    keep, score, bucket, _ = composite_search_score(q, card, clip_scores={"snake": 0.31, "leopard": 0.18}, v11_score=0.82)
    assert keep is True and bucket == 0
    assert TIER_VERY_SIMILAR

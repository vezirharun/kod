
from core.pattern_intelligence_v2 import (
    build_pattern_card,
    composite_search_score,
    parse_pattern_query_v2,
)
from core.query_evidence import build_search_plan, collect_candidate_evidence, apply_query_evidence


def test_composite_query_requires_all_individual_concepts():
    q = parse_pattern_query_v2("gül leopard")
    assert q.composition == "composite"
    assert set(q.required) >= {"rose", "leopard"}

    card = build_pattern_card(q, clip_scores={"leopard": 0.80, "rose": 0.05})
    keep, _, bucket, reason = composite_search_score(
        q, card, clip_scores={"leopard": 0.80, "rose": 0.05}
    )
    # Saf visual candidate generation has no text/base score; incomplete
    # candidates are still rejected at this stage. Real text candidates with
    # a base score are retained and Faz 3 ranks them below full matches.
    assert keep is False
    assert bucket == 9
    assert "missing_component" in reason


def test_composite_query_accepts_same_record_with_both_concepts():
    q = parse_pattern_query_v2("gül leopard")
    card = build_pattern_card(q, clip_scores={"leopard": 0.82, "rose": 0.74, "_composite": 0.78})
    keep, score, bucket, reason = composite_search_score(
        q,
        card,
        clip_scores={"leopard": 0.82, "rose": 0.74, "_composite": 0.78},
        channel_present={"leopard": True, "rose": True},
    )
    assert keep is True
    assert bucket == 0
    assert reason == "composite_match"
    assert score > 0.0


def test_composite_visual_does_not_fake_individual_concept_support():
    q = parse_pattern_query_v2("gül leopard")
    plan = build_search_plan("gül leopard", v2q=q)
    rec = {
        "filename": "leopard.tif",
        "ocr_text": "",
        "category_path": "",
        "semantic_tags": {"motif": "leopard"},
        "pattern_family": "animal_print",
        "animal_print_type": "leopard",
        "texture_map": {},
    }
    report = collect_candidate_evidence(
        plan, rec, clip_scores={"leopard": 0.82, "_composite": 0.88}
    )
    # Composite prompt is relation evidence, not proof that rose exists.
    assert report.visual_composite == 0.88
    assert report.concepts["rose"].state != "SUPPORTED"

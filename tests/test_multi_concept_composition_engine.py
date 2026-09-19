from types import SimpleNamespace

from core.composite_ranker import composite_rank_score
from core.multi_concept_composition import MultiConceptCompositionEngine
from core.pattern_intelligence_v2 import parse_pattern_query_v2
from core.query_evidence import build_search_plan


def row(score=.80, *, q=.70, comp=.80, visual=.80, concepts=None, clips=None, extra=None):
    return SimpleNamespace(
        score=score,
        score_percent=score * 100,
        debug={
            "query_evidence_report": {
                "query_evidence": q,
                "composite_evidence": comp,
                "visual_composite": visual,
                "concepts": concepts or {},
            },
            "v2_clip": clips or {},
            **(extra or {}),
        },
    )


def supported(names, *, signals=None):
    signals = signals or {}
    return {
        n: {
            "state": "SUPPORTED",
            "score": signals.get(n, .70),
            "channels": {"visual": signals.get(n, .70)},
        }
        for n in names
    }


def score_for(required, *, concepts=None, comp=.82, clips=None, extra=None):
    r = row(concepts=concepts or supported(required), comp=comp, visual=comp, clips=clips, extra=extra)
    return composite_rank_score(r, query_is_composite=len(required) >= 2, required=required)


def test_engine_supports_2_3_4_5_concepts_with_one_path():
    for required in [
        ["rose", "leopard"],
        ["rose", "leopard", "leaf"],
        ["rose", "leopard", "leaf", "floral"],
        ["rose", "leopard", "leaf", "floral", "geometric"],
    ]:
        score, features = score_for(required)
        assert score > .50
        assert features["concept_count"] == len(required)
        assert len(features["pairwise_relationships"]) == len(required) * (len(required) - 1) // 2
        assert features["coverage"] == 1.0


def test_engine_reports_partial_missing_and_never_full_coverage():
    concepts = supported(["leopard", "rose"])
    concepts["leaf"] = {"state": "UNKNOWN", "score": 0.0, "channels": {}}
    score, features = score_for(["leopard", "rose", "leaf"], concepts=concepts)
    assert features["coverage"] == round(2 / 3, 4)
    assert "leaf" in features["missing_concepts"]
    assert score < .80


def test_same_concepts_but_weak_composition_are_capped():
    score, features = score_for(["rose", "leopard", "leaf"], comp=.20)
    assert features["same_composition"] == .20
    assert features["group_relationship"]["score"] < .60
    assert score <= .60


def test_relative_contribution_keeps_dominant_and_tiny_concepts_separate():
    concepts = supported(["leopard", "rose", "leaf"], signals={
        "leopard": .85,
        "rose": .10,
        "leaf": .05,
    })
    _, features = score_for(["leopard", "rose", "leaf"], concepts=concepts)
    rel = features["relative_contribution"]
    assert rel["leopard"] > rel["rose"] > rel["leaf"]
    assert features["coverage"] == 1.0


def test_query_order_and_duplicate_concepts_do_not_change_score():
    concepts = supported(["leopard", "rose", "leaf"])
    a, fa = score_for(["leopard", "rose", "leaf"], concepts=concepts)
    b, fb = score_for(["leaf", "rose", "leopard", "rose"], concepts=concepts)
    assert abs(a - b) < 1e-9
    assert set(fa["required"]) == set(fb["required"])
    assert fb["concept_count"] == 3


def test_synonym_parser_feeds_same_required_set():
    q = parse_pattern_query_v2("gül leopar yaprak")
    plan = build_search_plan("gül leopar yaprak", v2q=q)
    required = [c.concept_id for c in plan.objects if c.required]
    assert {"rose", "leopard", "leaf"}.issubset(set(required))


def test_single_search_bypasses_multi_engine():
    r = row(score=.73, concepts=supported(["leopard"]), clips={"_composite": .99})
    score, features = composite_rank_score(r, query_is_composite=False, required=["leopard"])
    assert score == .73
    assert features["query_is_composite"] is False


def test_standalone_engine_output_schema_contains_required_debug_fields():
    out = MultiConceptCompositionEngine().evaluate(
        required=["rose", "leopard", "leaf"],
        signals=[.7, .8, .3],
        supported=[True, True, False],
        soft_weights=[1.0, 1.0, .5],
        relationship_seed=.75,
        patch=.6,
        semantic=.5,
        visual_embedding=.8,
        pattern_dna=.4,
        texture=.3,
    )
    assert "concept_evidence" in out
    assert "pairwise_relationships" in out
    assert "group_relationship" in out
    assert "same_composition_score" in out
    assert "composition_confidence" in out
    assert "final_multi_concept_score" in out

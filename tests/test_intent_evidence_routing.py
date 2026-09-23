"""Query intent → visual evidence routing tests (no fake engines)."""

from __future__ import annotations

from types import SimpleNamespace

from core.intent_evidence_routing import (
    apply_intent_evidence_routing,
    build_intent_evidence_plan,
    collect_candidate_evidence,
    should_suppress_pattern_first,
)


def test_intent_matrix_expected_classes():
    cases = [
        ("kadın", {"person"}),
        ("leopard", {"pattern"}),
        ("leopar", {"pattern"}),
        ("kadın leopard", {"person", "pattern"}),
        ("leopar kadın", {"person", "pattern"}),
        ("çanta", {"object"}),
        ("kadın çanta", {"person", "object"}),
        ("gözlük", {"object"}),
        ("gözlüklü kadın", {"person", "object"}),
        ("mavi çanta", {"object", "color"}),
    ]
    for q, expected in cases:
        plan = build_intent_evidence_plan(q)
        missing = expected - plan.wanted
        assert not missing, f"{q!r}: missing {missing}, got {plan.wanted}"

    # Product/garment may resolve via registry (learned) or garment stems (product).
    leo_g = build_intent_evidence_plan("leopard gömlek")
    assert "pattern" in leo_g.wanted
    assert leo_g.wanted.intersection({"product", "learned"})

    dress = build_intent_evidence_plan("çiçekli kadın elbise")
    assert "person" in dress.wanted
    assert "pattern" in dress.wanted or "motif" in dress.wanted

    gomlek = build_intent_evidence_plan("gömlek")
    assert gomlek.wanted.intersection({"product", "learned"})

def test_visual_only_no_routing():
    results = [
        SimpleNamespace(
            score=0.8,
            score_percent=80.0,
            is_self_match=False,
            debug={},
            breakdown={},
            pattern_family="animal_print",
            filename="x.jpg",
            path="/x.jpg",
        )
    ]
    out = apply_intent_evidence_routing(results, "", has_image=True)
    assert out[0].score == 0.8
    assert not out[0].debug.get("intent_routing")


def test_person_query_boosts_person_not_fabric_only():
    person = SimpleNamespace(
        score=0.70,
        score_percent=70.0,
        is_self_match=False,
        debug={"human_semantic_only": True, "gender_visual_score": 0.4},
        breakdown={},
        pattern_family="garment_photo",
        filename="model.jpg",
        path="/model.jpg",
    )
    fabric = SimpleNamespace(
        score=0.72,
        score_percent=72.0,
        is_self_match=False,
        debug={"animal_print_type": "leopard", "pattern_family": "animal_print"},
        breakdown={},
        pattern_family="animal_print",
        filename="leo_swatch.tif",
        path="/leo_swatch.tif",
    )
    out = apply_intent_evidence_routing([fabric, person], "kadın", has_image=True)
    by_name = {r.filename: r for r in out}
    assert by_name["model.jpg"].score > by_name["leo_swatch.tif"].score
    assert "person" in (by_name["model.jpg"].debug.get("intent_evidence_matched") or [])
    assert by_name["leo_swatch.tif"].debug.get("intent_conflict") == "pattern_without_person"


def test_pattern_query_boosts_fabric():
    person = SimpleNamespace(
        score=0.70,
        score_percent=70.0,
        is_self_match=False,
        debug={"human_semantic_only": True},
        breakdown={},
        pattern_family="garment_photo",
        filename="model.jpg",
        path="/model.jpg",
    )
    fabric = SimpleNamespace(
        score=0.68,
        score_percent=68.0,
        is_self_match=False,
        debug={"animal_print_type": "leopard", "pattern_family": "animal_print"},
        breakdown={},
        pattern_family="animal_print",
        filename="leo.tif",
        path="/leo.tif",
    )
    out = apply_intent_evidence_routing([person, fabric], "leopard", has_image=True)
    by_name = {r.filename: r for r in out}
    assert by_name["leo.tif"].score > by_name["model.jpg"].score


def test_compound_woman_leopard_rewards_both():
    both = SimpleNamespace(
        score=0.65,
        score_percent=65.0,
        is_self_match=False,
        debug={
            "human_semantic_only": True,
            "animal_print_type": "leopard",
            "pattern_family": "animal_print",
        },
        breakdown={},
        pattern_family="animal_print",
        filename="woman_leo.jpg",
        path="/woman_leo.jpg",
    )
    only_leo = SimpleNamespace(
        score=0.65,
        score_percent=65.0,
        is_self_match=False,
        debug={"animal_print_type": "leopard", "pattern_family": "animal_print"},
        breakdown={},
        pattern_family="animal_print",
        filename="swatch.tif",
        path="/swatch.tif",
    )
    out = apply_intent_evidence_routing(
        [only_leo, both], "kadın leopard", has_image=True
    )
    by_name = {r.filename: r for r in out}
    assert by_name["woman_leo.jpg"].score > by_name["swatch.tif"].score
    matched = by_name["woman_leo.jpg"].debug.get("intent_evidence_matched") or []
    assert "person" in matched and "pattern" in matched


def test_bag_query_not_dominated_by_leopard_fabric():
    bag = SimpleNamespace(
        score=0.60,
        score_percent=60.0,
        is_self_match=False,
        debug={},
        breakdown={},
        pattern_family="unknown",
        filename="deri_canta.jpg",
        path="/bags/deri_canta.jpg",
    )
    leo = SimpleNamespace(
        score=0.70,
        score_percent=70.0,
        is_self_match=False,
        debug={"animal_print_type": "leopard", "pattern_family": "animal_print"},
        breakdown={},
        pattern_family="animal_print",
        filename="leo.tif",
        path="/prints/leo.tif",
    )
    out = apply_intent_evidence_routing([leo, bag], "çanta", has_image=True)
    by_name = {r.filename: r for r in out}
    assert by_name["deri_canta.jpg"].score >= by_name["leo.tif"].score


def test_suppress_pattern_first_for_person_only():
    assert should_suppress_pattern_first(build_intent_evidence_plan("kadın"))
    assert should_suppress_pattern_first(build_intent_evidence_plan("çanta"))
    assert not should_suppress_pattern_first(build_intent_evidence_plan("leopard"))
    assert not should_suppress_pattern_first(build_intent_evidence_plan("kadın leopard"))


def test_collect_evidence_no_invention():
    r = SimpleNamespace(
        debug={},
        pattern_family="unknown",
        filename="x.jpg",
        path="/x.jpg",
        category_path="",
    )
    ev = collect_candidate_evidence(r)
    assert "person" not in ev
    assert "pattern" not in ev

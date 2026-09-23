"""Area Select should reuse normal pattern-aware ranking (no crop-only formula)."""

from __future__ import annotations

from types import SimpleNamespace

from core.intent_evidence_routing import (
    apply_intent_evidence_routing,
    plan_from_visual_profile,
)
from core.texture_profile import TextureProfile


def test_plan_from_leopard_crop_profile():
    prof = TextureProfile(
        pattern_family="animal_print",
        animal_print_type="leopard",
    )
    plan = plan_from_visual_profile(prof)
    assert "pattern" in plan.wanted
    assert plan.values.get("pattern") == "leopard"
    assert plan.raw == "[visual_profile]"


def test_plan_from_floral_crop_profile():
    plan = plan_from_visual_profile({"pattern_family": "floral"})
    assert "pattern" in plan.wanted
    assert plan.values.get("pattern") == "floral"


def test_crop_visual_routing_lifts_same_animal_over_texture_only():
    """Leopard crop evidence should prefer animal_print peers over floral texture."""
    plan = plan_from_visual_profile(
        TextureProfile(pattern_family="animal_print", animal_print_type="leopard")
    )
    leo = SimpleNamespace(
        score=0.62,
        score_percent=62.0,
        is_self_match=False,
        debug={"animal_print_type": "leopard", "pattern_family": "animal_print"},
        breakdown={},
        pattern_family="animal_print",
        animal_print_type="leopard",
        filename="leo_a.tif",
        path="/prints/leo_a.tif",
    )
    floral = SimpleNamespace(
        score=0.70,
        score_percent=70.0,
        is_self_match=False,
        debug={"pattern_family": "floral"},
        breakdown={},
        pattern_family="floral",
        animal_print_type="",
        filename="flower.jpg",
        path="/prints/flower.jpg",
    )
    out = apply_intent_evidence_routing(
        [floral, leo], "", has_image=True, plan=plan
    )
    by = {r.filename: r for r in out}
    assert by["leo_a.tif"].score > by["flower.jpg"].score
    assert "pattern" in (by["leo_a.tif"].debug.get("intent_evidence_matched") or [])


def test_visual_only_without_plan_unchanged():
    r = SimpleNamespace(
        score=0.55,
        score_percent=55.0,
        is_self_match=False,
        debug={},
        breakdown={},
        pattern_family="animal_print",
        filename="x.tif",
        path="/x.tif",
    )
    out = apply_intent_evidence_routing([r], "", has_image=True)
    assert out[0].score == 0.55


def test_score_record_source_has_no_crop_rewrite():
    """Guard: crop-only patch/texture rewrite must stay removed from _score_record."""
    import inspect

    from core.search_engine import SearchEngine

    src = inspect.getsource(SearchEngine._score_record)
    assert "score * 0.75 + patch_sim * 0.35" not in src
    assert "and not crop_search" not in src or "pattern_first" in src
    # pattern_first must not be gated off for crop
    assert "and not crop_search\n            and not protected_match" not in src

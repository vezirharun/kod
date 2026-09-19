"""Aşama 2B — Query Attribute Intelligence tests."""
from __future__ import annotations

from types import SimpleNamespace

from core.object_pattern_gate import apply_object_pattern_gate
from core.query_attribute_intel import (
    apply_query_attribute_intel,
    extract_query_attributes,
    score_attributes_against_dna,
)


def _row(
    *,
    score: float,
    family: str = "animal_print",
    subtype: str = "tiger",
    scale: str = "",
    density: str = "",
    repeat_type: str = "All Over",
    scale_score: float = 0.3,
    repeat_density: float = 0.4,
    color_family: str = "",
    dominant_colors: list | None = None,
    user_taught: bool = False,
    learned_exact: bool = False,
    learned_canonical: str = "",
) -> SimpleNamespace:
    dna = {
        "scale": scale,
        "density": density,
        "repeat_type": repeat_type,
        "color_family": color_family,
        "dominant_colors": dominant_colors or [],
        "motif": subtype.title() if subtype else "",
        "confidence": 0.7,
        "pattern_dna_confidence": 0.7,
    }
    tm = {
        "pattern_family": family,
        "animal_print_type": subtype,
        "pattern_dna": dna,
        "scale_pattern_score": scale_score,
        "repeat_density": repeat_density,
        "color_family": color_family,
    }
    dbg = {
        "texture_map": tm,
        "pattern_family": family,
    }
    if user_taught:
        dbg["user_taught_positive"] = True
    if learned_exact:
        dbg["learned_concept_exact"] = True
        dbg["learned_canonical"] = learned_canonical or subtype.title()
    return SimpleNamespace(
        score=score,
        score_percent=score * 100,
        pattern_family=family,
        animal_print_type=subtype,
        color_family=color_family,
        debug=dbg,
        is_self_match=False,
    )


def test_extract_kucuk_yogun_siyah_krem_kaplan():
    a = extract_query_attributes("küçük yoğun siyah krem kaplan")
    assert a.motif == "tiger"
    assert a.scale == "small"
    assert a.density == "dense"
    assert "black" in a.colors
    assert "cream" in a.colors


def test_extract_buyuk_seyrek_and_geometric():
    assert extract_query_attributes("büyük seyrek kaplan").scale == "large"
    assert extract_query_attributes("büyük seyrek kaplan").density == "sparse"
    g = extract_query_attributes("geometrik ince çizgili")
    assert g.style == "geometric" or g.pattern_type == "stripe" or g.orientation


def test_kucuk_kaplan_prefers_small_tiger():
    small = _row(score=0.80, scale="Low", density="Medium", scale_score=0.15, repeat_density=0.35)
    large = _row(score=0.82, scale="High", density="Medium", scale_score=0.60, repeat_density=0.35)
    apply_query_attribute_intel([small, large], "küçük kaplan")
    assert small.score > large.score


def test_buyuk_kaplan_prefers_large_tiger():
    small = _row(score=0.82, scale="Low", scale_score=0.12, repeat_density=0.3)
    large = _row(score=0.80, scale="High", scale_score=0.62, repeat_density=0.3)
    apply_query_attribute_intel([small, large], "büyük kaplan")
    assert large.score > small.score


def test_yogun_vs_seyrek_kaplan():
    dense = _row(score=0.80, density="High", repeat_density=0.62)
    sparse = _row(score=0.82, density="Low", repeat_density=0.08)
    apply_query_attribute_intel([dense, sparse], "yoğun kaplan")
    assert dense.score > sparse.score
    dense2 = _row(score=0.82, density="High", repeat_density=0.62)
    sparse2 = _row(score=0.80, density="Low", repeat_density=0.08)
    apply_query_attribute_intel([dense2, sparse2], "seyrek kaplan")
    assert sparse2.score > dense2.score


def test_siyah_krem_color_boost():
    match = _row(
        score=0.80,
        color_family="black_white",
        dominant_colors=["black", "cream"],
        scale="Medium",
        density="Medium",
    )
    miss = _row(
        score=0.81,
        color_family="blue",
        dominant_colors=["blue", "navy"],
        scale="Medium",
        density="Medium",
    )
    apply_query_attribute_intel([match, miss], "siyah krem kaplan")
    assert match.score > miss.score


def test_wrong_attribute_does_not_hard_filter():
    large = _row(score=0.85, scale="High", scale_score=0.7)
    apply_query_attribute_intel([large], "küçük kaplan")
    # Still present; only soft penalty.
    assert large.score > 0.70
    assert large.debug.get("query_attribute_intel", {}).get("applied")


def test_tiger_not_converted_to_leopard_by_attributes():
    tiger_wrong_scale = _row(
        score=0.90,
        subtype="tiger",
        scale="High",
        scale_score=0.7,
    )
    tiger_wrong_scale.debug["learned_concept"] = True
    tiger_wrong_scale.debug["learned_canonical"] = "Tiger"
    leopard_good_scale = _row(
        score=0.84,
        subtype="leopard",
        scale="Low",
        scale_score=0.1,
    )
    leopard_good_scale.debug["learned_concept"] = True
    leopard_good_scale.debug["learned_canonical"] = "Leopard"
    apply_query_attribute_intel(
        [tiger_wrong_scale, leopard_good_scale], "küçük kaplan"
    )
    assert tiger_wrong_scale.score >= leopard_good_scale.score
    # Leopard must not gain attribute boost for a Tiger query.
    assert leopard_good_scale.debug.get("query_attribute_intel", {}).get("delta", 0) == 0


def test_user_positive_authority_preserved():
    user = _row(score=0.93, scale="High", scale_score=0.8, user_taught=True)
    before = user.score
    apply_query_attribute_intel([user], "küçük yoğun kaplan")
    assert user.score == before


def test_object_pattern_still_prefers_fabric_for_kucuk_kaplan():
    fabric = _row(
        score=0.78,
        scale="Low",
        density="High",
        scale_score=0.15,
        repeat_density=0.55,
    )
    photo = SimpleNamespace(
        score=0.90,
        score_percent=90,
        pattern_family="unknown",
        animal_print_type="",
        debug={
            "texture_map": {
                "pattern_family": "unknown",
                "pattern_dna": {"confidence": 0.05},
                "repeat_density": 0.02,
                "visual_concept_dna": {
                    "objects": [
                        {
                            "lemma": "tiger",
                            "label": "tiger",
                            "category": "animal",
                            "confidence": 0.92,
                            "detected": True,
                        }
                    ]
                },
            }
        },
        is_self_match=False,
    )
    rows = [fabric, photo]
    # Explicit pattern CONTEXT required for textile preference (bare concept is neutral).
    apply_query_attribute_intel(rows, "küçük kaplan deseni")
    apply_object_pattern_gate(rows, "küçük kaplan deseni")
    assert fabric.score > photo.score


def test_score_meta_reports_matches():
    row = _row(score=0.8, scale="Low", density="High", repeat_density=0.6)
    attrs = extract_query_attributes("küçük yoğun kaplan")
    meta = score_attributes_against_dna(attrs, row)
    assert meta["applied"]
    assert meta["matches"].get("scale", 0) > 0
    assert meta["matches"].get("density", 0) > 0

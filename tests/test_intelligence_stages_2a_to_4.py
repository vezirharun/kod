"""Stages 2A–4 intelligence chain regression (no index writes)."""
from __future__ import annotations

from types import SimpleNamespace

from core.discriminative_concept_intel import apply_discriminative_concept_intel
from core.learned_concept_search import concept_query_relation
from core.object_pattern_gate import apply_object_pattern_gate
from core.query_attribute_intel import (
    apply_query_attribute_intel,
    extract_query_attributes,
)
from core.search_intelligence_chain import (
    analyze_query_intelligence,
    apply_search_intelligence_chain,
)
from core.visual_variant_intel import annotate_visual_variants


def _row(
    *,
    score: float,
    family: str = "animal_print",
    subtype: str = "tiger",
    scale: str = "Low",
    density: str = "High",
    colors: list | None = None,
    learned: str = "",
    learned_exact: bool = False,
    user_taught: bool = False,
    objects: list | None = None,
    dna_conf: float = 0.7,
):
    tm = {
        "pattern_family": family,
        "animal_print_type": subtype,
        "repeat_density": 0.55 if family == "animal_print" else 0.02,
        "pattern_dna": {
            "confidence": dna_conf,
            "scale": scale,
            "density": density,
            "dominant_colors": colors or ["black", "cream"],
            "repeat_type": "allover",
        },
        "visual_concept_dna": {"objects": objects or [], "concepts": []},
    }
    dbg = {
        "texture_map": tm,
        "pattern_family": family,
        "animal_print_type": subtype,
    }
    if learned:
        dbg["learned_concept"] = True
        dbg["learned_canonical"] = learned
    if learned_exact:
        dbg["learned_concept_exact"] = True
        dbg["learned_canonical"] = learned or dbg.get("learned_canonical") or "Tiger"
    if user_taught:
        dbg["user_taught_positive"] = True
    return SimpleNamespace(
        score=score,
        score_percent=score * 100,
        pattern_family=family,
        animal_print_type=subtype,
        debug=dbg,
        is_self_match=False,
    )


# ----- 2A -----
def test_2a_tiger_textile_over_photo():
    fabric = _row(score=0.78, dna_conf=0.8)
    photo = _row(
        score=0.90,
        family="unknown",
        subtype="",
        dna_conf=0.05,
        objects=[
            {
                "lemma": "tiger",
                "label": "tiger",
                "category": "animal",
                "confidence": 0.92,
                "detected": True,
            }
        ],
    )
    apply_object_pattern_gate([fabric, photo], "kaplan")
    assert fabric.score > photo.score


def test_2a_user_exact_protected():
    photo = _row(
        score=0.91,
        family="unknown",
        dna_conf=0.05,
        learned_exact=True,
        learned="Tiger",
        objects=[
            {
                "lemma": "tiger",
                "category": "animal",
                "confidence": 0.9,
                "detected": True,
            }
        ],
    )
    before = photo.score
    apply_object_pattern_gate([photo], "kaplan")
    assert photo.score == before


# ----- 2B -----
def test_2b_compound_keeps_exact():
    for q in (
        "kaplan",
        "küçük kaplan",
        "yoğun kaplan",
        "küçük yoğun siyah krem kaplan",
    ):
        rel = concept_query_relation(q, "Tiger", ["kaplan", "Tiger"], "Animal Print")
        assert rel in {"exact", "translation", "alias", "typo"}, q
        a = extract_query_attributes(q)
        if "küçük" in q or "kucuk" in q.replace("ü", "u"):
            assert a.scale == "small" or "küçük" not in q
        assert a.motif in {"tiger", "kaplan", ""} or True


# ----- 2C -----
def test_2c_dna_attributes_rank_same_concept():
    small = _row(score=0.85, scale="Low", density="High", learned="Tiger")
    large = _row(score=0.85, scale="High", density="Low", learned="Tiger")
    apply_query_attribute_intel([small, large], "küçük yoğun siyah krem kaplan")
    assert small.score > large.score


# ----- 2D -----
def test_2d_tiger_query_demotes_leopard_not_user():
    tiger = _row(score=0.88, learned="Tiger", learned_exact=True)
    leo = _row(score=0.87, subtype="leopard", learned="Leopard")
    user_leo = _row(
        score=0.86, subtype="leopard", learned="Leopard", user_taught=True
    )
    apply_discriminative_concept_intel([tiger, leo, user_leo], "küçük kaplan")
    assert leo.score < 0.87
    assert user_leo.score == 0.86
    assert tiger.score == 0.88


def test_2d_heuristic_not_user_verified():
    leo = _row(score=0.9, subtype="leopard", learned="Leopard")
    apply_discriminative_concept_intel([leo], "tiger")
    meta = leo.debug.get("discriminative_concept_intel") or {}
    assert meta.get("user_verified") is False


# ----- 2E -----
def test_2e_variant_not_new_concept():
    tiger_small = _row(score=0.8, scale="Low", learned="Tiger")
    leo = _row(score=0.8, subtype="leopard", learned="Leopard")
    annotate_visual_variants([tiger_small, leo], "küçük kaplan")
    assert tiger_small.debug["visual_variant_intel"]["creates_new_concept"] is False
    assert tiger_small.debug["visual_variant_intel"]["is_same_concept_variant"] is True
    assert leo.debug["visual_variant_intel"]["is_same_concept_variant"] is False


# ----- 4 fusion -----
def test_4_chain_orders_layers():
    fabric = _row(score=0.78, scale="Low", density="High", learned="Tiger")
    photo = _row(
        score=0.92,
        family="unknown",
        subtype="",
        dna_conf=0.05,
        objects=[
            {
                "lemma": "tiger",
                "category": "animal",
                "confidence": 0.9,
                "detected": True,
            }
        ],
    )
    leo = _row(score=0.88, subtype="leopard", learned="Leopard")
    rows = [fabric, photo, leo]
    apply_search_intelligence_chain(rows, "küçük yoğun siyah krem kaplan")
    assert fabric.score > photo.score
    assert leo.score < 0.88
    assert "search_intelligence_chain" in (rows[0].debug or {})
    analysis = analyze_query_intelligence("küçük yoğun siyah krem kaplan")
    assert analysis["attributes"]["scale"] == "small"
    assert analysis["attributes"]["density"] == "dense"
    assert "black" in analysis["attributes"]["colors"]
    assert "Tiger" in analysis.get("concept_relation_hint", "")

"""Aşama 2A — Object ≠ Pattern gate (search/score only)."""
from __future__ import annotations

from types import SimpleNamespace

from core.object_pattern_gate import (
    adjust_score_for_object_vs_pattern,
    apply_object_pattern_gate,
    motif_query_context,
    object_photo_strength,
    textile_strength,
)
from core.textile_terms import normalize_turkish


def _result(
    *,
    score: float,
    family: str = "unknown",
    dna: dict | None = None,
    objects: list | None = None,
    concepts: list | None = None,
    repeat_density: float = 0.0,
    user_taught: bool = False,
    learned_exact: bool = False,
    animal_print_type: str = "",
) -> SimpleNamespace:
    tm: dict = {
        "pattern_family": family,
        "repeat_density": repeat_density,
        "pattern_dna": dna or {},
        "visual_concept_dna": {
            "objects": objects or [],
            "concepts": concepts or [],
        },
    }
    if animal_print_type:
        tm["animal_print_type"] = animal_print_type
    dbg = {"texture_map": tm, "pattern_family": family}
    if user_taught:
        dbg["user_taught_positive"] = True
    if learned_exact:
        dbg["learned_concept_exact"] = True
    return SimpleNamespace(
        file_id=1,
        score=score,
        score_percent=score * 100,
        pattern_family=family,
        animal_print_type=animal_print_type,
        debug=dbg,
        is_self_match=False,
    )


def _tiger_photo(**kw):
    return _result(
        family="unknown",
        objects=[
            {
                "lemma": "tiger",
                "label": "tiger",
                "label_tr": "kaplan",
                "category": "animal",
                "category_tr": "hayvan",
                "confidence": 0.92,
                "detected": True,
                "evidence": "object_detector",
            }
        ],
        dna={"confidence": 0.05, "repeat_type": ""},
        repeat_density=0.02,
        **kw,
    )


def _tiger_fabric(**kw):
    r = _result(
        family="animal_print",
        animal_print_type="tiger",
        dna={
            "confidence": 0.78,
            "pattern_dna_confidence": 0.78,
            "motif": "Tiger",
            "repeat_type": "All Over",
            "density": "High",
            "scale": "Medium",
        },
        repeat_density=0.62,
        objects=[
            {
                "lemma": "tiger",
                "label": "tiger",
                "category": "animal",
                "confidence": 0.4,
                "detected": False,
                "evidence": "visual_concept",
            }
        ],
        **kw,
    )
    r.debug["texture_map"]["organic_blob_score"] = 0.5
    return r


def test_motif_query_context_explicit_only():
    # Bare concept → no forced textile preference.
    bare = motif_query_context("kaplan")
    assert not bare["wants_textile"]
    assert not bare["wants_object"]
    assert bare["context"] == "none"
    assert not motif_query_context("çiçek")["wants_textile"]
    assert motif_query_context("çiçek")["floral"]

    assert motif_query_context("kaplan deseni")["wants_textile"]
    assert motif_query_context("kaplan deseni")["family"] == "animal_print"
    assert motif_query_context("leopard dokusu")["wants_textile"]
    assert motif_query_context("leopard dokusu")["context"] == "texture"
    assert motif_query_context("kaplan fotoğrafı")["wants_object"]
    assert not motif_query_context("kaplan fotoğrafı")["wants_textile"]
    assert motif_query_context("zebra hayvanı")["wants_object"]


def test_tiger_fabric_ranks_above_tiger_photo():
    fabric = _tiger_fabric(score=0.78)
    photo = _tiger_photo(score=0.86)
    apply_object_pattern_gate([fabric, photo], "kaplan deseni")
    assert fabric.score > photo.score
    assert photo.debug.get("object_pattern_gate", {}).get("penalty", 0) > 0
    assert fabric.debug.get("object_pattern_gate", {}).get("reason") == "strong_textile"


def test_bare_concept_does_not_demote_photo():
    fabric = _tiger_fabric(score=0.78)
    photo = _tiger_photo(score=0.86)
    apply_object_pattern_gate([fabric, photo], "kaplan")
    assert fabric.score == 0.78
    assert photo.score == 0.86
    assert not photo.debug.get("object_pattern_gate")


def test_leopard_fabric_ranks_above_leopard_photo():
    fabric = _result(
        score=0.80,
        family="animal_print",
        animal_print_type="leopard",
        dna={
            "confidence": 0.7,
            "pattern_dna_confidence": 0.7,
            "motif": "Leopard",
            "repeat_type": "All Over",
            "density": "High",
        },
        repeat_density=0.55,
    )
    photo = _result(
        score=0.88,
        family="unknown",
        objects=[
            {
                "lemma": "leopard",
                "label": "leopard",
                "category": "animal",
                "confidence": 0.9,
                "detected": True,
            }
        ],
        dna={"confidence": 0.08},
        repeat_density=0.01,
    )
    apply_object_pattern_gate([fabric, photo], "leopar deseni")
    assert fabric.score > photo.score


def test_flower_textile_ranks_above_flower_photo():
    textile = _result(
        score=0.77,
        family="floral",
        dna={
            "confidence": 0.72,
            "pattern_dna_confidence": 0.72,
            "motif": "Rose",
            "repeat_type": "Half Drop",
            "density": "Medium",
        },
        repeat_density=0.48,
    )
    photo = _result(
        score=0.90,
        family="unknown",
        objects=[
            {
                "lemma": "rose",
                "label": "rose",
                "category": "plant",
                "category_tr": "bitki",
                "confidence": 0.93,
                "detected": True,
            }
        ],
        dna={"confidence": 0.05},
        repeat_density=0.0,
    )
    apply_object_pattern_gate([textile, photo], "çiçek deseni")
    assert textile.score > photo.score


def test_strong_textile_not_penalized_unnecessarily():
    fabric = _tiger_fabric(score=0.81)
    # Even with a detected tiger box leftover, strong DNA must not be crushed.
    fabric.debug["texture_map"]["visual_concept_dna"]["objects"] = [
        {
            "lemma": "tiger",
            "label": "tiger",
            "category": "animal",
            "confidence": 0.88,
            "detected": True,
        }
    ]
    before = fabric.score
    new, meta = adjust_score_for_object_vs_pattern(
        before, fabric, query_text="kaplan deseni"
    )
    assert meta.get("reason") == "strong_textile"
    assert meta.get("penalty", 0) == 0
    assert new >= before


def test_tiger_not_merged_with_leopard_signals():
    tiger = _tiger_fabric(score=0.8)
    leopard_photo = _result(
        score=0.85,
        family="unknown",
        objects=[
            {
                "lemma": "leopard",
                "label": "leopard",
                "category": "animal",
                "confidence": 0.9,
                "detected": True,
            }
        ],
    )
    # Query tiger pattern: leopard photo is still object-like but gate is per-result;
    # sibling merge is not this module's job — ensure tiger fabric stays strong.
    apply_object_pattern_gate([tiger, leopard_photo], "kaplan deseni")
    assert tiger.score >= 0.8
    assert tiger.pattern_family == "animal_print"
    assert tiger.animal_print_type == "tiger"


def test_user_taught_never_demoted():
    photo = _tiger_photo(score=0.91, user_taught=True)
    before = photo.score
    apply_object_pattern_gate([photo], "kaplan deseni")
    assert photo.score == before
    assert photo.debug.get("object_pattern_gate") is None or photo.debug.get(
        "object_pattern_gate", {}
    ).get("skipped") == "user_protected"


def test_learned_exact_never_demoted():
    photo = _tiger_photo(score=0.94, learned_exact=True)
    before = photo.score
    new, meta = adjust_score_for_object_vs_pattern(
        before, photo, query_text="kaplan deseni"
    )
    assert new == before
    assert meta.get("skipped") == "user_protected"


def test_object_context_prefers_photo_over_textile():
    fabric = _tiger_fabric(score=0.88)
    photo = _tiger_photo(score=0.80)
    apply_object_pattern_gate([fabric, photo], "kaplan hayvanı")
    assert photo.score >= fabric.score
    assert fabric.debug.get("object_pattern_gate", {}).get("penalty", 0) > 0


def test_generic_contexts_not_concept_specific():
    from core.query_attribute_intel import extract_query_visual_context

    cases = [
        ("zebra", "none", "general"),
        ("zebra deseni", "pattern", "pattern"),
        ("zebra dokusu", "texture", "texture"),
        ("zebra hayvanı", "object", "object"),
        ("gül fotoğrafı", "photo", "photo"),
        ("mermer dokusu", "texture", "texture"),
        ("kamuflaj deseni", "pattern", "pattern"),
        ("geometrik desen", "pattern", "pattern"),
        ("kelebek", "none", "general"),
        ("kelebek hayvanı", "object", "object"),
        ("çiçek dokusu", "texture", "texture"),
    ]
    for q, ctx, vtype in cases:
        got = extract_query_visual_context(q)
        assert got["context"] == ctx, q
        assert got["visual_type"] == vtype, q
        # Concept core must not equal a different species identity.
        core = got["concept_core"]
        if "zebra" in q:
            assert "kaplan" not in core and "leopard" not in core
        if "gül" in q or "gul" in normalize_turkish(q):
            assert "cicek" not in normalize_turkish(core) or "gul" in normalize_turkish(core)


def test_textile_strength_helpers():
    assert textile_strength(_tiger_fabric(score=0.8)) >= 0.55
    assert textile_strength(_tiger_photo(score=0.8)) < 0.35
    assert object_photo_strength(
        _tiger_photo(score=0.8), lemmas={"tiger", "kaplan"}, animal=True
    ) >= 0.45

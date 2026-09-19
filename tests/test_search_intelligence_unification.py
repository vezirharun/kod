"""Search intelligence unification — small targeted tests only."""
from __future__ import annotations

from types import SimpleNamespace

from core.search_intelligence_chain import (
    analyze_query_intelligence,
    apply_search_intelligence_chain,
    build_result_reason,
)


def _row(
    *,
    score: float = 0.7,
    subtype: str = "tiger",
    exact: bool = False,
    colors: list | None = None,
) -> SimpleNamespace:
    dna = {
        "motif": subtype.title(),
        "dominant_colors": colors or [],
        "confidence": 0.8,
    }
    tm = {
        "pattern_family": "animal_print",
        "animal_print_type": subtype,
        "pattern_dna": dna,
        "color_evidence": {
            "detected_colors": colors or [],
            "source": "ai",
        },
        "color_index": {
            "dominant_colors": [[1, 1, 1], [244, 235, 210]]
            if colors
            else [],
        },
    }
    dbg = {
        "texture_map": tm,
        "learned_concept_exact": exact,
        "learned_canonical": subtype.title() if exact else "",
    }
    return SimpleNamespace(
        score=score,
        score_percent=score * 100,
        pattern_family="animal_print",
        animal_print_type=subtype,
        is_self_match=False,
        debug=dbg,
    )


def test_query_meaning_kucuk_yogun_siyah_krem_kaplan():
    a = analyze_query_intelligence("küçük yoğun siyah krem kaplan deseni")
    assert a["attributes"]["motif"] == "tiger"
    assert a["attributes"]["scale"] == "small"
    assert a["attributes"]["density"] == "dense"
    assert "black" in a["colors"] and "cream" in a["colors"]
    assert a["intent_type"] == "pattern"
    assert a["context"] == "pattern"
    assert a["visual_type"] == "pattern"
    assert "leopard" not in (a.get("concept_relation_hint") or "").lower() or "Tiger" in (
        a.get("concept_relation_hint") or ""
    )


def test_visual_context_bare_vs_explicit():
    bare = analyze_query_intelligence("leopard")
    assert bare["context"] == "none"
    assert bare["intent_type"] == "general"
    assert bare["visual_type"] == "general"

    doku = analyze_query_intelligence("mermer dokusu")
    assert doku["context"] == "texture"
    assert doku["intent_type"] == "texture"
    assert "mermer" in (doku["concept_core"] or "")

    hayvan = analyze_query_intelligence("kelebek hayvanı")
    assert hayvan["context"] == "object"
    assert hayvan["intent_type"] == "object"


def test_basic_queries_concept_stable():
    cases = {
        "kaplan": "tiger",
        "tiger": "tiger",
        "küçük kaplan": "tiger",
        "siyah krem kaplan": "tiger",
        "leopard": "leopard",
        "yılan": None,  # may be snake via NL
        "snake": None,
        "çiçek": None,
        "flower": None,
        "dudak": None,
        "lips": None,
    }
    for q, motif in cases.items():
        a = analyze_query_intelligence(q)
        assert a["normalized"]
        if motif:
            assert a["attributes"]["motif"] == motif


def test_chain_tiger_not_become_leopard():
    tiger = _row(score=0.8, subtype="tiger", exact=True, colors=["black", "cream"])
    leopard = _row(score=0.79, subtype="leopard", colors=["black", "cream"])
    out = apply_search_intelligence_chain(
        [tiger, leopard], "siyah krem kaplan"
    )
    assert out[0].animal_print_type == "tiger" or any(
        r.animal_print_type == "tiger" for r in out
    )
    assert any(r.animal_print_type == "leopard" for r in out)
    # Identity fields unchanged
    assert tiger.animal_print_type == "tiger"
    assert leopard.animal_print_type == "leopard"


def test_reason_learned_and_exact():
    r = _row(exact=True, colors=["black"])
    r.is_self_match = True
    assert build_result_reason(r, {"concept_core": "Tiger", "colors": ["black"]}) == "EXACT"
    r2 = _row(exact=True, colors=["black", "cream"])
    reason = build_result_reason(
        r2, {"concept_core": "Tiger", "colors": ["black", "cream"]}
    )
    assert "LEARNED" in reason


def test_chain_attaches_unified_debug():
    row = _row(exact=True, colors=["black", "cream"])
    out = apply_search_intelligence_chain([row], "siyah krem kaplan")
    dbg = out[0].debug
    assert dbg.get("search_reason")
    assert dbg.get("query_meaning", {}).get("colors")
    chain = dbg.get("search_intelligence_chain") or {}
    assert chain.get("unified") is True
    assert "query_meaning" in (chain.get("layers") or [])
    assert "result_reasons" in (chain.get("layers") or [])

"""Ortak metin/görsel sonuç alaka sıralayıcı.

Amaç: Metin aramasının son sıralamasını görsel aramadaki kanıt-füzyon mantığına
yaklaştırmak. Bu katman indekslemez; yalnızca SearchResult üzerindeki mevcut
kanıtları birleştirir.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


def _f(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _max_values(mapping: dict[str, Any], keys: Iterable[str]) -> float:
    return max((_f(mapping.get(k)) for k in keys), default=0.0)


def _is_leaf_query(dbg: dict[str, Any], query_text: str = "") -> bool:
    """True when the text query is a leaf motif (TR yaprak / EN leaf)."""
    intent = dbg.get("semantic_intent") or {}
    if str(intent.get("motif") or "") == "leaf":
        return True
    hints = dbg.get("query_family_hints")
    if isinstance(hints, dict):
        parts: list[str] = []
        for key in ("pattern_type", "pattern_family", "motif", "types"):
            val = hints.get(key)
            if isinstance(val, (list, tuple, set)):
                parts.extend(str(x) for x in val)
            elif val:
                parts.append(str(val))
        blob = " ".join(parts).lower()
        if "leaf" in blob or "yaprak" in blob:
            return True
    qt = (query_text or str(dbg.get("query_text") or "")).lower()
    if qt and ("yaprak" in qt or " leaf" in f" {qt} " or qt.strip() in ("leaf", "leaves")):
        return True
    return False


_NON_FAMILY = frozenset(
    {"", "unknown", "texture_ground", "plain", "document", "garment_photo"}
)


@dataclass(frozen=True)
class QueryProfile:
    """Query-aware ranking profile. Not a per-keyword if/else table."""

    mode: str = "text"
    visual_retrieval: bool = False
    human_query: bool = False
    keyword_is_weak: bool = False
    visual_pattern_query: bool = False


def build_query_profile(
    *,
    mode: str = "text",
    visual_retrieval: bool = False,
    human_query: bool = False,
    visual_pattern_query: bool = False,
) -> QueryProfile:
    keyword_is_weak = bool(mode == "text" and visual_retrieval and not human_query)
    return QueryProfile(
        mode=mode,
        visual_retrieval=visual_retrieval,
        human_query=human_query,
        keyword_is_weak=keyword_is_weak,
        visual_pattern_query=visual_pattern_query,
    )


def compute_unified_relevance(
    result: Any,
    *,
    mode: str = "text",
    human_query: bool = False,
    profile: QueryProfile | None = None,
) -> tuple[float, dict[str, float]]:
    """Return calibrated final relevance and its evidence components.

    Text mode is visual-first when a CLIP/UVI signal exists, while retaining a
    strong metadata floor for exact indexed labels. Image mode keeps the
    existing visual score dominant and only adds persisted evidence.
    """
    prof = profile or build_query_profile(mode=mode, human_query=human_query)
    mode = prof.mode
    human_query = prof.human_query
    dbg = getattr(result, "debug", {}) or {}
    bd = getattr(result, "breakdown", {}) or {}
    base = _f(getattr(result, "score", 0.0))

    clip = _max_values(dbg, ("clip_score", "visual_score", "global_visual_score"))
    clip = max(clip, _f(bd.get("clip")))
    gender = _max_values(dbg, ("gender_visual_score", "human_semantic_score"))
    entity_visual = _max_values(dbg, ("entity_semantic_score", "entity_evidence_score"))
    visual = max(clip, gender, entity_visual)

    exact_text = _max_values(
        bd,
        ("filename_score", "ocr_score", "brand_alias_score", "entity_score"),
    )
    semantic = _max_values(
        bd,
        (
            "semantic_score", "auto_tag_score", "family_score", "texture_score",
            "feedback_score", "group_score", "nl_score", "text_score",
        ),
    )
    evidence = max(exact_text, semantic)

    qev = dbg.get("query_evidence_report") or {}
    visual_grade = str(dbg.get("visual_grade") or qev.get("visual_grade") or "")
    visual_grade_bonus = {
        "visual_exact": 0.06,
        "visual_strong": 0.04,
        "visual_medium": 0.015,
    }.get(visual_grade, 0.0)

    # A rival subtype is a real negative signal. Keep the penalty modest: the
    # existing UVI hard gate remains responsible for rejecting contradictions.
    rival = _f(dbg.get("rival_clip"))
    margin = _f((dbg.get("clip_margin") or 0.0) + 0.5) - 0.5
    rival_penalty = max(0.0, rival - max(0.0, _f(clip) - 0.08)) * 0.25

    visual_similarity = visual
    text_similarity = exact_text
    semantic_similarity = semantic
    pattern_dna_similarity = _max_values(bd, ("dna_score", "pattern_dna_score"))
    texture_similarity = _f(bd.get("texture_score"))
    patch_similarity = _max_values(bd, ("patch_sim", "patch_score"))
    color_similarity = _max_values(bd, ("color_score", "palette_similarity"))
    structure_similarity = _max_values(bd, ("family_score", "structure_score"))
    keyword_match = _f(bd.get("filename_score"))

    extra_visual = max(
        pattern_dna_similarity, texture_similarity, patch_similarity,
        color_similarity,
    )
    if mode == "text":
        extra_visual = max(
            extra_visual,
            _max_values(dbg, ("dino_score", "dino_sim")),
            _f(bd.get("dino")),
        )
    if extra_visual > 0:
        visual = max(visual, extra_visual * (0.55 if mode == "text" else 0.85))

    intent = dbg.get("semantic_intent") or {}
    q_family = str(
        intent.get("family")
        or dbg.get("query_family")
        or ""
    )
    if isinstance(dbg.get("query_family_hints"), dict):
        q_family = q_family or str(
            (dbg.get("query_family_hints") or {}).get("pattern_family") or ""
        )
    r_family = str(
        getattr(result, "pattern_family", "")
        or dbg.get("pattern_family")
        or dbg.get("result_family")
        or ""
    )
    family_aligned = bool(
        q_family
        and r_family
        and q_family not in _NON_FAMILY
        and r_family not in _NON_FAMILY
        and q_family == r_family
    )
    known_mismatch = bool(
        q_family
        and r_family
        and q_family not in _NON_FAMILY
        and r_family not in _NON_FAMILY
        and q_family != r_family
    )
    unknown_family = r_family in _NON_FAMILY
    query_has_family = q_family not in _NON_FAMILY

    if mode == "image":
        # Preserve the established visual engine score as the primary signal.
        unified = 0.72 * base + 0.18 * visual + 0.10 * semantic
    else:
        if human_query:
            # A person/gender text query is a semantic object query, not a
            # filename search. Non-human textile records must not win merely
            # because their filename happens to contain the query token.
            if gender >= 0.18 or dbg.get("face_gender_match"):
                human_strength = max(gender, 1.0 if dbg.get("face_gender_match") else 0.0)
                unified = 0.94 * human_strength + 0.06 * base
            else:
                unified = min(0.55 * base, 0.55)
        else:
            from core.semantic_pattern_intel import calibrate_text_clip

            clip_raw = max(clip, _f(dbg.get("clip_score")))
            vis_cal = calibrate_text_clip(clip_raw) if clip_raw > 0 else 0.0
            visual_win = bool(dbg.get("visual_win"))
            # Image-like CLIP (0.55+) is already a strong visual; text–image 0.33–0.38 is not.
            visual_win = visual_win or vis_cal >= 0.80 or clip_raw >= 0.55
            structure_boost = structure_similarity if family_aligned else 0.0
            leaf_q = _is_leaf_query(dbg)
            leaf_clip_floor = bool(leaf_q and clip_raw >= 0.20)
            if leaf_clip_floor:
                # CLIP ~0.20–0.32 is typical for leaf/yaprak; do not let the
                # generic CLIP-only 0.50/0.58 caps overwrite the leaf floor.
                unified = max(0.62, min(0.74, vis_cal + 0.16))
            elif clip_raw >= 0.20:
                if visual_win:
                    unified = (
                        0.82 * vis_cal
                        + 0.10 * extra_visual
                        + 0.08 * structure_boost
                    )
                    unified = max(unified, vis_cal)
                else:
                    unified = (
                        0.42 * vis_cal
                        + 0.16 * extra_visual
                        + 0.12 * structure_boost
                        + 0.10 * semantic_similarity
                        + 0.08 * evidence
                    )
                    unified = min(unified, 0.58 if clip_raw >= 0.32 else 0.50)
                if str(visual_grade) in ("", "visual_weak", "visual_none", "none"):
                    if clip_raw < 0.30 and not visual_win:
                        unified = min(unified, 0.50)
                if known_mismatch:
                    if not visual_win:
                        unified = min(unified, 0.48)
                    elif extra_visual < 0.40:
                        unified = min(unified, 0.56)
                if unknown_family and query_has_family and not visual_win:
                    if extra_visual < 0.40:
                        unified = min(unified, 0.55)
                channels = int(visual_win or clip_raw >= 0.32)
                channels += int(extra_visual >= 0.30)
                channels += int(family_aligned)
                channels += int(exact_text >= 0.70)
                if channels <= 1 and not visual_win:
                    unified = min(unified, 0.58)
            elif visual >= 0.75:
                unified = max(
                    0.90 * visual + 0.10 * evidence,
                    0.55 * visual + 0.25 * evidence + 0.20 * base,
                )
            elif visual >= 0.55:
                unified = 0.62 * visual + 0.20 * evidence + 0.18 * base
            elif visual >= 0.30:
                unified = 0.45 * visual + 0.30 * evidence + 0.25 * base
            else:
                unified = 0.62 * evidence + 0.38 * base

        # Exact metadata evidence is authoritative when the visual model is
        # unavailable. It should not be buried under a generic textile score.
        visual_pattern_q = bool(
            (dbg.get("query_family_hints") or {}).get("pattern_family")
            or dbg.get("visual_pattern_query")
            or prof.visual_pattern_query
        )
        leaf_q = _is_leaf_query(dbg)
        if visual_pattern_q and known_mismatch and not leaf_q:
            unified = min(unified, 0.42)
        elif visual_pattern_q and exact_text >= 0.70 and not family_aligned and extra_visual < 0.45:
            unified = min(unified, 0.46)
        elif not human_query and exact_text >= 0.88 and clip < 0.20 and not visual_pattern_q:
            unified = max(unified, 0.78 * exact_text + 0.22 * base)

        if (
            prof.keyword_is_weak
            and keyword_match >= 0.70
            and clip < 0.30
            and not leaf_q
        ):
            unified = min(unified, 0.58)
        if visual_pattern_q and family_aligned:
            unified = max(unified, 0.58 * max(structure_similarity, extra_visual, pattern_dna_similarity))

    unified += visual_grade_bonus * (0.65 if mode == "text" else 0.35)
    unified += max(0.0, margin) * 0.02
    unified -= rival_penalty
    unified = max(0.0, min(1.0, unified))

    components = {
        "unified_relevance": round(unified, 6),
        "base_score": round(base, 6),
        "visual_evidence": round(visual, 6),
        "text_exact_evidence": round(exact_text, 6),
        "semantic_evidence": round(semantic, 6),
        "evidence": round(evidence, 6),
        "rival_penalty": round(rival_penalty, 6),
        "visual_similarity": round(visual_similarity, 6),
        "text_similarity": round(text_similarity, 6),
        "semantic_similarity": round(semantic_similarity, 6),
        "pattern_dna_similarity": round(pattern_dna_similarity, 6),
        "texture_similarity": round(texture_similarity, 6),
        "patch_similarity": round(patch_similarity, 6),
        "color_similarity": round(color_similarity, 6),
        "structure_similarity": round(structure_similarity, 6),
        "keyword_match": round(keyword_match, 6),
        "final_score": round(unified, 6),
    }
    return unified, components


def apply_unified_text_ranking(
    results: list[Any],
    *,
    human_query: bool = False,
    visual_retrieval: bool = False,
    profile: QueryProfile | None = None,
    query_text: str = "",
) -> list[Any]:
    """Calibrate and sort text results by one relevance score."""
    vp = False
    if query_text:
        try:
            from core.visual_pattern_query import is_visual_pattern_query

            vp = is_visual_pattern_query(query_text)
        except Exception:
            vp = False
    prof = profile or build_query_profile(
        mode="text",
        visual_retrieval=visual_retrieval,
        human_query=human_query,
        visual_pattern_query=vp,
    )
    if prof.keyword_is_weak:
        has_strong_visual = any(
            max(
                _f((getattr(r, "debug", {}) or {}).get("clip_score")),
                _f((getattr(r, "breakdown", {}) or {}).get("clip")),
                _f((getattr(r, "debug", {}) or {}).get("gender_visual_score")),
            )
            >= 0.28
            or bool((getattr(r, "debug", {}) or {}).get("visual_win"))
            for r in results
        )
        if not has_strong_visual:
            prof = build_query_profile(
                mode="text",
                visual_retrieval=False,
                human_query=human_query,
                visual_pattern_query=vp,
            )
    qt = (query_text or "").lower()
    stamp_leaf = "yaprak" in qt or qt.strip() in ("leaf", "leaves") or " leaf" in f" {qt} "
    for r in results:
        if stamp_leaf:
            dbg0 = dict(getattr(r, "debug", {}) or {})
            intent0 = dict(dbg0.get("semantic_intent") or {})
            if str(intent0.get("motif") or "") != "leaf":
                intent0["motif"] = "leaf"
                dbg0["semantic_intent"] = intent0
                r.debug = dbg0
        score, comp = compute_unified_relevance(
            r, mode="text", human_query=human_query, profile=prof
        )
        r.score = score
        r.score_percent = round(score * 100.0, 1)
        r.debug = {
            **(getattr(r, "debug", {}) or {}),
            "unified_relevance": score,
            "unified_relevance_components": comp,
            "ranking_engine": "unified_relevance_v1",
            "query_profile": {
                "mode": prof.mode,
                "visual_retrieval": prof.visual_retrieval,
                "human_query": prof.human_query,
                "keyword_is_weak": prof.keyword_is_weak,
                "visual_pattern_query": prof.visual_pattern_query,
            },
        }
        r.breakdown["unified_relevance"] = score
    results.sort(
        key=lambda r: (
            -_f((getattr(r, "debug", {}) or {}).get("unified_relevance")),
            -_f((getattr(r, "debug", {}) or {}).get("gender_visual_score")),
            -_f((getattr(r, "debug", {}) or {}).get("clip_score")),
            -_f(getattr(r, "score", 0.0)),
            int(getattr(r, "file_id", 0) or 0),
        )
    )
    return results

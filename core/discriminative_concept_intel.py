"""Discriminative concept intelligence — search/score only (Stage 2D).

Soft-demotes rival sibling concepts when the query has an EXACT leaf match.
Never writes learning, never marks AUTO as USER VERIFIED.
User / learned_exact / manual results are never demoted.
"""
from __future__ import annotations

from typing import Any

from core.textile_terms import normalize_turkish

_MAX_RIVAL_PENALTY = 0.08

# Leaf siblings that must not share attribute/visual bonus under each other.
_ANIMAL_SIBLINGS = frozenset(
    {
        "tiger",
        "kaplan",
        "leopard",
        "leopar",
        "zebra",
        "snake",
        "yilan",
        "snake skin",
        "snakeskin",
    }
)
_FLORAL_SIBLINGS = frozenset(
    {
        "rose",
        "gul",
        "gül",
        "flower",
        "floral",
        "cicek",
        "çiçek",
        "tulip",
        "lale",
    }
)


def _user_protected(result: Any) -> bool:
    dbg = getattr(result, "debug", None) or {}
    if not isinstance(dbg, dict):
        return False
    if getattr(result, "is_self_match", False):
        return True
    if dbg.get("protected_exact") or dbg.get("user_taught_positive"):
        return True
    if dbg.get("learned_concept_exact"):
        return True
    if dbg.get("user_labeled") or dbg.get("category_source") == "manual_user":
        return True
    return False


def _label_keys(text: str) -> set[str]:
    n = normalize_turkish(str(text or ""))
    if not n:
        return set()
    out = {n}
    try:
        from core.concept_query_normalize import leaf_translation_keys

        out |= leaf_translation_keys(n)
    except Exception:
        pass
    return {x for x in out if x}


def query_leaf_keys(query: str) -> set[str]:
    """Concept identity keys for the query (attributes stripped)."""
    try:
        from core.query_attribute_intel import concept_core_text

        core = concept_core_text(query) or query
    except Exception:
        core = query
    return _label_keys(core)


def _result_concept_keys(result: Any) -> set[str]:
    dbg = getattr(result, "debug", None) or {}
    keys: set[str] = set()
    if isinstance(dbg, dict):
        for field in ("learned_canonical", "learned_concept_label"):
            keys |= _label_keys(str(dbg.get(field) or ""))
    sub = normalize_turkish(
        str(
            getattr(result, "animal_print_type", "")
            or (dbg.get("animal_print_type") if isinstance(dbg, dict) else "")
            or ""
        )
    )
    if sub:
        keys |= _label_keys(sub)
    return keys


def _are_rivals(query_keys: set[str], result_keys: set[str]) -> bool:
    if not query_keys or not result_keys:
        return False
    if query_keys & result_keys:
        return False  # same concept family leaf
    q_animal = query_keys & _ANIMAL_SIBLINGS
    r_animal = result_keys & _ANIMAL_SIBLINGS
    if q_animal and r_animal and not (q_animal & r_animal):
        return True
    q_flor = query_keys & _FLORAL_SIBLINGS
    r_flor = result_keys & _FLORAL_SIBLINGS
    if q_flor and r_flor and not (q_flor & r_flor):
        return True
    return False


def adjust_score_for_rivals(
    score: float,
    result: Any,
    query_text: str,
) -> tuple[float, dict[str, Any]]:
    """Soft penalty when result concept is a sibling rival of the query leaf."""
    if _user_protected(result):
        return float(score), {"applied": False, "skipped": "user_protected"}
    q_keys = query_leaf_keys(query_text)
    if not q_keys:
        return float(score), {"applied": False, "reason": "no_query_leaf"}
    r_keys = _result_concept_keys(result)
    if not r_keys:
        return float(score), {"applied": False, "reason": "no_result_leaf"}
    if not _are_rivals(q_keys, r_keys):
        return float(score), {"applied": False, "reason": "not_rival"}
    new = max(0.0, float(score) - _MAX_RIVAL_PENALTY)
    return new, {
        "applied": True,
        "penalty": _MAX_RIVAL_PENALTY,
        "before": round(float(score), 4),
        "after": round(new, 4),
        "query_keys": sorted(q_keys)[:8],
        "result_keys": sorted(r_keys)[:8],
        "user_verified": False,  # never promote heuristic to user authority
    }


def apply_discriminative_concept_intel(
    results: list[Any],
    query_text: str,
) -> list[Any]:
    if not results or not (query_text or "").strip():
        return results
    for rec in results:
        try:
            old = float(getattr(rec, "score", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        new, meta = adjust_score_for_rivals(old, rec, query_text)
        if not meta.get("applied") and not meta.get("skipped"):
            continue
        if meta.get("applied"):
            rec.score = new
            if hasattr(rec, "score_percent"):
                rec.score_percent = round(new * 100, 1)
        dbg = dict(getattr(rec, "debug", None) or {})
        dbg["discriminative_concept_intel"] = meta
        rec.debug = dbg
    return results

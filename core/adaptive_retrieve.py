"""Adaptive FAISS retrieve — ranking formülüne dokunmaz, sadece aday havuzunu büyütür."""

from __future__ import annotations

from typing import Any


UNRELATED_CLUSTERS = frozenset(
    {
        "unrelated",
        "far_texture",
        "distant",
        "CLUSTER_UNRELATED",
        "CLUSTER_DISTANT",
    }
)


def next_retrieve_k(current_k: int, index_n: int) -> int | None:
    """800 → 1600 → 3200 … index_n’i aşmadan."""
    index_n = max(0, int(index_n or 0))
    current_k = max(0, int(current_k or 0))
    if index_n <= 0 or current_k >= index_n:
        return None
    nxt = min(index_n, max(current_k * 2, current_k + 800))
    if nxt <= current_k:
        return None
    return nxt


def is_expand_worthy(result: Any) -> bool:
    """Genişlemeye değer kanıt — zayıf %40 eşikler FAISS'i sonsuza açmasın."""
    if getattr(result, "is_self_match", False):
        return True
    dbg = getattr(result, "debug", None) or {}
    if dbg.get("protected_exact"):
        return True
    if getattr(result, "same_pattern_family", False):
        return True
    if getattr(result, "same_animal_family", False):
        return True
    return float(getattr(result, "score", 0) or 0) >= 0.62


def is_relevant_hit(result: Any, floor: float) -> bool:
    """Mevcut evidence/threshold — FAISS’e geldi diye benzer sayma."""
    if getattr(result, "is_self_match", False):
        return True
    dbg = getattr(result, "debug", None) or {}
    if dbg.get("protected_exact"):
        return True
    if float(getattr(result, "score", 0) or 0) >= float(floor):
        return True
    if getattr(result, "same_pattern_family", False):
        return True
    if getattr(result, "same_animal_family", False):
        return True
    cg = str(getattr(result, "cluster_group", "") or "").lower()
    if cg in ("unrelated", "far_texture", "distant"):
        return False
    return False


def should_stop_expansion(
    *,
    new_id_count: int,
    relevant_new: int,
    max_new_score: float,
    faiss_median_new: float | None,
    faiss_tail_ref: float | None,
    floor: float,
) -> tuple[bool, str]:
    if new_id_count <= 0:
        return True, "no_new_ids"
    if (
        faiss_tail_ref is not None
        and faiss_median_new is not None
        and faiss_tail_ref > 0
        and faiss_median_new < 0.72 * float(faiss_tail_ref)
    ):
        return True, "faiss_similarity_drop"
    if relevant_new < 3 and float(max_new_score or 0) < max(0.55, float(floor)):
        return True, "low_relevance_yield"
    return False, ""

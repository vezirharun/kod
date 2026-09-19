"""Aynı pattern family adayları — neden elendi raporu."""

from __future__ import annotations

from typing import Any

from core.dynamic_groups import DEFAULT_HIDDEN_GROUPS, normalize_cluster_key
from core.search_display import visual_search_floor


def _score_fields(result: Any) -> dict[str, float]:
    d = getattr(result, "debug", {}) or {}
    b = getattr(result, "breakdown", {}) or {}
    return {
        "exact": float(d.get("exact_score", d.get("exact_search_score", 0)) or 0),
        "dna": float(d.get("dna_score", b.get("dna", 0)) or 0),
        "semantic": float(d.get("semantic_score", b.get("semantic", 0)) or 0),
        "texture": float(d.get("texture_score", b.get("texture", 0)) or 0),
        "family": float(
            d.get("family_score", d.get("pattern_family_score", d.get("family_variant_score", 0)))
            or 0
        ),
        "final": float(getattr(result, "score", 0) or 0),
    }


def _same_family_candidate(result: Any, query_family: str, query_animal: str) -> bool:
    pf = str(getattr(result, "pattern_family", "") or "")
    ap = str(getattr(result, "animal_print_type", "") or "")
    if query_family and pf and pf == query_family:
        return True
    if query_animal and ap and ap == query_animal:
        return True
    if getattr(result, "same_pattern_family", False) or getattr(result, "same_animal_family", False):
        return True
    return False


def _reject_reason(
    result: Any,
    *,
    floor: float,
    threshold: float,
    hidden_clusters: frozenset[str] | None,
) -> str:
    d = getattr(result, "debug", {}) or {}
    if d.get("reject_reason"):
        return str(d["reject_reason"])
    score = float(getattr(result, "score", 0) or 0)
    if score < floor:
        return f"floor_below:{score:.4f}<{floor:.4f}"
    if score < threshold:
        return f"threshold_below:{score:.4f}<{threshold:.4f}"
    if hidden_clusters:
        key = normalize_cluster_key(
            getattr(result, "cluster_group", "") or getattr(result, "category", "")
        )
        if key in hidden_clusters:
            return f"hidden_cluster:{key}"
    if d.get("user_wrong_block"):
        return "user_wrong_block"
    if d.get("text_family_conflict"):
        return f"text_family_conflict:{d['text_family_conflict']}"
    if d.get("why_animal_rejected"):
        return f"animal_rejected:{d['why_animal_rejected']}"
    return "not_in_display_list"


def build_family_reject_report(
    all_results: list[Any],
    shown_results: list[Any],
    *,
    query_family: str = "",
    query_animal: str = "",
    floor: float = 0.40,
    threshold: float = 0.60,
    hidden_clusters: frozenset[str] | None = None,
    limit: int = 40,
) -> list[dict[str, Any]]:
    """Aynı family'de olup listede görünmeyen adaylar."""
    if not query_family and not query_animal:
        return []
    shown_ids = {int(getattr(r, "file_id", 0) or 0) for r in shown_results}
    report: list[dict[str, Any]] = []
    for result in all_results:
        fid = int(getattr(result, "file_id", 0) or 0)
        if fid in shown_ids:
            continue
        if not _same_family_candidate(result, query_family, query_animal):
            continue
        scores = _score_fields(result)
        report.append(
            {
                "file_id": fid,
                "filename": getattr(result, "filename", ""),
                "pattern_family": getattr(result, "pattern_family", ""),
                "animal_print_type": getattr(result, "animal_print_type", ""),
                "cluster_group": getattr(result, "cluster_group", ""),
                "exact_score": round(scores["exact"], 4),
                "dna_score": round(scores["dna"], 4),
                "semantic_score": round(scores["semantic"], 4),
                "texture_score": round(scores["texture"], 4),
                "family_score": round(scores["family"], 4),
                "final_score": round(scores["final"], 4),
                "reject_reason": _reject_reason(
                    result,
                    floor=floor,
                    threshold=threshold,
                    hidden_clusters=hidden_clusters,
                ),
            }
        )
    report.sort(key=lambda x: (-x["final_score"], x["filename"]))
    return report[:limit]

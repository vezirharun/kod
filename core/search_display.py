"""Görsel arama sonuçlarını skor sırasına göre listeleme."""

from __future__ import annotations

from typing import Any


def visual_search_floor(settings: Any) -> float:
    return float(getattr(settings, "visual_search_floor", 0.40) or 0.40)


def is_ranked_visual_search(
    meta: dict[str, Any] | None, *, has_text: bool = False
) -> bool:
    """Saf görsel aramada gruplar yerine %100→%40 düz liste."""
    if has_text:
        return False
    meta = meta or {}
    if meta.get("mode") != "image":
        return False
    if str(meta.get("text") or "").strip():
        return False
    return True


def filter_ranked_visual_results(
    results: list[Any],
    floor: float,
    *,
    hidden_clusters: frozenset[str] | None = None,
    normalize_cluster_key=None,
    query_family: str = "",
    query_animal: str = "",
) -> list[Any]:
    """Skora göre azalan sıra — %floor ve üzeri."""
    out: list[Any] = []
    for r in results:
        if float(r.score) < floor:
            continue
        if hidden_clusters and normalize_cluster_key:
            key = normalize_cluster_key(
                getattr(r, "cluster_group", "") or getattr(r, "category", "")
            )
            if key in hidden_clusters:
                # Face identity is an independent evidence layer. A person's
                # photo must never be hidden merely because its textile family
                # is unrelated to the query image.
                if bool((getattr(r, "debug", {}) or {}).get("face_match")):
                    out.append(r)
                    continue
                # Aynı pattern family / leopard ailesi gizlenmesin
                same_fam = bool(getattr(r, "same_pattern_family", False))
                same_animal = bool(getattr(r, "same_animal_family", False))
                pf = str(getattr(r, "pattern_family", "") or "")
                ap = str(getattr(r, "animal_print_type", "") or "")
                if same_fam or same_animal:
                    pass
                elif query_family and pf == query_family:
                    pass
                elif query_animal and ap == query_animal:
                    pass
                else:
                    continue
        out.append(r)
    out.sort(
        key=lambda r: (
            0 if getattr(r, "is_self_match", False) else 1,
            -float(r.score),
            -float(getattr(r, "hierarchy_score", 0) or 0),
            int(getattr(r, "file_id", 0)),
        ),
    )
    return out

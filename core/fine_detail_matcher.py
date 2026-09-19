"""İnce fark analizi — aynı family içi sıralama (top-K rerank)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FineDetailSignature:
    motif_density: float = 0.0
    blob_density: float = 0.0
    edge_density: float = 0.0
    negative_space_ratio: float = 0.0
    dominant_motif_size: float = 0.0
    repeat_spacing: float = 0.0
    scale_similarity: float = 0.0
    palette_distance: float = 0.0
    hue_shift_score: float = 0.0
    curve_density: float = 0.0
    tile_repeat_confidence: float = 0.0
    crop_similarity_score: float = 0.0


def _f(rec: dict[str, Any], key: str, default: float = 0.0) -> float:
    tm = rec.get("texture_map") or {}
    if isinstance(tm, str):
        return default
    val = tm.get(key, rec.get(key, default))
    try:
        return float(val or default)
    except (TypeError, ValueError):
        return default


def extract_fine_signature(rec: dict[str, Any]) -> FineDetailSignature:
    """Mevcut texture_map / skor alanlarından hafif imza çıkar."""
    tm = rec.get("texture_map") or {}
    organic = _f(rec, "organic_blob_score")
    stripe = _f(rec, "stripe_score")
    scale = _f(rec, "scale_pattern_score")
    contrast = _f(rec, "contrast_score")
    return FineDetailSignature(
        motif_density=organic,
        blob_density=organic * 0.85,
        edge_density=min(1.0, contrast * 1.2),
        negative_space_ratio=max(0.0, 1.0 - organic - stripe * 0.5),
        dominant_motif_size=scale,
        repeat_spacing=scale * 0.7 + stripe * 0.3,
        scale_similarity=scale,
        palette_distance=1.0 - _f(rec, "palette_similarity", 0.5),
        hue_shift_score=_f(rec, "hue_shift", 0.0),
        curve_density=organic * 0.6 + scale * 0.4,
        tile_repeat_confidence=_f(rec, "patch_sim", rec.get("patch_score", 0.0)),
        crop_similarity_score=_f(rec, "crop_sim", 0.0),
    )


def _vec(sig: FineDetailSignature) -> np.ndarray:
    return np.array(
        [
            sig.motif_density,
            sig.blob_density,
            sig.edge_density,
            sig.negative_space_ratio,
            sig.dominant_motif_size,
            sig.repeat_spacing,
            sig.scale_similarity,
            1.0 - sig.palette_distance,
            sig.hue_shift_score,
            sig.curve_density,
            sig.tile_repeat_confidence,
            sig.crop_similarity_score,
        ],
        dtype=np.float32,
    )


def fine_detail_similarity(query: dict[str, Any], candidate: dict[str, Any]) -> float:
    """0..1 — aynı family içi ince benzerlik."""
    qv = _vec(extract_fine_signature(query))
    cv = _vec(extract_fine_signature(candidate))
    denom = (np.linalg.norm(qv) * np.linalg.norm(cv)) or 1.0
    cos = float(np.dot(qv, cv) / denom)
    return max(0.0, min(1.0, (cos + 1.0) / 2.0))


def rerank_search_results(
    query_rec: dict[str, Any],
    results: list[Any],
    *,
    top_k: int = 80,
    weight: float = 0.12,
) -> list[Any]:
    """Top-K ince fark rerank — exact/self sırasını bozmaz."""
    if not results or top_k <= 0:
        return results

    head = list(results[:top_k])
    tail = list(results[top_k:])

    for rec in head:
        is_self = bool(getattr(rec, "is_self_match", False))
        debug = getattr(rec, "debug", None) or {}
        if is_self or debug.get("protected_exact"):
            continue
        q_family = (
            query_rec.get("pattern_family")
            or (query_rec.get("texture_map") or {}).get("pattern_family")
            or ""
        )
        r_family = (
            getattr(rec, "pattern_family", "") or debug.get("pattern_family") or ""
        )
        if (
            q_family
            and r_family
            and q_family not in ("unknown", "")
            and r_family not in ("unknown", "")
            and q_family != r_family
        ):
            continue
        cand = _result_to_fine_dict(rec)
        fine = fine_detail_similarity(query_rec, cand)
        base = float(getattr(rec, "score", 0.0))
        new_score = max(0.0, min(1.0, base + weight * fine))
        rec.score = new_score
        if hasattr(rec, "score_percent"):
            rec.score_percent = round(new_score * 100, 1)
        debug["fine_detail_boost"] = round(fine, 4)
        rec.debug = debug

    head.sort(
        key=lambda r: (
            (
                0
                if getattr(r, "is_self_match", False)
                else (
                    1 if (getattr(r, "debug", None) or {}).get("protected_exact") else 2
                )
            ),
            -float(getattr(r, "score", 0.0)),
        ),
    )
    return head + tail


def _result_to_fine_dict(result: Any) -> dict[str, Any]:
    debug = getattr(result, "debug", None) or {}
    tm = debug.get("texture_map") or {}
    if not isinstance(tm, dict):
        tm = {}
    return {
        "texture_map": tm,
        "patch_sim": float(debug.get("patch_score", 0) or 0),
        "patch_score": float(debug.get("patch_score", 0) or 0),
        "palette_similarity": float(getattr(result, "palette_similarity", 0) or 0),
    }


def rerank_top_candidates(
    query_rec: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    top_k: int = 300,
    weight: float = 0.15,
) -> list[dict[str, Any]]:
    """Top adaylara ince fark boost uygula — exact adayları koru."""
    if not candidates:
        return candidates
    head = candidates[:top_k]
    tail = candidates[top_k:]
    for rec in head:
        if rec.get("protected_exact") or rec.get("is_self_match"):
            continue
        fine = fine_detail_similarity(query_rec, rec)
        base = float(rec.get("score", 0.0))
        rec["score"] = max(0.0, min(1.0, base + weight * fine))
        rec["fine_detail_boost"] = round(fine, 4)
    head.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
    return head + tail

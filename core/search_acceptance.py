"""Kanonik arama sonucu kabul kuralları.

Tek kaynak: SearchEngine, SearchResponse/UI ve istatistikler aynı kararı kullanır.
İnsan/cinsiyet semantik sonuçları genel desen eşiğinden ayrı değerlendirilir.
"""
from __future__ import annotations

from typing import Any

HUMAN_SEMANTIC_FLOOR = 0.18
ENTITY_SEMANTIC_FLOOR = 0.18


def result_passes_threshold(result: Any, threshold: float) -> bool:
    """Return the canonical display/search acceptance decision."""
    dbg = getattr(result, "debug", {}) or {}

    if dbg.get("face_gender_match"):
        return True

    if dbg.get("human_semantic_mode") or dbg.get("human_semantic_only"):
        try:
            score = float(
                dbg.get("human_semantic_score",
                        dbg.get("gender_visual_score", 0.0)) or 0.0
            )
        except (TypeError, ValueError):
            return False
        return score >= HUMAN_SEMANTIC_FLOOR

    if dbg.get("entity_semantic_mode") or dbg.get("entity_semantic_only"):
        try:
            score = float(
                dbg.get("entity_semantic_score", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            return False
        return score >= ENTITY_SEMANTIC_FLOOR

    if dbg.get("entity_evidence"):
        try:
            entity_score = float(dbg.get("entity_evidence_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            entity_score = 0.0
        if entity_score >= ENTITY_SEMANTIC_FLOOR:
            return True

    if dbg.get("protected_exact"):
        return True

    try:
        return float(getattr(result, "score", 0.0) or 0.0) >= float(threshold)
    except (TypeError, ValueError):
        return False

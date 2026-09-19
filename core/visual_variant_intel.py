"""Visual family / variant intelligence — Stage 2E (read-only).

Tags scale/density/color variants under the SAME concept.
Never creates new concepts (small Tiger ≠ Leopard).
"""
from __future__ import annotations

from typing import Any

from core.query_attribute_intel import (
    QueryAttributes,
    extract_query_attributes,
    score_attributes_against_dna,
)
from core.textile_terms import normalize_turkish


def variant_profile_from_attrs(attrs: QueryAttributes) -> dict[str, Any]:
    return {
        "scale": attrs.scale or "",
        "density": attrs.density or "",
        "colors": list(attrs.colors or []),
        "repeat": attrs.repeat or "",
        "orientation": attrs.orientation or "",
        "style": attrs.style or "",
        "motif": attrs.motif or "",
    }


def annotate_visual_variants(
    results: list[Any],
    query_text: str,
    *,
    attrs: QueryAttributes | None = None,
) -> list[Any]:
    """Attach variant debug under the queried concept; no score rewrite of siblings."""
    if not results:
        return results
    attrs = attrs or extract_query_attributes(query_text)
    profile = variant_profile_from_attrs(attrs)
    motif = normalize_turkish(attrs.motif or "")
    for rec in results:
        dbg = dict(getattr(rec, "debug", None) or {})
        meta = score_attributes_against_dna(attrs, rec) if attrs.has_attributes else {
            "applied": False,
            "aligned": True,
            "delta": 0.0,
        }
        # Sibling must never be framed as a "variant" of the queried leaf.
        is_variant = bool(meta.get("aligned")) and bool(attrs.has_attributes)
        dbg["visual_variant_intel"] = {
            "concept_motif": motif,
            "query_variants": profile,
            "is_same_concept_variant": is_variant,
            "attr_delta": float(meta.get("delta") or 0),
            "aligned": bool(meta.get("aligned", True)),
            "creates_new_concept": False,
        }
        rec.debug = dbg
    return results

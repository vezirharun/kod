"""Family / subfamily / variant graph helpers (read-only).

Similarity alone ≠ same family. Distinct taxonomy leaves (Tiger ≠ Leopard)
must never auto-merge. Variants stay under the same concept.
"""

from __future__ import annotations

from typing import Any


def may_merge_as_same_family(label_a: str, label_b: str) -> bool:
    """False when labels are known distinct concepts. Never auto-merge those."""
    a = " ".join(str(label_a or "").strip().split())
    b = " ".join(str(label_b or "").strip().split())
    if not a or not b:
        return False
    try:
        from core.canonical_correction import are_distinct_concepts
        from core.concept_query_normalize import concept_match_key

        if are_distinct_concepts(a, b):
            return False
        return concept_match_key(a) == concept_match_key(b)
    except Exception:
        return a.lower() == b.lower()


def family_snapshot(
    file_row: dict[str, Any] | None = None,
    texture_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reuse Pattern Family tree + DNA; no writes."""
    tm = dict(texture_map or {})
    row = dict(file_row or {})
    dna = tm.get("pattern_dna") if isinstance(tm.get("pattern_dna"), dict) else {}
    tree: dict[str, Any] = {}
    try:
        from core.pattern_family_tree import build_pattern_family_tree

        tree = build_pattern_family_tree(
            tm,
            ocr_text=str(row.get("ocr_text") or tm.get("ocr_text") or ""),
            filename=str(row.get("filename") or ""),
        )
    except Exception:
        tree = {}

    family = (
        str(row.get("pattern_family") or "")
        or str(tm.get("pattern_family") or "")
        or str(dna.get("motif_family") or dna.get("family") or "")
        or str(tree.get("root") or "")
    )
    sub = str(tree.get("branch") or dna.get("scale") or "")
    variant = {
        "scale": str(dna.get("scale") or tree.get("scale_branch") or ""),
        "color": str(tree.get("color_branch") or dna.get("color_family") or tm.get("color_family") or ""),
        "density": str(dna.get("density") or ""),
        "repeat": str(dna.get("repeat_type") or dna.get("repeat") or ""),
    }
    return {
        "family": family,
        "subfamily": sub,
        "variant": variant,
        "tree": tree,
        "dna_confidence": float(
            dna.get("pattern_dna_confidence") or dna.get("confidence") or 0
        ),
        "auto_merge_allowed": False,
        "note": "Benzerlik tek başına aynı aile demek değildir; distinct kavramlar birleştirilmez",
    }

"""Shared geometric concept resolution.

Stripe works today because query→family→needles→DNA is complete.
Other shapes must use the same mechanism, not per-word hacks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from core.textile_terms import normalize_turkish


@dataclass(frozen=True)
class GeometricConcept:
    concept_id: str
    family: str
    motif: str
    shape: str
    structure: str
    aliases: tuple[str, ...]
    label: str = "pattern"

    @property
    def is_leaf(self) -> bool:
        return self.motif not in ("", "geometric") and self.motif != self.family


_CONCEPTS: tuple[GeometricConcept, ...] = (
    GeometricConcept(
        "stripe", "stripe", "stripe", "line", "LINEAR_REPEAT",
        ("cizgi", "cizgili", "stripe", "striped", "line", "lines", "linear"),
        "pattern",
    ),
    GeometricConcept(
        "polka_dot", "polka_dot", "polka_dot", "dot", "REPEATED_DOTS",
        ("puantiye", "puantiyeli", "polka", "polka_dot", "polka dot", "dot", "dots",
         "nokta", "noktali", "benek"),
        "pattern",
    ),
    GeometricConcept(
        "square", "geometric", "square", "square", "GRID",
        ("kare", "square", "squares"),
        "pattern",
    ),
    GeometricConcept(
        "triangle", "geometric", "triangle", "triangle", "TESSELLATION",
        ("ucgen", "triangle", "triangles"),
        "pattern",
    ),
    GeometricConcept(
        "circle", "geometric", "circle", "circle", "REPEATED_SHAPES",
        ("daire", "circle", "circles", "yuvarlak"),
        "pattern",
    ),
    GeometricConcept(
        "rectangle", "geometric", "rectangle", "rectangle", "GRID",
        ("dikdortgen", "rectangle", "rectangles"),
        "pattern",
    ),
    GeometricConcept(
        "oval", "geometric", "oval", "oval", "REPEATED_SHAPES",
        ("oval", "ovals", "elips", "ellipse"),
        "pattern",
    ),
    GeometricConcept(
        "diamond", "geometric", "diamond", "diamond", "TESSELLATION",
        ("baklava", "elmas", "diamond", "diamonds", "rhombus"),
        "pattern",
    ),
    GeometricConcept(
        "zigzag", "geometric", "zigzag", "zigzag", "LINEAR_REPEAT",
        ("zikzak", "zigzag", "zig zag", "chevron"),
        "pattern",
    ),
    GeometricConcept(
        "wave", "geometric", "wave", "wave", "LINEAR_REPEAT",
        ("dalga", "wave", "waves", "wavy"),
        "pattern",
    ),
    GeometricConcept(
        "spiral", "geometric", "spiral", "spiral", "ORGANIC_REPEAT",
        ("spiral", "helezon"),
        "pattern",
    ),
    GeometricConcept(
        "star", "geometric", "star", "star", "REPEATED_SHAPES",
        ("yildiz", "star", "stars"),
        "pattern",
    ),
    GeometricConcept(
        "hexagon", "geometric", "hexagon", "hexagon", "TESSELLATION",
        ("altigen", "hexagon", "hexagons", "petek", "honeycomb"),
        "pattern",
    ),
    GeometricConcept(
        "pentagon", "geometric", "pentagon", "pentagon", "TESSELLATION",
        ("besgen", "pentagon", "pentagons"),
        "pattern",
    ),
    GeometricConcept(
        "plaid", "plaid_check", "tartan", "check", "GRID",
        ("ekose", "kareli", "tartan", "plaid", "check", "checked", "gingham"),
        "pattern",
    ),
    GeometricConcept(
        "geometric", "geometric", "geometric", "geometric", "STRUCTURED",
        ("geometrik", "geometric", "geo"),
        "pattern",
    ),
    GeometricConcept(
        "camouflage", "camouflage", "camouflage", "camouflage", "ORGANIC_REPEAT",
        ("kamuflaj", "camouflage", "camo"),
        "pattern",
    ),
)

_BY_ALIAS: dict[str, GeometricConcept] = {}
_BY_ID: dict[str, GeometricConcept] = {}
for _c in _CONCEPTS:
    _BY_ID[_c.concept_id] = _c
    for _a in (_c.concept_id, *_c.aliases):
        key = normalize_turkish(_a)
        if key and key not in _BY_ALIAS:
            _BY_ALIAS[key] = _c


def all_concepts() -> tuple[GeometricConcept, ...]:
    return _CONCEPTS


def concept_by_id(concept_id: str) -> GeometricConcept | None:
    return _BY_ID.get(str(concept_id or "").strip())


def resolve_geometric_query(text: str) -> GeometricConcept | None:
    """Exact alias/token match. Does not use substring (kare vs kareli)."""
    raw = str(text or "").strip()
    if not raw:
        return None
    norm = normalize_turkish(raw)
    if not norm:
        return None
    hit = _BY_ALIAS.get(norm)
    if hit:
        return hit
    tokens = [t for t in norm.replace("-", " ").split() if len(t) >= 2]
    # Longest token first so "polka_dot" phrases win if split oddly.
    for tok in sorted(tokens, key=len, reverse=True):
        found = _BY_ALIAS.get(tok)
        if found:
            return found
    return None


def family_hints_for_query(text: str) -> dict[str, str]:
    geo = resolve_geometric_query(text)
    if not geo:
        return {}
    out = {"pattern_family": geo.family, "pattern_type": geo.motif}
    if geo.structure:
        out["structure"] = geo.structure
    if geo.shape:
        out["shape"] = geo.shape
    return out


def retrieval_needles(concept: GeometricConcept | None) -> list[str]:
    if concept is None:
        return []
    return list(dict.fromkeys([concept.concept_id, *concept.aliases]))


def _dna_blob(rec: dict[str, Any], texture_map: dict[str, Any]) -> str:
    dna = texture_map.get("pattern_dna") if isinstance(texture_map.get("pattern_dna"), dict) else {}
    sem = texture_map.get("semantic_tags") if isinstance(texture_map.get("semantic_tags"), dict) else {}
    motifs = sem.get("motifs") if isinstance(sem.get("motifs"), list) else []
    parts = [
        rec.get("pattern_family"),
        rec.get("pattern_type") or rec.get("pattern_subtype"),
        texture_map.get("pattern_family"),
        texture_map.get("pattern_type"),
        dna.get("family"),
        dna.get("motif"),
        dna.get("motif_class"),
        dna.get("structure"),
        dna.get("repeat_type") or dna.get("repeat_class"),
        sem.get("motif"),
        " ".join(str(m) for m in motifs),
    ]
    return normalize_turkish(" ".join(str(p or "") for p in parts))


def dna_concept_alignment(
    query_text: str,
    rec: dict[str, Any],
    texture_map: dict[str, Any] | None = None,
) -> tuple[float, str]:
    """Leaf motif beats parent 'geometric'. Sibling shapes do not steal rank."""
    geo = resolve_geometric_query(query_text)
    if geo is None:
        return 0.0, ""
    tm = texture_map if isinstance(texture_map, dict) else {}
    blob = _dna_blob(rec or {}, tm)
    if not blob:
        return 0.0, ""
    alias_hits = [
        normalize_turkish(a)
        for a in (geo.motif, geo.concept_id, geo.shape, *geo.aliases[:8])
        if normalize_turkish(a)
    ]
    if any(a and a in blob for a in alias_hits):
        return 0.90, geo.motif
    if geo.is_leaf:
        for other in _CONCEPTS:
            if other.concept_id == geo.concept_id or not other.is_leaf:
                continue
            if other.family not in (geo.family, "geometric") and geo.family != "geometric":
                continue
            other_m = normalize_turkish(other.motif)
            if other_m and other_m in blob and other_m not in alias_hits:
                return 0.0, ""
        rec_fam = normalize_turkish(
            str(rec.get("pattern_family") or tm.get("pattern_family") or "")
        )
        fam_n = normalize_turkish(geo.family)
        if rec_fam == fam_n == "geometric":
            return 0.40, geo.family
        return 0.0, ""
    fam_n = normalize_turkish(geo.family)
    if fam_n and fam_n in blob:
        return 0.78, geo.family
    return 0.0, ""


def sibling_concepts(concept_id: str) -> list[str]:
    geo = concept_by_id(concept_id)
    if not geo:
        return []
    return [
        c.concept_id
        for c in _CONCEPTS
        if c.concept_id != geo.concept_id and c.family == geo.family and c.is_leaf
    ]

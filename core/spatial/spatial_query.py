"""Turkish OBJECT_A + RELATION + OBJECT_B parser. Detector labels only — not CLIP."""
from __future__ import annotations

from typing import Any

from core.object_search import _EXPAND, _fold, resolve_object_class
from core.spatial.spatial_relation import Relation

# Spatial-only tokens (not added to object_search: "çiçek"/"gül" must stay textile queries).
_SPATIAL_ALIASES: dict[str, str] = {
    "cicek": "flower",
    "ciceg": "flower",
    "cicegin": "flower",
    "flower": "flower",
    "yaprak": "leaf",
    "leaf": "leaf",
    "gul": "rose",
    "rose": "rose",
    "leopar": "leopard",
    "leopard": "leopard",
    "kelebek": "butterfly",
    "butterfly": "butterfly",
    "zebra": "zebra",
    "yilan": "snake",
    "snake": "snake",
}
_SPATIAL_EXPAND: dict[str, set[str]] = {
    "flower": {"flower", "potted plant"},
    "rose": {"rose", "flower"},
    "leaf": {"leaf"},
    "leopard": {"leopard"},
    "butterfly": {"butterfly"},
    "zebra": {"zebra"},
    "snake": {"snake"},
}

# Longest phrase first. Folded Turkish (ı→i, diacritics stripped).
_REL_PHRASES: tuple[tuple[str, Relation], ...] = (
    ("yani basinda", Relation.NEAR),
    ("sol tarafinda", Relation.LEFT_OF),
    ("sag tarafinda", Relation.RIGHT_OF),
    ("alt tarafinda", Relation.BELOW),
    ("ust tarafinda", Relation.ABOVE),
    ("sol ustte", Relation.UPPER_LEFT),
    ("sag ustte", Relation.UPPER_RIGHT),
    ("sol altta", Relation.LOWER_LEFT),
    ("sag altta", Relation.LOWER_RIGHT),
    ("yaninda", Relation.NEAR),
    ("solunda", Relation.LEFT_OF),
    ("saginda", Relation.RIGHT_OF),
    ("ustunde", Relation.ABOVE),
    ("uzerinde", Relation.ABOVE),
    ("altinda", Relation.BELOW),
    ("merkezde", Relation.CENTER),
    ("icinde", Relation.INSIDE),
    ("kapsar", Relation.CONTAINS),
    ("ortusur", Relation.OVERLAPS),
    ("ortusen", Relation.OVERLAPS),
    ("yakininda", Relation.NEAR),
    ("uzağinda", Relation.FAR),
    ("uzaginda", Relation.FAR),
)


def _token_stems(token: str) -> list[str]:
    t = _fold(token).strip()
    stems = [t]
    for suf in ("nin", "nun", "gin", "in", "un"):
        if len(t) > len(suf) + 1 and t.endswith(suf):
            stems.append(t[: -len(suf)])
    return stems


def resolve_spatial_class(token: str) -> dict[str, Any] | None:
    folded = _fold(token)
    if not folded:
        return None
    mapped = resolve_object_class(folded)
    if mapped:
        return mapped
    label = _SPATIAL_ALIASES.get(folded)
    if not label:
        return None
    labels = set(_SPATIAL_EXPAND.get(label, {label}))
    extra = _EXPAND.get(label)
    if extra:
        labels |= set(extra)
    return {"label": label, "labels": labels}


def parse_spatial_query(text: str) -> dict[str, Any] | None:
    """Return OBJECT_A (anchor) + RELATION + OBJECT_B (located), or None if not spatial."""
    q = _fold(text)
    if not q:
        return None
    hit_rel: Relation | None = None
    hit_phrase = ""
    for phrase, rel in _REL_PHRASES:
        if phrase in q:
            hit_rel, hit_phrase = rel, phrase
            break
    if not hit_rel:
        return None
    left, right = q.split(hit_phrase, 1)
    a_raw = _fold(left).strip().split()[-1] if _fold(left).strip() else ""
    b_raw = _fold(right).strip().split()[0] if _fold(right).strip() else ""
    if a_raw and not b_raw:
        left_toks = _fold(left).strip().split()
        if len(left_toks) >= 2:
            b_raw, a_raw = left_toks[0], left_toks[-1]
    if not a_raw or not b_raw:
        return None
    obj_a = next((resolve_spatial_class(s) for s in _token_stems(a_raw) if resolve_spatial_class(s)), None)
    obj_b = next((resolve_spatial_class(s) for s in _token_stems(b_raw) if resolve_spatial_class(s)), None)
    if not obj_a or not obj_b:
        return None
    return {
        "kind": "spatial",
        "relation": hit_rel,
        "object_a": obj_a,
        "object_b": obj_b,
        "phrase": hit_phrase,
        "source": "object_bbox",
        "clip_as_relation": False,
    }

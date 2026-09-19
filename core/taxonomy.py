"""Tekstil desen taksonomisi — family, subtype, confidence."""

from __future__ import annotations

from typing import Any

PATTERN_FAMILIES = frozenset(
    {
        "animal_print",
        "floral",
        "marble_abstract",
        "geometric",
        "plaid_check",
        "stripe",
        "paisley",
        "lace",
        "monogram_logo",
        "baroque",
        "chain",
        "scarf_border",
        "texture_ground",
        "document",
        "unknown",
        "typography_text",
    }
)

ANIMAL_SUBTYPES = frozenset(
    {
        "leopard",
        "zebra",
        "snake",
        "tiger",
        "crocodile",
        "cow",
        "giraffe",
        "cheetah",
        "jaguar",
        "mixed_animal",
    }
)

FLORAL_SUBTYPES = frozenset(
    {
        "rose",
        "daisy",
        "leaf",
        "small_floral",
        "ditsy_floral",
        "mixed_floral",
        "big_flower",
        "tropical_flower",
    }
)

MARBLE_SUBTYPES = frozenset(
    {
        "marble",
        "watercolor",
        "fluid_art",
        "color_swirl",
        "smoke",
        "ink",
        "abstract_paint",
    }
)


def normalize_family(family: str) -> str:
    if family in ("marble", "abstract"):
        return "marble_abstract"
    return family or "unknown"


def infer_subtype(
    pattern_family: str,
    animal_print_type: str = "",
    *,
    organic_blob: float = 0.0,
    stripe: float = 0.0,
    scale: float = 0.0,
    contrast: float = 0.0,
    filename: str = "",
    texture_map: dict[str, Any] | None = None,
) -> tuple[str, float]:
    """pattern_subtype ve confidence döndür. Dosya adı tek başına karar vermez."""
    fam = normalize_family(pattern_family)
    fname = (filename or "").lower()
    conf = 0.55

    if fam == "animal_print":
        sub = animal_print_type or "mixed_animal"
        if sub in ANIMAL_SUBTYPES:
            conf = 0.72 if organic_blob > 0.2 else 0.58
        return sub, conf

    if fam == "floral":
        if any(
            t in fname
            for t in ("mini", "ditsy", "small", "kucuk", "minik", "citir", "çıtır")
        ):
            return "small_floral", 0.65
        if organic_blob < 0.35 and scale > 0.35:
            return "small_floral", 0.62
        if organic_blob > 0.55:
            return "big_flower", 0.60
        if any(t in fname for t in ("rose", "gul", "gül")):
            return "rose", 0.55
        if any(t in fname for t in ("daisy", "papatya")):
            return "daisy", 0.55
        return "mixed_floral", 0.58

    if fam == "marble_abstract":
        if contrast > 0.25 and organic_blob > 0.3:
            return "fluid_art", 0.65
        if any(t in fname for t in ("marble", "mermer")):
            return "marble", 0.55
        if any(t in fname for t in ("watercolor", "suluboya")):
            return "watercolor", 0.55
        return "abstract_paint", 0.58

    if fam == "stripe":
        if stripe > 0.5:
            return "vertical_stripe", 0.65
        return "wavy_stripe", 0.55

    if fam == "plaid_check":
        return "plaid", 0.60

    if fam == "paisley":
        return "paisley", 0.60

    if fam == "lace":
        return "lace", 0.60

    if fam == "geometric":
        return "repeat_geo", 0.58

    if fam == "monogram_logo":
        return "monogram", 0.55

    if fam == "baroque":
        return "baroque", 0.55

    if fam == "chain":
        return "chain", 0.55

    if texture_map and texture_map.get("pattern_subtype"):
        return str(texture_map["pattern_subtype"]), float(
            texture_map.get("classification_confidence", 0.5)
        )

    return "", 0.40 if fam == "unknown" else 0.50

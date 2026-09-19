"""Build persistent semantic textile tags from the indexed visual metadata."""

from __future__ import annotations

from typing import Any

from core.semantic_dictionary import detect_brand_references
from core.textile_terms import normalize_turkish

SEMANTIC_TAG_VERSION = 1

FAMILY_TAGS: dict[str, dict[str, tuple[str, ...]]] = {
    "typography_text": {
        "motifs": ("typography", "text", "letter", "word"),
        "styles": ("repeated text", "brand text style"),
        "themes": ("communication",),
    },
    "monogram_logo": {
        "motifs": ("monogram", "logo", "letter", "emblem", "symbol"),
        "styles": ("luxury", "repeat", "brand pattern", "interlocking"),
        "themes": ("fashion",),
    },
    "animal_print": {
        "motifs": ("animal print",),
        "styles": ("organic", "repeat"),
        "themes": ("animal", "wildlife"),
    },
    "floral": {
        "motifs": ("floral", "flower", "leaf", "botanical", "bloom"),
        "styles": ("organic", "repeat"),
        "themes": ("garden", "nature"),
    },
    "geometric": {
        "motifs": ("geometric", "geometry", "mosaic"),
        "styles": ("repeat", "structured"),
        "themes": (),
    },
    "plaid_check": {
        "motifs": ("plaid", "check", "tartan", "grid"),
        "styles": ("repeat", "structured"),
        "themes": (),
    },
    "stripe": {
        "motifs": ("stripe", "line"),
        "styles": ("repeat", "linear"),
        "themes": (),
    },
    "polka_dot": {
        "motifs": ("polka dot", "dot", "circle"),
        "styles": ("repeat",),
        "themes": (),
    },
    "paisley": {
        "motifs": ("paisley", "boteh", "medallion"),
        "styles": ("oriental", "ornamental", "repeat"),
        "themes": ("ethnic",),
    },
    "baroque": {
        "motifs": ("baroque", "ornament", "scroll", "acanthus"),
        "styles": ("luxury", "ornate", "gold ornament"),
        "themes": ("classical",),
    },
    "chain": {
        "motifs": ("chain", "link", "gold chain"),
        "styles": ("luxury", "ornamental", "repeat"),
        "themes": (),
    },
    "lace": {
        "motifs": ("lace", "openwork", "embroidery"),
        "styles": ("delicate", "repeat"),
        "themes": (),
    },
    "marble_abstract": {
        "motifs": ("marble", "stone", "fluid art"),
        "styles": ("abstract", "organic"),
        "themes": (),
    },
    "texture_ground": {
        "motifs": ("texture", "ground"),
        "styles": ("allover",),
        "themes": (),
    },
}

SUBTYPE_TAGS: dict[str, tuple[str, ...]] = {
    "leopard": ("leopard", "leopar", "spot", "rosette"),
    "zebra": ("zebra", "stripe"),
    "snake": ("snake", "snake skin", "scale"),
    "tiger": ("tiger", "stripe"),
    "crocodile": ("crocodile", "scale"),
    "cow": ("cow", "spot"),
    "giraffe": ("giraffe", "spot"),
    "rose": ("rose", "flower"),
    "daisy": ("daisy", "flower"),
    "leaf": ("leaf", "botanical"),
    "small_floral": ("small floral", "mini floral", "ditsy"),
    "ditsy_floral": ("ditsy floral", "small floral"),
    "military_camo": ("military", "camouflage", "camo", "combat"),
    "digital_camo": ("digital camo", "camouflage", "military"),
    "urban_camo": ("urban camo", "camouflage"),
    "camouflage": ("camouflage", "camo", "woodland", "forest camo"),
    "denim": ("denim", "jean", "fabric texture"),
    "knit": ("knit", "knitted", "weave", "fabric texture"),
}


def build_semantic_tags(
    texture_map: dict[str, Any] | None,
    *,
    category_path: str = "",
    ocr_text: str = "",
    filename: str = "",
) -> dict[str, Any]:
    tm = dict(texture_map or {})
    family = str(tm.get("pattern_family") or "unknown")
    subtype = str(
        tm.get("animal_print_type") or tm.get("pattern_subtype") or ""
    )
    trusted_category = bool(
        tm.get("user_labeled")
        or tm.get("manual_category_path")
        or str(tm.get("category_source") or "") in ("manual_user", "gold_dataset")
    )
    # Leopard is a high-risk label: visual texture/family heuristics alone are not
    # semantic evidence.  Keep the broad animal family, but do not manufacture a
    # Leopard tag unless an admin/gold label supplied it.
    if family == "animal_print" and subtype == "leopard" and not trusted_category:
        leaf_repeat = (
            str(tm.get("color_family") or "") in ("black_white", "grayscale")
            and float(tm.get("floral_score", 0) or 0) >= 0.35
            and float(tm.get("scale_pattern_score", 0) or 0) >= 0.35
            and float(tm.get("repeat_density", 0) or 0) >= 0.15
        )
        if leaf_repeat:
            family, subtype = "floral", "leaf"
        else:
            family, subtype = "unknown", ""
    base = FAMILY_TAGS.get(family, {})
    motifs = list(base.get("motifs", ()))
    styles = list(base.get("styles", ()))
    themes = list(base.get("themes", ()))
    motifs.extend(SUBTYPE_TAGS.get(subtype, ()))

    if float(tm.get("repeat_density", 0) or 0) >= 0.15:
        styles.append("repeat")
    if float(tm.get("stripe_score", 0) or 0) >= 0.42:
        motifs.extend(("stripe", "line"))
    if family == "geometric" or float(tm.get("scale_pattern_score", 0) or 0) >= 0.35:
        motifs.append("geometric")
    if category_path:
        motifs.extend(part.strip() for part in category_path.split("/") if part.strip())
    if ocr_text and any(ch.isalpha() for ch in ocr_text):
        motifs.extend(("typography", "text", "letter", "word"))

    evidence_text = normalize_turkish(f"{filename} {ocr_text}")
    brands = detect_brand_references(evidence_text)
    semantic_evidence = " ".join(
        str(v).lower()
        for v in [
            filename,
            ocr_text,
            *motifs,
            *styles,
            *themes,
        ]
        if v
    )
    semantic_evidence_norm = normalize_turkish(semantic_evidence)
    logo_terms = (
        "lv", "louis vuitton", "monogram", "logo", "logo repeat",
        "repeated logo", "emblem", "amblem", "arma", "symbol pattern",
    )
    text_terms = ("typography", "text", "letter", "word", "yazi", "yazili")
    semantic_confidence = float(tm.get("classification_confidence", 0) or 0)
    if (brands or any(term in semantic_evidence_norm for term in logo_terms)) and family in ("", "unknown"):
        family = "monogram_logo"
        subtype = subtype or "repeated_logo"
        semantic_confidence = max(semantic_confidence, 0.72 if brands else 0.62)
        base = FAMILY_TAGS.get(family, {})
        motifs = list(base.get("motifs", ())) + motifs
        styles = list(base.get("styles", ())) + styles
        themes = list(base.get("themes", ())) + themes
    elif any(term in semantic_evidence_norm for term in text_terms) and family in ("", "unknown"):
        family = "typography_text"
        subtype = subtype or "repeated_text"
        semantic_confidence = max(semantic_confidence, 0.68)
        base = FAMILY_TAGS.get(family, {})
        motifs = list(base.get("motifs", ())) + motifs
        styles = list(base.get("styles", ())) + styles
        themes = list(base.get("themes", ())) + themes
    if brands:
        styles.extend(("brand pattern", "luxury"))

    color_family = str(tm.get("color_family") or "")
    if color_family in ("black_white", "grayscale"):
        styles.append("minimal")
    repeat_type = "repeat" if float(tm.get("repeat_density", 0) or 0) >= 0.15 else ""
    text_present = bool(ocr_text and any(ch.isalpha() for ch in ocr_text)) or family == "typography_text"
    logo_present = family == "monogram_logo" or "logo" in styles or bool(brands)
    material_hint = ""
    if "jakar" in semantic_evidence_norm or "jacquard" in semantic_evidence_norm:
        material_hint = "jacquard"
        motifs.append("jacquard")
        styles.append("woven")
    motifs = _unique(motifs)
    styles = _unique(styles)

    return {
        "version": SEMANTIC_TAG_VERSION,
        "family": family,
        "subtype": subtype,
        "subfamily": subtype,
        "motif": motifs[0] if motifs else "",
        "motifs": motifs,
        "repeat_type": repeat_type,
        "color_family": color_family,
        "style": styles[0] if styles else "",
        "styles": styles,
        "text_present": text_present,
        "logo_present": logo_present,
        "logo_detected": logo_present,
        "text_detected": text_present,
        "brand_style": brands[0] if brands else ("luxury" if "luxury" in styles else ""),
        "material_hint": material_hint,
        "themes": _unique(themes),
        "brand_references": brands,
        "objects": _unique(_objects_from_category(category_path)),
        "textures": _unique(_texture_tags(family, subtype)),
        "confidence": round(semantic_confidence, 4),
        "source": "indexed_visual_metadata",
    }


def flatten_semantic_tags(tags: dict[str, Any] | None) -> list[str]:
    if not isinstance(tags, dict):
        return []
    values: list[str] = []
    for key in (
        "family", "subtype", "subfamily", "motif", "motifs", "repeat_type",
        "color_family", "style", "styles", "themes",
        "brand_references", "objects", "textures",
    ):
        value = tags.get(key)
        if isinstance(value, list):
            values.extend(str(item) for item in value if item)
        elif value:
            values.append(str(value))
    return _unique(values)


def _objects_from_category(path: str) -> list[str]:
    parent, _, child = path.partition("/")
    if parent in ("Object", "Animal") and child:
        return [child]
    return []


def _texture_tags(family: str, subtype: str) -> list[str]:
    if family == "texture_ground":
        return [subtype or "fabric texture", "textile texture"]
    if family in ("lace", "plaid_check", "stripe"):
        return ["textile texture"]
    return []


def _unique(values) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))

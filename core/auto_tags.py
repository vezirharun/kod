"""AI otomatik etiketler — Pattern DNA + semantic analizden."""

from __future__ import annotations

from typing import Any

from core.textile_terms import normalize_turkish

AUTO_TAG_VERSION = 1

# Kanonik etiketler (arama + gösterim)
CANONICAL_AUTO_TAGS = (
    "Floral",
    "Geometric",
    "Luxury",
    "Women",
    "Kids",
    "Abstract",
    "Premium",
    "Animal",
    "Modern",
    "Vintage",
)

# Sorgu eşanlamlıları (küçük harf anahtar → kanonik etiket)
AUTO_TAG_SYNONYMS: dict[str, str] = {
    "floral": "Floral",
    "cicek": "Floral",
    "çiçek": "Floral",
    "flower": "Floral",
    "botanical": "Floral",
    "geometric": "Geometric",
    "geometrik": "Geometric",
    "geo": "Geometric",
    "luxury": "Luxury",
    "luks": "Luxury",
    "lüks": "Luxury",
    "premium": "Premium",
    "luks desen": "Luxury",
    "women": "Women",
    "woman": "Women",
    "kadin": "Women",
    "kadın": "Women",
    "bayan": "Women",
    "kids": "Kids",
    "kid": "Kids",
    "cocuk": "Kids",
    "çocuk": "Kids",
    "children": "Kids",
    "abstract": "Abstract",
    "soyut": "Abstract",
    "animal": "Animal",
    "hayvan": "Animal",
    "leopar": "Animal",
    "leopard": "Animal",
    "zebra": "Animal",
    "modern": "Modern",
    "minimal": "Modern",
    "contemporary": "Modern",
    "vintage": "Vintage",
    "retro": "Vintage",
    "classic": "Vintage",
    "barok": "Vintage",
    "baroque": "Vintage",
    "cocuk deseni": "Kids",
    "çocuk deseni": "Kids",
    "cocuk": "Kids",
    "çocuk": "Kids",
}


def _f(tm: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(tm.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _norm_join(*parts: Any) -> str:
    return normalize_turkish(" ".join(str(p) for p in parts if p))


def build_auto_tags(texture_map: dict[str, Any] | None) -> list[str]:
    """Pattern DNA + semantic_tags → Floral, Luxury, Animal, …"""
    tm = dict(texture_map or {})
    sem = tm.get("semantic_tags") if isinstance(tm.get("semantic_tags"), dict) else {}
    dna = tm.get("pattern_dna") if isinstance(tm.get("pattern_dna"), dict) else {}

    family = str(tm.get("pattern_family") or sem.get("family") or dna.get("family") or "")
    subtype = str(
        tm.get("animal_print_type")
        or tm.get("pattern_subtype")
        or sem.get("subtype")
        or dna.get("subfamily")
        or ""
    )
    styles = [str(x).lower() for x in (sem.get("styles") or []) if x]
    themes = [str(x).lower() for x in (sem.get("themes") or []) if x]
    motifs = [str(x).lower() for x in (sem.get("motifs") or []) if x]
    brands = [str(x).lower() for x in (sem.get("brand_references") or []) if x]
    category = str(
        tm.get("manual_category_path")
        or tm.get("category_path")
        or ""
    ).lower()
    blob = _norm_join(
        family,
        subtype,
        sem.get("style"),
        sem.get("motif"),
        dna.get("style"),
        dna.get("motif"),
        dna.get("designer_style"),
        category,
        " ".join(styles),
        " ".join(motifs),
        " ".join(themes),
    )

    tags: list[str] = []

    # Floral
    if (
        family == "floral"
        or _f(tm, "floral_score") >= 0.35
        or any(x in blob for x in ("floral", "flower", "cicek", "botanical", "bloom"))
    ):
        tags.append("Floral")

    # Geometric
    if (
        family == "geometric"
        or _f(tm, "scale_pattern_score") >= 0.40
        or any(x in blob for x in ("geometric", "geometry", "mosaic", "grid", "check"))
    ):
        tags.append("Geometric")

    # Animal
    if (
        family == "animal_print"
        or _f(tm, "animal_score") >= 0.55
        or any(x in blob for x in ("animal", "leopard", "leopar", "zebra", "snake", "tiger"))
    ):
        tags.append("Animal")

    # Abstract
    if (
        family in ("marble_abstract",)
        or any(x in blob for x in ("abstract", "soyut", "marble", "fluid", "watercolor"))
    ):
        tags.append("Abstract")

    # Luxury
    luxury_signal = (
        family == "monogram_logo"
        or family in ("baroque", "chain")
        or bool(brands)
        or "luxury" in styles
        or sem.get("logo_present")
        or any(x in blob for x in ("luxury", "luks", "monogram", "logo", "brand"))
    )
    if luxury_signal:
        tags.append("Luxury")

    # Premium (yüksek güven + lüks sinyal)
    conf = max(
        _f(tm, "classification_confidence"),
        float(sem.get("confidence", 0) or 0),
        float(dna.get("confidence", 0) or 0),
    )
    if luxury_signal and conf >= 0.65:
        tags.append("Premium")
    elif family in ("baroque", "chain", "lace") and conf >= 0.55:
        tags.append("Premium")

    # Modern
    if any(
        x in blob or x in styles
        for x in ("modern", "minimal", "contemporary", "clean", "geometric")
    ) and "baroque" not in blob and family not in ("baroque", "paisley"):
        if "Modern" not in tags:
            tags.append("Modern")

    # Vintage
    if any(
        x in blob or x in styles or x in motifs
        for x in ("vintage", "retro", "classic", "baroque", "ornament", "paisley", "oriental")
    ) or family in ("baroque", "paisley"):
        tags.append("Vintage")

    # Women
    if any(
        x in category
        for x in ("women", "woman", "kadin", "kadın", "bayan", "fashion", "lingerie")
    ) or any(x in themes for x in ("fashion",)) or family in ("lace", "floral"):
        if "Kids" not in tags:
            tags.append("Women")

    # Kids
    if any(
        x in category for x in ("kids", "kid", "child", "children", "cocuk", "çocuk", "bebek")
    ) or (
        _f(tm, "floral_score") >= 0.25
        and str(tm.get("color_family") or "") in ("neon_multicolor", "yellow", "red_pink")
        and family in ("polka_dot", "floral", "geometric")
    ):
        tags.append("Kids")
        if "Women" in tags:
            tags.remove("Women")

    # Sıralı tekilleştir — kanonik sıra
    ordered = [t for t in CANONICAL_AUTO_TAGS if t in tags]
    return ordered


def flatten_auto_tags(texture_map: dict[str, Any] | None) -> list[str]:
    if not isinstance(texture_map, dict):
        return []
    raw = texture_map.get("auto_tags")
    if isinstance(raw, list):
        return [str(x) for x in raw if x]
    # Geriye dönük: yoksa üret (arama blob için)
    return build_auto_tags(texture_map)


def resolve_auto_tag_query(token: str) -> str:
    key = normalize_turkish((token or "").strip().lower())
    if not key:
        return ""
    if key in AUTO_TAG_SYNONYMS:
        return AUTO_TAG_SYNONYMS[key]
    for tag in CANONICAL_AUTO_TAGS:
        if normalize_turkish(tag) == key:
            return tag
    return ""


def expand_auto_tag_query(query: str) -> list[str]:
    q = normalize_turkish((query or "").strip().lower())
    if not q:
        return []
    out: list[str] = []
    # Tam ifade
    tag = resolve_auto_tag_query(q)
    if tag:
        out.append(tag)
    # Token
    for part in q.split():
        t = resolve_auto_tag_query(part)
        if t:
            out.append(t)
    # Alt string
    for key, tag in AUTO_TAG_SYNONYMS.items():
        if key in q:
            out.append(tag)
    return list(dict.fromkeys(out))


def auto_tag_match_score(query: str, texture_map: dict[str, Any] | None) -> float:
    tags = {normalize_turkish(t) for t in flatten_auto_tags(texture_map)}
    if not tags:
        return 0.0
    wanted = expand_auto_tag_query(query)
    if not wanted:
        return 0.0
    matched = sum(1 for tag in wanted if normalize_turkish(tag) in tags)
    if matched == len(wanted) and matched >= 2:
        return 0.94
    if matched == len(wanted):
        return 0.92
    if matched > 0:
        return 0.78 + 0.06 * matched
    return 0.0

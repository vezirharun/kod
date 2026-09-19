"""Pattern Family ağacı — marka kök + renk/ölçek dalları."""

from __future__ import annotations

from typing import Any

from core.brand_aliases import expand_brand_terms, resolve_brand_alias
from core.textile_terms import normalize_turkish

# color_family → kullanıcı dostu dal adı
BRANCH_COLOR_LABELS: dict[str, str] = {
    "brown_tan": "Brown",
    "beige": "Beige",
    "camel": "Camel",
    "cream": "Cream",
    "black_white": "Black",
    "grayscale": "Gray",
    "gold": "Gold",
    "yellow": "Gold",
    "blue": "Blue",
    "red_pink": "Red",
    "burgundy": "Burgundy",
    "green": "Green",
    "khaki": "Khaki",
    "orange": "Orange",
    "neon_multicolor": "Multicolor",
}

BRAND_DISPLAY: dict[str, str] = {
    "louis vuitton": "Louis Vuitton",
    "gucci": "Gucci",
    "yves saint laurent": "Yves Saint Laurent",
    "christian dior": "Christian Dior",
    "fendi": "Fendi",
    "burberry": "Burberry",
    "michael kors": "Michael Kors",
    "dolce gabbana": "Dolce & Gabbana",
    "celine": "Celine",
    "chanel": "Chanel",
    "hermes": "Hermès",
    "prada": "Prada",
    "versace": "Versace",
    "amiri": "Amiri",
}


def _brand_from_map(texture_map: dict[str, Any], *, ocr_text: str = "", filename: str = "") -> str:
    sem = texture_map.get("semantic_tags") if isinstance(texture_map.get("semantic_tags"), dict) else {}
    dna = texture_map.get("pattern_dna") if isinstance(texture_map.get("pattern_dna"), dict) else {}
    # brand_references: louis_vuitton
    for ref in sem.get("brand_references") or []:
        key = str(ref).replace("_", " ").strip().lower()
        alias = resolve_brand_alias(key) or key
        if alias in BRAND_DISPLAY:
            return alias
    designer = str(dna.get("designer_style") or dna.get("brand_style") or "").strip()
    if designer:
        alias = resolve_brand_alias(designer) or normalize_turkish(designer)
        if alias in BRAND_DISPLAY or resolve_brand_alias(alias):
            return resolve_brand_alias(alias) or alias
    blob = f"{ocr_text} {filename} {designer}"
    brands = expand_brand_terms(blob)
    if brands:
        return brands[0]
    return ""


def _color_branch(texture_map: dict[str, Any]) -> str:
    ci = texture_map.get("color_index") if isinstance(texture_map.get("color_index"), dict) else {}
    fam = str(
        (ci or {}).get("color_family")
        or texture_map.get("color_family")
        or ""
    )
    if fam in BRANCH_COLOR_LABELS:
        return BRANCH_COLOR_LABELS[fam]
    # palette'den tahmin
    for label in (ci or {}).get("palette") or texture_map.get("dominant_palette") or []:
        low = str(label).lower()
        if low in ("siyah", "black"):
            return "Black"
        if low in ("beyaz", "white"):
            return "White"
        if low in ("kahverengi", "brown", "camel", "bej", "beige"):
            return "Brown" if low in ("kahverengi", "brown") else low.title()
        if low in ("altin", "altın", "gold", "golden"):
            return "Gold"
    if fam:
        return fam.replace("_", " ").title()
    return ""


def _scale_branch(texture_map: dict[str, Any]) -> str:
    dna = texture_map.get("pattern_dna") if isinstance(texture_map.get("pattern_dna"), dict) else {}
    scale = str(dna.get("scale") or "").lower()
    if scale in ("large", "macro", "buyuk", "büyük"):
        return "Large Repeat"
    if scale in ("small", "fine", "mini", "kucuk", "küçük"):
        return "Small Repeat"
    try:
        sc = float(texture_map.get("scale_pattern_score", 0) or 0)
        dens = float(texture_map.get("repeat_density", 0) or 0)
    except (TypeError, ValueError):
        sc, dens = 0.0, 0.0
    if sc >= 0.45:
        return "Large Repeat"
    if sc and sc < 0.28 and dens >= 0.15:
        return "Small Repeat"
    return ""


def family_root_label(texture_map: dict[str, Any], *, ocr_text: str = "", filename: str = "") -> str:
    brand = _brand_from_map(texture_map, ocr_text=ocr_text, filename=filename)
    if brand:
        return BRAND_DISPLAY.get(brand, brand.title())
    pf = str(texture_map.get("pattern_family") or "")
    if pf and pf not in ("unknown", ""):
        return pf.replace("_", " ").title()
    return ""


def build_pattern_family_tree(
    texture_map: dict[str, Any],
    *,
    ocr_text: str = "",
    filename: str = "",
) -> dict[str, Any]:
    """
    Louis Vuitton
     ├── Brown
     ├── Black
     ├── Large Repeat
     └── Small Repeat
    """
    root = family_root_label(texture_map, ocr_text=ocr_text, filename=filename)
    color = _color_branch(texture_map)
    scale = _scale_branch(texture_map)
    branches = [b for b in (color, scale) if b]
    # Tek birincil dal: renk tercih, yoksa ölçek
    primary = color or scale or ""
    path = root
    if primary and root:
        path = f"{root} / {primary}"
    elif primary:
        path = primary

    tree = {
        "version": 1,
        "root": root,
        "branch": primary,
        "color_branch": color,
        "scale_branch": scale,
        "branches": branches,
        "path": path,
        "label": path or root,
    }
    return tree


def format_family_tree_text(tree: dict[str, Any] | None, siblings: list[str] | None = None) -> str:
    if not tree or not tree.get("root"):
        return ""
    root = tree["root"]
    lines = [root]
    kids = list(siblings or [])
    for b in tree.get("branches") or []:
        if b not in kids:
            kids.append(b)
    if not kids and tree.get("branch"):
        kids = [tree["branch"]]
    for i, kid in enumerate(kids):
        prefix = " └── " if i == len(kids) - 1 else " ├── "
        lines.append(f"{prefix}{kid}")
    return "\n".join(lines)


def apply_family_tree_to_texture_map(
    texture_map: dict[str, Any],
    *,
    ocr_text: str = "",
    filename: str = "",
) -> dict[str, Any]:
    tm = dict(texture_map or {})
    tree = build_pattern_family_tree(tm, ocr_text=ocr_text, filename=filename)
    if tree.get("root") or tree.get("branch"):
        tm["pattern_family_tree"] = tree
    return tm

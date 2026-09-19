"""İndeks zenginleştirme — OCR/DNA/embedding yeniden çalıştırmadan opsiyonel alanlar."""

from __future__ import annotations

from typing import Any

from core.auto_tags import build_auto_tags
from core.pattern_family_tree import apply_family_tree_to_texture_map

# texture_map içinde opsiyonel, türetilmiş alanlar (yeniden analiz gerektirmez)
OPTIONAL_DERIVED_KEYS = (
    "auto_tags",
    "pattern_family_tree",
    "color_index",
    "duplicate_info",
)


def enrich_texture_map_if_missing(
    texture_map: dict[str, Any] | None,
    *,
    ocr_text: str = "",
    filename: str = "",
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Mevcut semantic/DNA/OCR'a dokunmadan eksik türetilmiş alanları doldur.
    Geriye dönük uyumlu — alan varsa korunur.
    """
    tm = dict(texture_map or {})
    prev = dict(existing or {})

    # Önce mevcut kayıttaki türetilmiş alanları koru
    for key in OPTIONAL_DERIVED_KEYS:
        if prev.get(key) and not tm.get(key):
            tm[key] = prev[key]

    if not tm.get("auto_tags"):
        tags = build_auto_tags(tm)
        if tags:
            tm["auto_tags"] = tags

    from core.color_index import apply_auto_color

    tm = apply_auto_color(tm)

    tree = tm.get("pattern_family_tree")
    if not (isinstance(tree, dict) and tree.get("root")):
        tm = apply_family_tree_to_texture_map(
            tm, ocr_text=ocr_text, filename=filename
        )

    return tm


def preserve_derived_fields(
    merged: dict[str, Any],
    existing: dict[str, Any] | None,
    *,
    reanalysed_semantic: bool = False,
    reanalysed_dna: bool = False,
) -> dict[str, Any]:
    """DNA/semantic yeniden hesaplanmadıysa türetilmiş alanları koru."""
    out = dict(merged or {})
    prev = dict(existing or {})
    if not reanalysed_semantic and not reanalysed_dna:
        for key in OPTIONAL_DERIVED_KEYS:
            if prev.get(key):
                out[key] = prev[key]
    return out

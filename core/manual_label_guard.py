"""Manuel kullanıcı etiketlerinin otomatik index tarafından ezilmesini engelle."""

from __future__ import annotations

import json
from typing import Any

MANUAL_TEXTURE_KEYS = frozenset(
    {
        "pattern_family",
        "pattern_subtype",
        "animal_print_type",
        "classification_confidence",
        "user_labeled",
        "user_label_source",
        "category_path",
        "manual_category_path",
        "category_confidence",
        "category_source",
        "category_aliases",
        "user_custom_tag",
        "user_similarity_tier",
        "user_tier_label",
    }
)


def parse_texture_map(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return dict(parsed) if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def is_manual_labeled(texture_map: dict[str, Any] | None) -> bool:
    tm = texture_map or {}
    if tm.get("user_labeled"):
        return True
    if str(tm.get("manual_category_path") or "").strip():
        return True
    if str(tm.get("category_source") or "") == "manual_user":
        return True
    return False


def merge_texture_map_preserve_manual(
    existing: dict[str, Any] | None,
    fresh: dict[str, Any] | None,
) -> dict[str, Any]:
    """Otomatik analiz + kullanıcı etiketini birleştir; manuel alanlar korunur."""
    old = parse_texture_map(existing)
    new = dict(fresh or {})
    if not is_manual_labeled(old):
        return new
    merged = dict(new)
    for key in MANUAL_TEXTURE_KEYS:
        if key in old and old[key] not in ("", None, [], {}):
            merged[key] = old[key]
    merged["user_labeled"] = True
    merged["user_label_source"] = old.get("user_label_source") or "manual_user"
    return merged

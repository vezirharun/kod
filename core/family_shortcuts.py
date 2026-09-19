"""Kullanıcının öğrettiği özel kategori kısayolları — diyalog butonları."""

from __future__ import annotations

import json
import re
from typing import Any

from core.db import Database
from core.textile_terms import (
    FAMILY_UI_CHOICES,
    normalize_turkish,
    resolve_user_family_label,
)

META_KEY = "user_family_shortcuts"
MAX_SHORTCUTS = 24
_LABEL_PARTS = re.compile(r"[/()]")


def _builtin_keys() -> set[str]:
    keys: set[str] = set()
    for label, family_id in FAMILY_UI_CHOICES:
        keys.add(normalize_turkish(label))
        keys.add(normalize_turkish(family_id))
    return keys


def _normalize_entry(raw: dict[str, Any]) -> dict[str, str] | None:
    label = str(raw.get("label") or "").strip()
    tag = str(raw.get("tag") or "").strip()
    pattern_family = str(raw.get("pattern_family") or "").strip()
    pattern_subtype = str(raw.get("pattern_subtype") or "").strip()
    if not label and tag:
        label = tag[:1].upper() + tag[1:]
    if not label:
        return None
    return {
        "label": label,
        "tag": tag,
        "tier_id": str(raw.get("tier_id") or "").strip(),
        "cluster_group": str(raw.get("cluster_group") or "").strip(),
        "pattern_family": pattern_family,
        "pattern_subtype": pattern_subtype,
    }


def load_shortcuts(db: Database) -> list[dict[str, str]]:
    """Kayıtlı özel kategori butonları."""
    raw = db.get_meta(META_KEY, "[]")
    try:
        items = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        items = []
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        entry = _normalize_entry(item)
        if not entry:
            continue
        key = normalize_turkish(entry["tag"] or entry["label"])
        if not key or key in seen or key in _builtin_keys():
            continue
        seen.add(key)
        out.append(entry)
    return out[:MAX_SHORTCUTS]


def _save_shortcuts(db: Database, shortcuts: list[dict[str, str]]) -> None:
    db.set_meta(META_KEY, json.dumps(shortcuts[:MAX_SHORTCUTS], ensure_ascii=False))


def _matches_builtin_button_label(tag: str) -> bool:
    """Yerleşik butonlarda zaten varsa tekrar ekleme."""
    norm = normalize_turkish(tag)
    if not norm:
        return True
    for label, family_id in FAMILY_UI_CHOICES:
        if norm == normalize_turkish(label) or norm == normalize_turkish(family_id):
            return True
        for part in _LABEL_PARTS.split(label):
            part_norm = normalize_turkish(part.strip())
            if part_norm and part_norm == norm:
                return True
    return False


def _is_redundant_builtin(tag: str, pattern_family: str) -> bool:
    return _matches_builtin_button_label(tag)


def remember_typed_category(
    db: Database,
    *,
    tag: str,
    pattern_family: str = "",
    pattern_subtype: str = "",
    tier_id: str = "",
    tier_label: str = "",
    cluster_group: str = "",
) -> dict[str, str] | None:
    """
    Yazılan yeni kategoriyi kalıcı kısayol olarak kaydet.
    Zaten varsa veya yerleşik butonla aynıysa dokunmaz.
    """
    tag = (tag or "").strip()
    display = (tier_label or "").strip()
    if not display and tag:
        display = tag[:1].upper() + tag[1:]
    if not display and not tier_id:
        return None
    if tag and _is_redundant_builtin(tag, pattern_family) and not tier_id:
        return None

    resolved = resolve_user_family_label(tag) if tag else {}
    family = pattern_family or resolved.get("pattern_family", "")
    subtype = pattern_subtype or resolved.get("pattern_subtype", "")
    key = normalize_turkish(tier_id or tag or display)

    shortcuts = load_shortcuts(db)
    for entry in shortcuts:
        entry_key = normalize_turkish(
            entry.get("tier_id") or entry.get("tag") or entry.get("label", "")
        )
        if entry_key == key:
            return entry

    entry = {
        "label": display,
        "tag": tag or display,
        "tier_id": tier_id,
        "cluster_group": cluster_group,
        "pattern_family": family,
        "pattern_subtype": subtype,
    }
    shortcuts.insert(0, entry)
    _save_shortcuts(db, shortcuts[:MAX_SHORTCUTS])
    return entry


def bootstrap_shortcuts_from_feedback(db: Database) -> int:
    """Geçmiş custom_tag kayıtlarından eksik kısayolları ekle."""
    added = 0
    try:
        with db.connect() as conn:
            rows = conn.execute(
                """
                SELECT label, MAX(created_at) AS last_used
                FROM user_feedback
                WHERE action='custom_tag' AND TRIM(label) != ''
                GROUP BY lower(label)
                ORDER BY last_used DESC
                LIMIT ?
                """,
                (MAX_SHORTCUTS,),
            ).fetchall()
    except Exception:
        return 0

    for row in rows:
        tag = str(row["label"] or "").strip()
        if not tag:
            continue
        resolved = resolve_user_family_label(tag)
        if remember_typed_category(
            db,
            tag=tag,
            pattern_family=resolved.get("pattern_family", ""),
            pattern_subtype=resolved.get("pattern_subtype", ""),
        ):
            added += 1
    return added

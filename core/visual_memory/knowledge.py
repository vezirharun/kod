"""Global Vezir Knowledge Pack — customer-independent.

Product knowledge only. Never mix customer Visual Memory, feedback labels,
or claimed_class. This module must not import the search/index stack.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

KNOWLEDGE_PACK_ID = "vezir_core"
KNOWLEDGE_PACK_VERSION = 2

_PACK_PATH = Path(__file__).resolve().parents[2] / "data" / "knowledge" / "vezir_core.json"

# Fallback slots if the compiled pack file is missing. Still not customer data.
_FALLBACK: dict[str, Any] = {
    "id": KNOWLEDGE_PACK_ID,
    "version": KNOWLEDGE_PACK_VERSION,
    "customer_memory": False,
    "objects": ("animal", "bird", "car", "flower", "logo", "vehicle"),
    "types": ("leopard", "tiger", "snake", "zebra", "rose", "daisy", "tulip"),
    "colors": ("red", "blue", "black", "white", "green"),
    "textures": ("scale", "spot", "stripe", "denim"),
    "pattern_families": ("animal_print", "floral", "geometric", "plain"),
    "brands": ("gucci", "versace", "togg"),
    "textile_concepts": ("repeat", "placement", "allover", "skin"),
    "composition": ("count", "orientation", "spatial"),
    "channels": (
        "visual",
        "ocr",
        "visual_dna",
        "filename",
        "metadata",
        "family",
        "texture",
    ),
    "unsupported": ("count", "orientation", "spatial", "claimed_class_auto"),
}


def _class_names(pack: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for key in ("objects", "types", "colors", "textures", "pattern_families", "brands"):
        for item in pack.get(key) or ():
            if isinstance(item, dict):
                names.add(str(item.get("id") or "").strip().lower())
                for a in item.get("aliases") or ():
                    names.add(str(a).strip().lower())
            else:
                names.add(str(item).strip().lower())
    names.discard("")
    return names


def load_knowledge_pack() -> dict[str, Any]:
    if _PACK_PATH.is_file():
        try:
            data = json.loads(_PACK_PATH.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        if isinstance(data, dict) and data.get("id") == KNOWLEDGE_PACK_ID:
            data.setdefault("customer_memory", False)
            return data
    return dict(_FALLBACK)


def knowledge_pack_info() -> dict[str, Any]:
    pack = load_knowledge_pack()

    def _count(key: str) -> int:
        val = pack.get(key) or ()
        if isinstance(val, dict):
            return len(val)
        return len(val)

    return {
        "id": str(pack.get("id") or KNOWLEDGE_PACK_ID),
        "version": int(pack.get("version") or KNOWLEDGE_PACK_VERSION),
        "customer_memory": bool(pack.get("customer_memory")),
        "source": str(pack.get("source") or "fallback"),
        "slot_counts": {
            k: _count(k)
            for k in (
                "objects",
                "types",
                "colors",
                "textures",
                "pattern_families",
                "brands",
                "textile_concepts",
                "composition",
                "channels",
                "unsupported",
            )
            if k in pack or k in _FALLBACK
        },
    }


def is_global_knowledge_class(name: str) -> bool:
    n = (name or "").strip().lower()
    if not n:
        return False
    return n in _class_names(load_knowledge_pack())


def is_unsupported_slot(name: str) -> bool:
    n = (name or "").strip().lower()
    pack = load_knowledge_pack()
    items = pack.get("unsupported") or ()
    names = []
    for item in items:
        if isinstance(item, dict):
            names.append(str(item.get("id") or "").lower())
        else:
            names.append(str(item).lower())
    return n in names
